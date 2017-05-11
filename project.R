library(data.table)
library(FSelector)
library(plyr)
library(e1071)

zip <- unzip("en.openfoodfacts.org.products.tsv.zip") # Need to set your working directory
trainSet <- fread(zip,sep='\t')

trainSet <- rename(trainSet,c("nutrition-score-uk_100g"="score"))
trainSet <- trainSet[!is.na(trainSet$score),]

trainSet <- as.data.frame(unclass(trainSet))
#trainSet[sapply(trainSet, is.character)] <- lapply(trainSet[sapply(trainSet, is.character)], as.factor)

trainSet$url <- NULL
trainSet$last_modified_datetime <- NULL
trainSet$image_url <- NULL
trainSet$image_small_url <- NULL
trainSet$created_datetime <- NULL
trainSet$`nutrition-score-fr_100g` <- NULL
trainSet$states <- NULL
trainSet$states_tags <- NULL
trainSet$states_en <- NULL
trainSet$nutrition_grade_fr <- NULL
trainSet$nutrition_grade_uk <- NULL
trainSet$cities <- NULL
trainSet$cities_tags <- NULL
trainSet$countries <- NULL
trainSet$countries_en <- NULL
trainSet$countries_tags <- NULL

trainSet$score <- trainSet[,143] + 15
trainSet$score <- round(trainSet[,143] / 51)

sampleTrain <- trainSet[sample(14000, 10000, replace=FALSE),]
sampleTrain[is.na(sampleTrain)] <- 0

#weights <- information.gain(score~.,sampleTrain)

folds <- cut(seq(1,nrow(sampleTrain)),breaks=10,labels = FALSE)

averageAcc <- 0
averageFM <- 0
averageRe <- 0
averagePre <- 0

for(i in 1:10) {
  testIndex <- which(folds == i,arr.ind = TRUE)
  testSet <- sampleTrain[testIndex,]
  trainSetIteration <- sampleTrain[-testIndex,]
  
  #trainSetIteration$score <- as.factor(trainSetIteration$score)
  
  #logisticModel <- glm(score~.,data=trainSetIteration,family = binomial)
  #logisticPrediction <- predict(logisticModel,testSet,type='response')
  
  bayesModel <- naiveBayes(as.factor(score) ~ .,data = trainSetIteration[,-1])
  temp <- testSet[,-1]
  predictionsBayes <- predict(bayesModel,as.data.frame(temp[,-143]))

  cmBayesProject <- table(testSet$score,predictionsBayes)
  
  #cmLR <- table(testSet$score,logisticPrediction > 0.5) # Confusion Matrix
  
  totalBayes <- sum(cmBayesProject)
  diagBayes <- diag(cmBayesProject)
  accuracyBayes <- sum(diagBayes)/totalBayes # Fraction of instances that are correctly classified
  precisionBayes <- diagBayes[2]/(diagBayes[2]+cmBayesProject[3]) # The fraction of correct predictions over a certain class
  recallBayes <- diagBayes[2]/(diagBayes[2]+cmBayesProject[2]) # Fraction of instances that were correctly predicted
  fmeasureBayes <- 2 * precisionBayes * recallBayes / (precisionBayes + recallBayes) # weighted average of precision and recallfmeas
  
  averageAcc <- averageAcc + accuracyBayes
  averagePre <- precisionBayes + averagePre
  averageRe <- averageRe + recallBayes
  averageFM <- averageFM + fmeasureBayes
  
}


print("For 10 fold cross validation:")

print(paste0("Naive Bayes Average Accuracy: ",averageAcc/10))

print(paste0("Naive Bayes Average Precision: ",averagePre/10))

print(paste0("Naive Bayes Average Recall: ",averageRe/10))

print(paste0("Naive Bayes Average F-Measure: ",averageFM/10))

