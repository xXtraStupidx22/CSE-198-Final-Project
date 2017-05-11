library(data.table)
library(FSelector)
library(plyr)

zip <- unzip("en.openfoodfacts.org.products.tsv.zip") # Need to set your working directory
trainSet <- fread(zip,sep='\t')

trainSet <- rename(trainSet,c("nutrition-score-uk_100g"="score"))
trainSet <- trainSet[!is.na(trainSet$score),]

trainSet <- as.data.frame(unclass(trainSet))

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

trainSet[is.na(trainSet)] <- 0

folds <- cut(seq(1,nrow(trainSet)),breaks=10,labels = FALSE)

averageAcc <- 0
averageFM <- 0
averageRe <- 0
averagePre <- 0

for(i in 1:10) {
  testIndex <- which(folds == i,arr.ind = TRUE)
  testSet <- trainSet[testIndex,]
  trainSetIteration <- trainSet[-testIndex,]
  
  logisticModel <- glm(score~.,data=trainSet,family = binomial)
  logisticPrediction <- predict(logisticModel,testSet,type='response')
  
  cmLR <- table(testSet$score,logisticPrediction > 0.5) # Confusion Matrix
  
  print("Logistic Regression Confusion Matrix: ")
  print(cmLR)
  #Now computing measurements i.e. accuracy,precision,f-measure, and recall
  
  totalLR <- sum(cmLR) # number of instances
  diagLR <- diag(cmLR) #Number of correctly classified instances per class
  accuracyLR <- sum(diagLR)/totalLR # Fraction of instances that are correctly classified
  precisionLR <- diagLR[2]/(diagLR[2]+cmLR[3]) # The fraction of correct predictions over a certain class
  recallLR <- diagLR[2]/(diagLR[2] + cmLR[2]) # Fraction of instances that were correctly predicted
  fmeasureLR <- 2 * precisionLR * recallLR / (precisionLR + recallLR) # weighted average of precision and recallfmeas
  
  averageAcc <- averageAcc + accuracyLR
  averagePre <- precisionLR + averagePre
  averageRe <- averageRe + recallLR
  averageFM <- averageFM + fmeasureLR
  
}

print("For 10 fold cross validation:")

print(paste0("Naive Bayes Average Accuracy: ",averageAcc/10))

print(paste0("Naive Bayes Average Precision: ",averagePre/10))

print(paste0("Naive Bayes Average Recall: ",averageRe/10))

print(paste0("Naive Bayes Average F-Measure: ",averageFM/10))