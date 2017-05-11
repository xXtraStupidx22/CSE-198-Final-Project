library(data.table)
library(FSelector)
library(plyr)
library(e1071)

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
trainSet$`vitamin-b12_100g`<- NULL
trainSet$categories <- NULL
trainSet$categories_en <- NULL
trainSet$product_name <- NULL
trainSet$generic_name <- NULL
trainSet$categories_tags <- NULL
trainSet$code <- NULL
trainSet$ingredients_text <- NULL
trainSet$additives <- NULL
trainSet$additives_en <- NULL
trainSet$additives_tags <- NULL
trainSet$brands <- NULL
trainSet$brands_tags <- NULL
trainSet$creator <- NULL
trainSet$created_t <- NULL
trainSet$packaging <- NULL
trainSet$packaging_tags <- NULL
trainSet$origins <- NULL
trainSet$origins_tags <- NULL

trainSet$score <- trainSet[,126] + 15
trainSet$score <- round(trainSet[,126] / 51)

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
  
  bayesModel <- naiveBayes(as.factor(score) ~ .,data = trainSetIteration)
  temp <- testSet
  predictionsBayes <- predict(bayesModel,as.data.frame(temp[,-126]))

  cmBayesProject <- table(testSet$score,predictionsBayes)
  
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

#ROCRpred <- prediction(logisticPrediction,testSet$survived)
#ROCRperf <- performance(ROCRpred,'tpr','fpr')


