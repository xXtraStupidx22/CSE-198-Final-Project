library(data.table)
library(FSelector)
library(plyr)
library(e1071)
library(ROCR)

zip <- unzip("en.openfoodfacts.org.products.tsv.zip") # Need to set your working directory
trainSet <- fread(zip,sep='\t')

trainSet <- rename(trainSet,c("nutrition-score-uk_100g"="score"))
trainSet <- trainSet[!is.na(trainSet$score),]

trainSet <- as.data.frame(unclass(trainSet))

trainSet$url <- NULL
trainSet$quantity <- NULL
trainSet$manufacturing_places <- NULL
trainSet$manufacturing_places_tags <- NULL
trainSet$labels <- NULL
trainSet$labels_tags <- NULL
trainSet$labels_en <- NULL
trainSet$emb_codes <- NULL
trainSet$emb_codes_tags <- NULL
trainSet$first_packaging_code_geo <- NULL
trainSet$purchase_places <- NULL
trainSet$stores <- NULL
trainSet$allergens <- NULL
trainSet$traces <- NULL
trainSet$traces_en <- NULL
trainSet$traces_tags <- NULL
trainSet$serving_size <- NULL
trainSet$ingredients_that_may_be_from_palm_oil_tags <- NULL
trainSet$main_category <- NULL
trainSet$main_category_en <- NULL
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

trainSet$score <- trainSet[,107] + 15
trainSet$score <- round(trainSet[,107] / 51)

trainSet[is.na(trainSet)] <- 0

folds <- cut(seq(1,nrow(trainSet)),breaks=10,labels = FALSE)

averageAcc <- 0
averageFM <- 0
averageRe <- 0
averagePre <- 0
averageAUC <- 0

count <- 1

jpeg('bayes_roc.jpg')

for(i in 1:10) {
  testIndex <- which(folds == i,arr.ind = TRUE)
  testSet <- trainSet[testIndex,]
  trainSetIteration <- trainSet[-testIndex,]
  
  bayesModel <- naiveBayes(as.factor(score) ~ .,data = trainSetIteration)
  temp <- testSet
  predictionsBayes <- predict(bayesModel,as.data.frame(temp[,-107]))

  cmBayesProject <- table(as.numeric(testSet$score),as.numeric(predictionsBayes))
  
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
  
  ROCRpred <- prediction(as.numeric(predictionsBayes),testSet$score)
  ROCRperf <- performance(ROCRpred,'tpr','fpr')
  
  bayesAUC <- performance(ROCRpred,measure='auc')
  averageAUC <- averageAUC + as.numeric(bayesAUC@y.values)
  
  if(count == 1) {
    plot(ROCRperf,col=count,main="ROC Curve")
  }
  else {
    plot(ROCRperf,col=count,add=TRUE)
  }
  
  count <- count + 1
  
}

dev.off()

print("For 10 fold cross validation:")

print(paste0("Naive Bayes Average Accuracy: ",averageAcc/10))

print(paste0("Naive Bayes Average Precision: ",averagePre/10))

print(paste0("Naive Bayes Average Recall: ",averageRe/10))

print(paste0("Naive Bayes Average F-Measure: ",averageFM/10))

print(paste0("Naive Bayes Average AUC: ",averageAUC/10))

