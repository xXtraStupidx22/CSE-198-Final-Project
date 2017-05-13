library(data.table)
library(FSelector)
library(plyr)
library(randomForest)

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
trainSet$main_category <- as.numeric(trainSet$main_category)
trainSet$main_category_en <- as.numeric(trainSet$main_category_en)
trainSet$pnns_groups_1 <- as.numeric(trainSet$pnns_groups_1)
trainSet$pnns_groups_2 <- as.numeric(trainSet$pnns_groups_2)
trainSet$ingredients_that_may_be_from_palm_oil_tags <- as.numeric(trainSet$ingredients_that_may_be_from_palm_oil_tags)
trainSet$ingredients_from_palm_oil_tags <- as.numeric(trainSet$ingredients_from_palm_oil_tags)
trainSet$serving_size <- as.numeric(trainSet$serving_size)
trainSet$traces_en <- as.numeric(trainSet$traces_en)
trainSet$traces_tags <- as.numeric(trainSet$traces_tags)
trainSet$traces <- as.numeric(trainSet$traces)
trainSet$allergens <- as.numeric(trainSet$allergens)
trainSet$stores <- as.numeric(trainSet$stores)
trainSet$purchase_places <- as.numeric(trainSet$purchase_places)
trainSet$first_packaging_code_geo <- as.numeric(trainSet$first_packaging_code_geo)
trainSet$emb_codes <- as.numeric(trainSet$emb_codes)
trainSet$emb_codes_tags <- as.numeric(trainSet$emb_codes_tags)
trainSet$labels <- as.numeric(trainSet$labels)
trainSet$labels_tags <- as.numeric(trainSet$labels_tags)
trainSet$labels_en <- as.numeric(trainSet$labels_en)
trainSet$manufacturing_places <- as.numeric(trainSet$manufacturing_places)
trainSet$quantity <- as.numeric(trainSet$quantity)
trainSet$manufacturing_places_tags <- as.numeric(trainSet$manufacturing_places_tags)

names(trainSet) <- make.names(names(trainSet))

trainSet$nutrition.score.fr_100g <- NULL
trainSet$vitamin.b12_100g <- NULL

trainSet$score <- trainSet[,124] + 15
trainSet$score <- round(trainSet[,124] / 51)
trainSet[is.na(trainSet)] <- 0

#smp_size <- floor(0.75 * nrow(trainSet))

#set.seed(123)
#train_ind <- sample(seq_len(nrow(trainSet)), size = smp_size)

#sampleTrain <- trainSet[train_ind, ]
#testSet <- trainSet[-train_ind, ]

#start.time <- Sys.time()
#print(start.time)

#forestModel <- randomForest(score ~ ., data = sampleTrain)
#predicted <- predict(forestModel, newdata=testSet[,-124])

#end.time <- Sys.time()
#print(end.time-start.time)

folds <- cut(seq(1,nrow(trainSet)),breaks=10,labels = FALSE)

averageAcc <- 0
averageFM <- 0
averageRe <- 0
averagePre <- 0
averageAUC <- 0

count <- 1

jpeg('randomForest.jpg')

for(i in 1:10) {
  testIndex <- which(folds == i,arr.ind = TRUE)
  testSet <- trainSet[testIndex,]
  trainSetIteration <- trainSet[-testIndex,]
  
  start.time <- Sys.time()
  print(start.time)
  
  forestModel <- randomForest(score ~ ., data = trainSetIteration)
  predicted <- predict(forestModel, newdata=testSet[,-124])
  cmRF <- table(testSet$score,predicted > 0.5) 
  
  end.time <- Sys.time()
  print(end.time)
  print(end.time-start.time)
  
  #totalRF <- sum(cmRF)
  #diagRF <- diag(cmRF)
  #accuracyRF <- sum(diagRF)/totalRF # Fraction of instances that are correctly classified
  #precisionRF <- diagRF[2]/(diagRF[2]+cmRF[3]) # The fraction of correct predictions over a certain class
  #recallRF <- diagKNN[2]/(diagRF[2]+cmRF[2]) # Fraction of instances that were correctly predicted
  #fmeasureRF <- 2 * precisionRF * recallRF / (precisionRF + recallRF) # weighted average of precision and recall
  
  #averageAcc <- averageAcc + accuracyRF
 # averagePre <- precisionRF + averagePre
#  averageRe <- averageRe + recallRF
 # averageFM <- averageFM + fmeasureRF
  
  ROCRpred <- prediction(predicted,testSet$score)
  ROCRperf <- performance(ROCRpred,'tpr','fpr')
  
  rfAUC <- performance(ROCRpred,measure='auc')
  averageAUC <- averageAUC + as.numeric(rfAUC@y.values)
  
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

#print(paste0("RF Average Accuracy: ",averageAcc/10))

#print(paste0("RF Average Precision: ",averagePre/10))

#print(paste0("RF Average Recall: ",averageRe/10))

#print(paste0("RF Average F-Measure: ",averageFM/10))

print(paste0("RF Average AUC: ",averageAUC/10))

