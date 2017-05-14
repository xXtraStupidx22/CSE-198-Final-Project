library(kknn)
library(data.table)
library(FSelector)
library(plyr)
library(ROCR)

zip <- unzip("en.openfoodfacts.org.products.tsv.zip") # Need to set your working directory
trainSet <- fread(zip,sep='\t')

trainSet <- rename(trainSet,c("nutrition-score-uk_100g"="score"))
trainSet <- trainSet[!is.na(trainSet$score),]

trainSet <- as.data.frame(unclass(trainSet))

"trainSet$url <- NULL
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
trainSet$vitamin.b12_100g <- NULL
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
trainSet$manufacturing_places_tags <- as.numeric(trainSet$manufacturing_places_tags)"

pnns_groups_1 <- trainSet$pnns_groups_1
pnns_groups_2 <- trainSet$pnns_groups_2
energy_100g <- trainSet$energy_100g
fat_100g <- trainSet$fat_100g
saturated.fat_100g <- trainSet$saturated.fat_100g
monounsaturated.fat_100g <- trainSet$monounsaturated.fat_100g
carbohydrates_100g <- trainSet$carbohydrates_100g
sugars_100g <- trainSet$sugars_100g
proteins_100g <- trainSet$proteins_100g
salt_100g <- trainSet$salt_100g
sodium_100g <- trainSet$sodium_100g
vitamin.c_100g <- trainSet$vitamin.c_100g
calcium_100g <- trainSet$calcium_100g
score <- trainSet$score

trainSet <- data.frame(pnns_groups_1,pnns_groups_2,energy_100g,fat_100g,saturated.fat_100g,monounsaturated.fat_100g,carbohydrates_100g,sugars_100g,proteins_100g,salt_100g,sodium_100g,vitamin.c_100g,calcium_100g,score)


trainSet$score <- trainSet[,14] + 15
trainSet$score <- round(trainSet[,14] / 51)

trainSet[is.na(trainSet)] <- 0
#sampleTrain <- trainSet[sample(30000, 20000, replace=FALSE),]

folds <- cut(seq(1,nrow(trainSet)),breaks=10,labels = FALSE)

averageAcc <- 0
averageFM <- 0
averageRe <- 0
averagePre <- 0
averageAUC <- 0

count <- 1

jpeg('knn_roc.jpg')

for(i in 1:10) {
  testIndex <- which(folds == i,arr.ind = TRUE)
  testSet <- trainSet[testIndex,]
  trainSetIteration <- trainSet[-testIndex,]
  
  start.time <- Sys.time()
  print(start.time)

  kNearestNeighborsModel <- kknn(score~.,trainSetIteration,testSet,distance = 2,k = 8)
  fittedKNN <- fitted(kNearestNeighborsModel)
  cmKNN <- table(testSet$score,fittedKNN > 0.5) 
  
  end.time <- Sys.time()
  print(end.time)
  print(end.time-start.time)
  
  totalKNN <- sum(cmKNN)
  diagKNN <- diag(cmKNN)
  accuracyKNN <- sum(diagKNN)/totalKNN # Fraction of instances that are correctly classified
  precisionKNN <- diagKNN[2]/(diagKNN[2]+cmKNN[3]) # The fraction of correct predictions over a certain class
  recallKNN <- diagKNN[2]/(diagKNN[2]+cmKNN[2]) # Fraction of instances that were correctly predicted
  fmeasureKNN <- 2 * precisionKNN * recallKNN / (precisionKNN + recallKNN) # weighted average of precision and recall
  
  averageAcc <- averageAcc + accuracyKNN
  averagePre <- precisionKNN + averagePre
  averageRe <- averageRe + recallKNN
  averageFM <- averageFM + fmeasureKNN
  
  ROCRpred <- prediction(fittedKNN,testSet$score)
  ROCRperf <- performance(ROCRpred,'tpr','fpr')
  
  knnAUC <- performance(ROCRpred,measure='auc')
  averageAUC <- averageAUC + as.numeric(knnAUC@y.values)
  
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

print(paste0("KNN Average Accuracy: ",averageAcc/10))

print(paste0("KNN Average Precision: ",averagePre/10))

print(paste0("KNN Average Recall: ",averageRe/10))

print(paste0("KNN Average F-Measure: ",averageFM/10))

print(paste0("KNN Average AUC: ",averageAUC/10))


  
  
  