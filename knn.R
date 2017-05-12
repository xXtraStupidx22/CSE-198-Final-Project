library(kknn)
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
sampleTrain <- trainSet[sample(600, 500, replace=FALSE),]

#smp_size <- floor(0.75 * nrow(sampleTrain))
#set.seed(123)
#train_ind <- sample(seq_len(nrow(sampleTrain)), size = smp_size)

#train <- sampleTrain[train_ind, ]
#test <- sampleTrain[-train_ind, ]

folds <- cut(seq(1,nrow(sampleTrain)),breaks=10,labels = FALSE)

averageAcc <- 0
averageFM <- 0
averageRe <- 0
averagePre <- 0

for(i in 1:10) {
  testIndex <- which(folds == i,arr.ind = TRUE)
  testSet <- sampleTrain[testIndex,]
  trainSetIteration <- sampleTrain[-testIndex,]
  
  start.time <- Sys.time()
  print(start.time)

  kNearestNeighborsModel <- kknn(score~.,trainSetIteration,testSet,distance = 3)
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
  
}

print("For 10 fold cross validation:")

print(paste0("KNN Average Accuracy: ",averageAcc))

print(paste0("KNN Average Precision: ",averagePre))

print(paste0("KNN Average Recall: ",averageRe))

print(paste0("KNN Average F-Measure: ",averageFM))
  
  
  
  