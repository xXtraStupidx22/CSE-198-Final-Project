library(data.table)
library(FSelector)
library(plyr)
library(randomForest)
library(ROCR)

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
trainSet$nutrition.score.fr_100g <- NULL
trainSet$vitamin.b12_100g <- NULL
trainSet$labels <- NULL
trainSet$labels_tags <- NULL
trainSet$labels_en <- NULL
trainSet$emb_codes <- NULL
trainSet$emb_codes_tags <- NULL
trainSet$first_packaging_code_geo <- NULL
trainSet$manufacturing_places <- NULL
trainSet$main_category <- NULL
trainSet$main_category_en <- NULL
trainSet$stores <- NULL
trainSet$allergens <- NULL
trainSet$manufacturing_places_tags <- NULL
trainSet$traces <- NULL
trainSet$traces_en <- NULL
trainSet$traces_tags <- NULL
trainSet$ingredients_that_may_be_from_palm_oil_tags <- NULL
trainSet$serving_size <- NULL
trainSet$purchase_places <- NULL
trainSet$quantity <- NULL

names(trainSet) <- make.names(names(trainSet))

trainSet$score <- trainSet[,105] + 15
trainSet$score <- round(trainSet[,105] / 51)
trainSet[is.na(trainSet)] <- 0

jpeg('randomForest.jpg')
  
start.time <- Sys.time()
print(start.time)

smp_size <- floor(0.75 * nrow(trainSet))

set.seed(123)
train_ind <- sample(seq_len(nrow(trainSet)), size = smp_size)

testSet <- trainSet[-train_ind, ]
trainSet <- trainSet[train_ind, ]
  
forestModel <- randomForest(score ~ ., data = trainSet)
predicted <- predict(forestModel, newdata=testSet[,-105])
cmRF <- table(testSet$score,predicted > 0.5) 
  
end.time <- Sys.time()
print(end.time)
print(end.time-start.time)
  
totalRF <- sum(cmRF)
diagRF <- diag(cmRF)
accuracyRF <- sum(diagRF)/totalRF # Fraction of instances that are correctly classified
precisionRF <- diagRF[2]/(diagRF[2]+cmRF[3]) # The fraction of correct predictions over a certain class
recallRF <- diagRF[2]/(diagRF[2]+cmRF[2]) # Fraction of instances that were correctly predicted
fmeasureRF <- 2 * precisionRF * recallRF / (precisionRF + recallRF) # weighted average of precision and recall
  
ROCRpred <- prediction(predicted,testSet$score)
ROCRperf <- performance(ROCRpred,'tpr','fpr')
  
rfAUC <- performance(ROCRpred,measure='auc')

plot(ROCRperf,col=1,main="ROC Curve")

dev.off()

print(paste0("RF Accuracy: ",accuracyRF))
print(paste0("RF Precision: ",precisionRF))
print(paste0("RF Recall: ",recallRF))
print(paste0("RF F-Measure: ",fmeasureRF))
print(paste0("RF AUC: ",rfAUC@y.values))

