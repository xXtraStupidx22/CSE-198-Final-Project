library(data.table)
library(FSelector)
library(plyr)

zip <- unzip("en.openfoodfacts.org.products.tsv.zip")
trainSet <- fread(zip,sep='\t')

trainSet <- rename(trainSet,c("nutrition-score-uk_100g"="score"))
trainSet$url <- NULL
trainSet$last_modified_datetime <- NULL
trainSet$image_url <- NULL
trainSet$image_small_url <- NULL
trainSet$created_datetime <- NULL

sampleTrain <- trainSet[sample(900, 600, replace=FALSE),]
sampleTrain[is.na(sampleTrain)] <- 0

weights <- information.gain(score~.,sampleTrain)

