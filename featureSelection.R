library(data.table)
library(FSelector)
library(plyr)
library(e1071)

zip <- unzip("en.openfoodfacts.org.products.tsv.zip") # Need to set your working directory
trainSet <- fread(zip,sep='\t')

trainSet <- rename(trainSet,c("nutrition-score-uk_100g"="score"))
trainSet <- trainSet[!is.na(trainSet$score),]
trainSet[is.na(trainSet)] <- 0

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
trainSet$last_modified_t <- NULL
trainSet$allergens_en <- NULL
trainSet$no_nutriments <- NULL
trainSet$additives_n <- NULL
trainSet$ingredients_from_palm_oil <- NULL
trainSet$ingredients_from_palm_oil_n <- NULL
trainSet$ingredients_that_may_be_from_palm_oil <- NULL
trainSet$ingredients_that_may_be_from_palm_oil_n <- NULL
trainSet$energy.from.fat_100g <- NULL

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


#sampleTrain <- trainSet[sample(60000, 50000, replace=FALSE),]

weights <- information.gain(score~.,trainSet)

#weights <- chi.squared(score~.,sampleTrain)




