library(data.table)

zip <- unzip("en.openfoodfacts.org.products.tsv.zip")
temp <- fread(zip,sep='\t')