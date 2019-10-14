


# ____________________
# ~~~~~~~~~~~~~~~~~~~~
# set up environment
#
rm(list = ls())
#setwd(##@filepathhere~##)

# source functions script
source('func_ModelFunctions')

filepath_train <- 'Data/train.csv'
filepath_test <- 'Data/test.csv'

glb_seed <- 5059
#glb_debug_dataProportion <- 0.1
glb_debug_dataProportion <- 0.001




# ____________________
# ~~~~~~~~~~~~~~~~~~~~
# load packages
#
libs <- c("plyr", "dplyr","readr","caret", "Metrics", "miscTools", "glmnet", "caretEnsemble", "rpart", 'parallel', 'iterators', 'doParallel')
func_loadLibraries(libs)

#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load Data & Initialise 
#
train <- fread(filepath_train)
test <- fread(filepath_test)

test$target <- NA
data <- rbind(train, test)
rm(train,test);gc()

# create features for model
data[, amount_nas := rowSums(data == -1, na.rm = T)]
data[, high_nas := ifelse(amount_nas>4,1,0)]
data[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
data[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
data[, ps_custom_bin := ifelse( ps_ind_05_cat == -1 |  ps_ind_05_cat == -2 | ps_car_07_cat == -1 | ps_car_11_cat == 41, 1, 0) ]
#unique(data$ps_custom_bin==1)

# initialise variables for XGBoost
cvFolds <- createFolds(data$target[!is.na(data$target)], k=5, list=TRUE, returnTrain=FALSE)
varnames <- setdiff(colnames(data), c("id", "target"))
train_sparse <- Matrix(as.matrix(data[!is.na(target), varnames, with=F]), sparse=TRUE)
test_sparse <- Matrix(as.matrix(data[is.na(target), varnames, with=F]), sparse=TRUE)
y_train <- data[!is.na(target),target]
test_ids <- data[is.na(target),id]
dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse)



