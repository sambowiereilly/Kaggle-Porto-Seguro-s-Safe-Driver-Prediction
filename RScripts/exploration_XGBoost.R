#Scripts

#setwd(##@filepathhere@##)
filepath_train <- 'Data/train.csv'
filepath_test <- 'Data/test.csv'


#install.packages('xgboost')
#install.packages('MLmetrics')


library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)
library(MLmetrics)

cat("Init")
set.seed(5059)
PATH <- "../input/"

cat("Functions")
xgb_normalizedgini <- function(preds, dtrain){
  actual <- getinfo(dtrain, "label")
  score <- NormalizedGini(preds,actual)
  return(list(metric = "NormalizedGini", value = score))
}

cat("Load data")
train <- fread(filepath_train, sep=",", na.strings = "", stringsAsFactors=T)
test <- fread(filepath_test, sep=",", na.strings = "", stringsAsFactors=T)

cat("Combine train and test files")
test$target <- NA
data <- rbind(train, test)
rm(train,test);gc()

cat("Feature engineering")
data[, amount_nas := rowSums(data == -1, na.rm = T)]
data[, high_nas := ifelse(amount_nas>4,1,0)]
data[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
data[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]

data[, ps_custom_bin := ifelse( ps_ind_05_cat == -1 |  ps_ind_05_cat == -2 | ps_car_07_cat == -1 | ps_car_11_cat == 41, 1, 0) ]


#unique(data$ps_custom_bin==1)



#data[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]

cat("Prepare for xgb")
cvFolds <- createFolds(data$target[!is.na(data$target)], k=5, list=TRUE, returnTrain=FALSE)
varnames <- setdiff(colnames(data), c("id", "target"))
train_sparse <- Matrix(as.matrix(data[!is.na(target), varnames, with=F]), sparse=TRUE)
test_sparse <- Matrix(as.matrix(data[is.na(target), varnames, with=F]), sparse=TRUE)
y_train <- data[!is.na(target),target]
test_ids <- data[is.na(target),id]
dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse)


# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Further refinements


# to optimise:
# eta = 0.02 { between 0.01 & 0.03 }
# subsample = { between 0.6 - 0.8 }
# max_depth = {  between 6 or 8 }
# colsample_bytree  = { between 0.6 or 0.8}
nrow_gridOptimise = 36
optimiseGrid <- matrix( data = 0, nrow = nrow_gridOptimise, ncol = 9 )
# set up optimisation grid
gridIDs <- seq(from = 1, to = 36, by = 1)
gridIterations <- 1000
gridETA <- c(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03 )
gridSubSample <- c(0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8)
gridMax_depth <- c(6, 6, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8)
gridColsample_byTree <- c(6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8)
# set up grid
optimiseGrid[,1] <- gridIDs
optimiseGrid[,2] <- gridIterations
optimiseGrid[,3] <- gridETA
optimiseGrid[,4] <- gridSubSample
optimiseGrid[,5] <- gridMax_depth 
optimiseGrid[,6] <-gridColsample_byTree
# now perform loop through all those Grids
for(i in 1:nrow_gridOptimise){
  #cat("Params for xgb")
  param <- list(booster="gbtree",
                objective="binary:logistic",
                eta = optimiseGrid[i,3] ,
                gamma = 1,
                max_depth = optimiseGrid[i,5] ,
                min_child_weight = 1,
                subsample = optimiseGrid[i,4],
                colsample_bytree = optimiseGrid[i,6]
  )
  #cat("xgb cross-validation, uncomment when running locally")
  xgb_cv <- xgb.cv(data = dtrain,
                   params = param,
                   nrounds = optimiseGrid[i,2],
                   feval = xgb_normalizedgini,
                   maximize = TRUE,
                   prediction = TRUE,
                   folds = cvFolds,
                   print_every_n = 50,
                   early_stopping_round = 30)
  #print(xgb_cv)
  best_iter <- xgb_cv$best_iteration
  optimiseGrid[i,7] <- as.numeric(xgb_cv$best_iteration)
  optimiseGrid[i,8] <- as.numeric(xgb_cv$evaluation_log[best_iter, 2])
  optimiseGrid[i,9] <- as.numeric(xgb_cv$evaluation_log[best_iter, 4])
}
# now check 
optimiseGrid





# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Original Split Down



optimiseGrid <- matrix( data = 0, nrow = 16, ncol = 7 )
optimisePredictions <- matrix( data = 0, nrow = nrow(dtest), ncol = nrow(optimiseGrid)+1 )
optimiseResults <- matrix( data = 0, nrow = nrow(optimiseGrid), ncol = 2 )


gridIDs <- c(1,2,3,4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16)
gridIterations <- 1000
gridETA <- c(0.02, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.25)

gridSubSample <- c(0.4, 0.6, 0.8, 0.9, 0.4, 0.6, 0.8, 0.9, 0.4, 0.6, 0.8, 0.9, 0.4, 0.6, 0.8, 0.9)



optimiseGrid[,1] <- gridIDs
optimiseGrid[,2] <- gridIterations
optimiseGrid[,3] <- gridETA
optimiseGrid[,4] <- gridSubSample




optimiseResults[,1] <- gridIDs

optimisePredictions[,1] <- test_ids


# param <- list(booster="gbtree",
#               objective="binary:logistic",
#               eta = 0.02,
#               gamma = 1,
#               max_depth = 6,
#               min_child_weight = 1,
#               subsample = 0.8,
#               colsample_bytree = 0.8
# )


for(i in 1:nrow(optimiseGrid)){
  
  #cat("Params for xgb")
  param <- list(booster="gbtree",
                objective="binary:logistic",
                eta = optimiseGrid[i,3] ,
                gamma = 1,
                max_depth = 6,
                min_child_weight = 1,
                subsample = optimiseGrid[,4],
                colsample_bytree = 0.8
  )
  
  cat("xgb cross-validation, uncomment when running locally")
  xgb_cv <- xgb.cv(data = dtrain,
                   params = param,
                   nrounds = optimiseGrid[i,2],
                   feval = xgb_normalizedgini,
                   maximize = TRUE,
                   prediction = TRUE,
                   folds = cvFolds,
                   print_every_n = 50,
                   early_stopping_round = 30)
  
  #print(xgb_cv)
  #best_iter <- 420
  best_iter <- xgb_cv$best_iteration
  
  optimiseGrid[i,5] <- as.numeric(xgb_cv$best_iteration)
  optimiseGrid[i,6] <- as.numeric(xgb_cv$evaluation_log[best_iter, 2])
  optimiseGrid[i,7] <- as.numeric(xgb_cv$evaluation_log[best_iter, 4])
  
  
}


#best_iter <- 100

# cat("xgb model")
# xgb_model <- xgb.train(data = dtrain,
#                        params = param,
#                        nrounds = best_iter,
#                        feval = xgb_normalizedgini,
#                        maximize = TRUE,
#                        watchlist = list(train = dtrain),
#                        verbose = 1,
#                        print_every_n = 50
# )


#optimiseResults[i,2] <- xgb_model$evaluation_log[100,2]

#cat("Feature importance")
# names <- dimnames(train_sparse)[[2]]
# importance_matrix <- xgb.importance(names, model=xgb_model)
# xgb.plot.importance(importance_matrix)
#  
# #importance_matrix[importance_matrix[,1]=="ps_custom_bin",]
# 
# cat("Predict and output csv")
# preds <- data.table(id=test_ids, target=predict(xgb_model,dtest))
# 
# optimisePredictions[,i] <- preds$target


write.table(preds, "submission.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)





