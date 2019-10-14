
rm(list=ls())

#setwd(##@filepathhere~##)

filepath_train <- 'Data/train.csv'
filepath_test <- 'Data/test.csv'

library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)
library(MLmetrics)
library(caret)

set.seed(5059)

cat("Functions")
xgb_normalizedgini <- function(preds, dtrain){
  actual <- getinfo(dtrain, "label")
  score <- NormalizedGini(preds,actual)
  return(list(metric = "NormalizedGini", value = score))
}

train <- fread(filepath_train, sep=",")
test <- fread(filepath_test, sep=",")
is.data.frame(train)

base_train_withHighPredictors <- fread('predTrain/Train_IndividualModels.csv', sep=",")
base_train_withHighPredictors <- as_tibble(base_train_withHighPredictors)

base_train_withHighestPredictor <- base_train_withHighPredictors[,1:8]

base_train <- base_train_withHighestPredictor[, 1:7]
#head(base_train)




# now apply same setup for test predictions
test_GB1 <- fread('predTest/GBPred1.csv', sep=",")
test_GB2 <- fread('predTest/GBPred2.csv', sep=",")
test_GB3 <- fread('predTest/GBPred3.csv', sep=",")
test_GAM <- fread('predTest/GAMPred.csv', sep=",")
test_LOG <- fread('predTest/LogPred.csv', sep=",")

base_test <- as_tibble(test_GB1)
base_test <- base_test %>% rename(  GB1 = target)
GB2 <- test_GB2$target
base_test <- as_tibble(cbind(base_test, GB2))
GB3 <- test_GB3$target
base_test <- as_tibble(cbind(base_test, GB3))
GAM <- test_GAM$target
base_test <- as_tibble(cbind(base_test, GAM))
LOG <- test_LOG$target
base_test <- as_tibble(cbind(base_test, LOG))

head(base_test[,-1])

cor(base_test[,-1])


ps_car_13 <- test$ps_car_13
base_test_withHighestPredictor <- as_tibble(cbind(base_test, ps_car_13))

ps_reg_03 <- test$ps_reg_03
base_test_withHighPredictors <- as_tibble(cbind(base_test_withHighestPredictor, ps_reg_03))


base_test$target <- NA
base <- rbind(base_train, base_test)
rm(base_train, base_test);gc()
head(base)
is.data.frame(base)


base_test_withHighestPredictor$target <- NA
base_withHighestPredictor <- rbind(base_train_withHighestPredictor, base_test_withHighestPredictor)
rm(base_train_withHighestPredictor, base_test_withHighestPredictor);gc()
head(base_withHighestPredictor)


base_test_withHighPredictors$target <- NA
base_withHighPredictors <- rbind(base_train_withHighPredictors, base_test_withHighPredictors)
rm(base_train_withHighPredictors, base_test_withHighPredictors);gc()
head(base_withHighPredictors)


# reference from original XGBoost
# test$target <- NA
# data <- rbind(train, test)
#rm(train,test);gc()


ensembledata <- base
ensembledata <- base_withHighestPredictor
ensembledata <- base_withHighPredictors

setattr(ensembledata, "class", c("tbl", "tbl_df", "data.frame"))
ensembledata <- as.data.table(ensembledata)


cat("Prepare for xgb")
cvFolds <- createFolds(ensembledata$target[!is.na(ensembledata$target)], k=5, list=TRUE, returnTrain=FALSE)
varnames <- setdiff(colnames(ensembledata), c("id", "target"))
train_sparse <- Matrix(as.matrix(ensembledata[!is.na(target), varnames, with=F]), sparse=TRUE)
test_sparse <- Matrix(as.matrix(ensembledata[is.na(target), varnames, with=F]), sparse=TRUE)
y_train <- ensembledata[!is.na(target),target]
test_ids <- ensembledata[is.na(target),id]
dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse)
#nrow(dtrain)
#nrow(dtest)


param <- list(booster="gbtree",
              objective="binary:logistic",
              eta = 0.02 ,
              gamma = 1,
              max_depth = 6,
              min_child_weight = 1,
              subsample = 0.8,
              colsample_bytree = 0.8
)

cat("xgb cross-validation")
xgb_cv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 500,
                 feval = xgb_normalizedgini,
                 maximize = TRUE,
                 prediction = TRUE,
                 folds = cvFolds,
                 print_every_n = 50,
                 early_stopping_round = 30)

best_iter <- xgb_cv$best_iteration

cat("xgb model")
xgb_model <- xgb.train(data = dtrain,
                       params = param,
                       nrounds = best_iter,
                       feval = xgb_normalizedgini,
                       maximize = TRUE,
                       watchlist = list(train = dtrain),
                       verbose = 1,
                       print_every_n = 50
)


# Feature       Gain      Cover  Frequency Importance
# 1:       GB3 0.49543391 0.39457890 0.21155404 0.49543391
# 2:       GB2 0.14352091 0.16514523 0.15686275 0.14352091
# 3:       GB1 0.13749400 0.13696095 0.16541079 0.13749400
# 4:       LOG 0.07607745 0.11101206 0.13591800 0.07607745
# 5:       GAM 0.06407870 0.07415137 0.12874737 0.06407870
# 6: ps_car_13 0.04652746 0.06727309 0.11076001 0.04652746
# 7: ps_reg_03 0.03686758 0.05087839 0.09074704 0.03686758



#cat("Feature importance")
names <- dimnames(train_sparse)[[2]]
importance_matrix <- xgb.importance(names, model=xgb_model)
xgb.plot.importance(importance_matrix)

#importance_matrix[importance_matrix[,1]=="ps_custom_bin",]

cat("Predict and output csv")
preds <- data.table(id=test_ids, target=predict(xgb_model,dtest))


write.table(preds, "xg2featuresensemble_submission.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)



# ~~~~~~~~~~~~~
# could also just do a glm
data_train <- base[!is.na(target),]
data_test <- base[is.na(target),]
nrow(data_train)
nrow(data_test)


control <- trainControl(method = 'cv', number = 5, allowParallel =  TRUE)
set.seed(5059)
fit.glm <- train(target~., data = data_train, method='rf', metric = "ROC", trControl = control, verbose = FALSE)






