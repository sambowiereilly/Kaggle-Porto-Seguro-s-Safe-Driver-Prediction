rm(list=ls())

#setwd(##@filepathhere@##)

# --------------------------------Import required Packages ------------------------------------

# Visualisation
library(ggplot2)
library(grid)

# Data handling
library(tidyverse)
library('data.table')
library(tfestimators)

# Model Creation
library(caret)
library(xgboost)
library(MLmetrics)
library(randomForest)
library(nnet)
library(mgcv)

# -------------------------------------- Read in Data ------------------------------------------
# Read in data
train <- as.data.table(fread('Data/train.csv'))
test <- as.data.table(fread('Data/test.csv'))
response <- "target"
# ------------------------------- Exploratory Data Analysis ------------------------------------

# Count missing values
perc_na <- function(data, na_val) {
  return(sum(data == na_val)/(nrow(data)*ncol(data))*100)
}

perc_na(train, -1)
perc_na(test, -1)

claimproportion <- function(data) {
  return(sum(data$target == 1)/nrow(data))
}

claimproportion(train)


boxplot(train$ps_car_13, test$ps_car_13,
        main = "Box Plot ; ps_car_13",
        ylab = "Differential of Values", names = c('train', 'test') )


boxplot(train$ps_reg_03, test$ps_reg_03,
        main = "Box Plot ; ps_reg_03",
        ylab = "Differential of Values", names = c('train', 'test') )


boxplot(train$ps_car_03_cat, test$ps_car_03_cat,
        main = "Box Plot ; ps_car_03_cat",
        ylab = "Differential of Values", names = c('train', 'test') )


boxplot(train$ps_car_05_cat, test$ps_car_05_cat,
        main = "Box Plot ; ps_car_05_cat",
        ylab = "Differential of Values", names = c('train', 'test') )


boxplot(train$ps_car_14, test$ps_car_14,
        main = "Box Plot ; ps_car_14",
        ylab = "Differential of Values", names = c('train', 'test') )





meta = data.frame(varname = names(train), type = rep(0, ncol(train)))
meta$varname = names(train)
meta[grepl("cat", names(train)),]$type <- "categorical"
meta[grepl("bin", names(train)),]$type <- "binary"
meta[which(sapply(train, class) == "integer" & meta$type == 0) ,]$type <- "ordinal"
meta[which(sapply(train, class) == "numeric" & meta$type == 0) ,]$type <- "continuous"
meta[meta$varname == "target",]$type <- "response"
meta$type <- as.factor(meta$type)
summary(meta)


#boxplot(train$ps_car_14, test$ps_car_14)
#boxplot(train$ps_car_07_cat, test$ps_car_07_cat)


