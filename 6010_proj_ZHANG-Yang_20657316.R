library(xgboost)
library(dplyr)
library(data.table) 
setwd("/Users/zhangyang/Desktop/MAFS 6010S machine learning and its application/CourseProject/data")  #set my working directory

## read data from the local path
members_raw <- fread("members.csv")
songs_raw <- fread("songs.csv")
train_raw <- fread("train.csv")
test_raw <- fread("test.csv")

## convert the date from long integer to date format
time <- function(date){ # date is numerical form
  date1<-as.character(date)
  paste0(substr(date1, 1, 4), "-", substr(date1, 5, 6), "-",substr(date1, 7, 8))
}

members_raw[, registration_init_time := as.Date(time(registration_init_time))]
members_raw[, expiration_date := as.Date(time(expiration_date))]

## combine the train and test data
train_raw [, id := -1]
test_raw [, target := -1]
all_data<- rbind(train_raw, test_raw)

print(sum(is.na(all_data)))

## Merge train and test data with songs and members dataset
all_data <- merge(all_data, members_raw, by = "msno", all.x=TRUE)
all_data <- merge(all_data, songs_raw, by = "song_id", all.x=TRUE)

print(sum(is.na(all_data)))

## Label encode the char columns
for (i in names(all_data)){
  if( class(all_data[[i]]) == "character"){
    all_data[is.na(all_data[[i]]), eval(i) := ""]
    all_data[, eval(i) := as.integer(as.factor(all_data[[i]]))]
  } else {
    all_data[is.na(all_data[[i]]), eval(i) := -999]
  }
}

## Transfer the registration_init_time and expiration_date into length_membership
## For now to Jilian only: Extract the Julian time (days since some origin).
all_data[, registration_init_time := julian(registration_init_time)]
all_data[, expiration_date := julian(expiration_date)]
all_data[, length_membership := expiration_date - registration_init_time]

## remove registration_init_time, expiration_date, composer and lyricist(too many NA)
all_data <- all_data[,-c(12,13,17,18)]

setDF(all_data)
train <- all_data[all_data$id== -1,]
test <- all_data[all_data$target == -1,]
train$id <- NULL
test$target <- NULL

target<- train$target
test_id <- test$id
train$target <- NULL
test$id <- NULL


## divide the original train dataset into train and validation
set.seed(1)
test_size <- 0.3
train_index <- sample(nrow(train), test_size*nrow(train))
vali_data <- train[train_index,]
vali_target <- target[train_index]

train_data <- data.frame(train[-train_index,])
train_target <- target[-train_index]

## scale the data by columns
#for(i in names(train_data)){
#  meam <- mean(train_data[[i]])
#  sd <- sd(train_data[[i]])
#  train_data[[i]] <- (train_data[[i]] - mean)/sd
#  test[[i]] <- (test[[i]] - mean)/sd
#  vali_data <- (vali_data[[i]] - mean)/sd
#}


## Auxiliary fucntions
## Accuracy
accu <- function(actual, pred_prob, threshold=0.5){
  prediction <- ifelse(pred_prob > threshold, 1, 0)
  accuracy <- mean(prediction == actual)
  return(accuracy)
}

## AUC 
auc <- function(actual, pred_prob){
  N <- length(actual)
  if(length(pred_prob) != N){
    return(NULL)  # error occur
  } else{
    if(is.factor(actual)) {
      actual <- as.numeric(as.character(actual))
    }
  }
  roc_actual <- actual[order(pred_prob, decreasing = FALSE)]
  stack_x = cumsum(roc_actual == 1) / sum(roc_actual == 1)
  stack_y = cumsum(roc_actual == 0) / sum(roc_actual == 0)
  auc = sum((stack_x[2:N] - stack_x[1:(N - 1)]) * stack_y[2:N])
  return(auc)
}

criteria <- function(actual, pred_prob, title=""){
  cat("\nSummary results for", title
      , "\nAUC:", auc(actual, pred_prob)
      , "\nAccuracy:", accu(actual, pred_prob)
      , "\n"
  )
}


## Model building: xgboost
## Convert the data into xgb.DMatrix object
xgb_train_data <- xgb.DMatrix(as.matrix(train_data),label = train_target, missing=-1)
xgb_vali_data <- xgb.DMatrix(as.matrix(vali_data), label = vali_target, missing=-1)

## set parameter
parameter = list(
  eta  = 0.2, 
  max_depth= 10,
  min_child= 6,
  subsample= 0.5, 
  colsample_bytree=0.45, 
  objective="binary:logistic",
  eval_metric= "auc",
  tree_method= "approx", 
  nthreads = 8
)

## build the model
xgb <- xgb.train(
  data = xgb_train_data,
  nrounds = 100, 
  params = parameter,  
  maximize= T,
  print_every_n = 50
)

## out-sample prediction
pred  <- predict(xgb, xgb_vali_data)
criteria(vali_target, pred, title="xgb")
# confusion matrix
XGB.pred=rep("0",length(pred))
XGB.pred[pred>0.5] <- 1 
table(XGB.pred,vali_target)

# In-sample
pred_in  <- predict(xgb, xgb_train_data)
criteria(train_target, pred_in, title="xgb")




















