# Course Project

# training data - https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
mypath <- c("C:\\courseproj_ml")
setwd(mypath)

mydest_train <- c("pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile=mydest_train)
training <- read.csv(file=mydest_train, header=TRUE)

# test data - https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
mydest_test <- c("pml-testing.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile=mydest_test)
testing <- read.csv(file=mydest_test, header=TRUE)

# The goal of your project is to predict the manner in which they did the exercise. 
# This is the "classe" variable in the training set. You may use any of the other 
# variables to predict with. You should create a report describing how you built 
# your model, how you used cross validation, what you think the expected out of 
# sample error is, and why you made the choices you did. You will also use your 
# prediction model to predict 20 different test cases.

# data from accelerometers on: belt, forearm, arm, and dumbell of 6 participants
# barbell lifts performed correctly and incorrectly in 5 different ways

library(caret)
library(kernlab)
library(plyr)
library(dplyr)
library(rpart.plot)
library(rpart)

# removing near zero covariates
# zerovar TRUE means that there is only one distinct value in the predictor
# nzv TRUE means that the predictor is a near zero variance predictor
# consider removing all zerovar = TRUE or nzv = TRUE ; the identifiers
# and "classe" are all still intact

nsv <- nearZeroVar(training, saveMetrics=TRUE)
nsvind <- mutate(nsv, include=!(nsv$zeroVar|nsv$nzv))
trainingnew <- training[,nsvind$include]

mynas <- matrix(0, nrow=1, ncol = dim(trainingnew)[2])
for (i in 1:dim(trainingnew)[2]){
  mynas[,i] <- sum(is.na(trainingnew[,i]))
}

#looks like columns 1:7, 11:26, 50:53, 57:62, 64:73, 86:88, 90 have 19216 NA values (out of 19622 obs)

removecols <- c(1:7, 11:26, 40, 50:53, 57:62, 64:73, 86:88, 90)
trainingnew2 <- trainingnew[,-removecols]

#remove the same columns from testing
testingnew <- testing[,nsvind$include]
testingnew2 <- testingnew[,-removecols]

#partition training set into training/test sets for cross validation (to decide
# what type of model to use, based on accuracy)
set.seed(234)
inTrain <- createDataPartition(trainingnew2$classe, p=0.6)[[1]]
validation <- trainingnew2[-inTrain,]
trainingnew3 <- trainingnew2[inTrain,]
trainingnew2 <- trainingnew3

dim(validation)
dim(trainingnew2)


# trees - accuracy 71%
mytree <- rpart(classe ~. ,method="class", data=trainingnew2)
predtree <- predict(mytree, validation, type="class")
conftree <- confusionMatrix(predtree, validation$classe)

rpart.plot(mytree,main="Classification Treet", extra=102, under=TRUE, faclen=0)


# random forest - accuracy 99%
set.seed(234)
library(randomForest)
myrf <- randomForest(classe ~., data=trainingnew2)
predrf <- predict(myrf, validation)
confrf <- confusionMatrix(predrf, validation$classe)

# boosting - accuracy 96%
set.seed(234)
mygbm <- train(classe ~., method="gbm", data=trainingnew2, verbose=FALSE, trControl=trainControl(method="cv", number=3))
predgbm <- predict(mygbm, validation)
confgbm <- confusionMatrix(predgbm, validation$classe)

#choose random forest, and predict
predrftesting <- predict(myrf, testingnew2)


