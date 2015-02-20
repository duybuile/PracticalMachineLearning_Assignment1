# Practical Machine Learning - Assignment 1
Duy Bui  
Saturday, January 10, 2015  

##Synopsis
The following analysis practices the usage of machine learning techniques to solve a real-life problem. The goal of this practice is to predict the manner of each personal activity set. The data were collected from the accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The analysis was initiated by pre-processing the data. After that, the data were split into training and testing sets, then were used to build the classification model. Out-of-sample and in-sample errors were estimated to evaluate the classification model.    

##Analysis
First of all, we read the data from csv files

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pml.train <- read.csv(file = "pml-training.csv", header = TRUE, sep = ",")
pml.test <- read.csv(file = "pml-testing.csv", header = TRUE, sep = ",")
```

###Pre-process the data
#####1. Remove NA columns
We remove all the columns having only NAs value in the testing data since they are meaningless when it comes to prediction. Simultaneously, we remove those columns in the training data 


```r
pml.train <- pml.train[,which(unlist(lapply(pml.test, function(x)!all(is.na(x)))))]
pml.test <- pml.test[,which(unlist(lapply(pml.test, function(x)!all(is.na(x)))))]
```

#####2. Remove unimportant columns
The data set now only contains 60 variables. It is quite obvious that we could eliminate timestamp and name variables in the prediction model. Therefore, 5 following variables will be eliminated (X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp and name)


```r
excl.list <- c("X", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "user_name")
eliminateCol <- function(dat, excl.list){
  excl.index = which(names(dat) %in% excl.list)
  dat <- subset(dat, select = -excl.index)
  dat
}

pml.train <- eliminateCol(pml.train, excl.list)
pml.test <- eliminateCol(pml.test, excl.list)
```

#####3. Remove near-zero variance columns
As some predictors could become zero-variance or near zero-variance when the data is split into training/testing set, it is better to eliminate the near-zero variance predictors. In order to find the near-zero variance predictors, we use the method nearZeroVar in caret package


```r
nzv <- nearZeroVar(pml.train, saveMetrics= TRUE)
nzv[nzv$nzv,]
```

```
##            freqRatio percentUnique zeroVar  nzv
## new_window  47.33005    0.01019264   FALSE TRUE
```

From the analysis, we know that new_window is a near-zero variance predictor, which should be eliminated. 


```r
excl.list <- c("new_window")
pml.train <- eliminateCol(pml.train, excl.list)
pml.test <- eliminateCol(pml.test, excl.list)
```

#####4. Remove correlated variables
It is obvious that highly correlated variables would not contribute much on the training. In order to find the highly correlated variables, we calculate the correlation matrix first. It is noted that the factor variables should not be present in the correlation matrix. Also, all variables having correlation greater 0.75b are considered as highly correlated


```r
pmlCor <-  cor(pml.train[sapply(pml.train, function(x) !is.factor(x))])
highlyCor <- findCorrelation(pmlCor, cutoff = .75)
length(highlyCor)
```

```
## [1] 20
```

The analysis shows that we could eliminate up to 20 highly correlated variables. This leaves our data with 34 variables only.


```r
pml.train <- pml.train[,-highlyCor]
pml.test <- pml.test[, -highlyCor]
```

###Split the data
Let's divide the data into training and testing set. The ratio is 75/25

```r
trainIndex = createDataPartition(pml.train$classe, p=0.75, list=FALSE)
training = pml.train[ trainIndex, ]
testing = pml.train[ - trainIndex, ]
```

###Train data and fit the model
We fit a model using random forest method

```r
set.seed(33833)
modFit <- train(classe ~ ., method = "rf", data=training)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

Now we test the accuracy of the classifier

```r
predicted <- predict(modFit, newdata = training)
table(predicted == training$classe)
```

```
## 
##  TRUE 
## 14718
```

```r
predicted2 <- predict(modFit, newdata = testing)
table(predicted2 == testing$classe)
```

```
## 
## FALSE  TRUE 
##    13  4891
```

The accuracy is very high, giving us a strong confidence on the classification. Indeed, when it comes to error estimation, we could use the confusionMatrix function to calculate the in-sample and out-of-sample error as follows:


```r
#In-sample error
confusionMatrix(table(predicted, training$classe))$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      1.0000000      1.0000000      0.9997494      1.0000000      0.2843457 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

```r
#Out-of-sample error
confusionMatrix(table(predicted2, testing$classe))$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9973491      0.9966468      0.9954712      0.9985878      0.2844617 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

The in-sample and out-of-sample error are 0 and 0.0018352 respectively, which suggests a strongly accurate classification model.  

###Test with the provided testing data
We use the fit model to test with the testing data from the problem

```r
predicted3 <- predict(modFit, newdata = pml.test)
predicted3
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Write the output into 20 separated files.

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(predicted3)
```

##Conclusion
The practice of machine learning found a highly accurate classification model by using the random forest method (the in-sample and out-of-sample errors were estimated as approximately 0). By using this classification model, the 20 test cases were predicted with an accuracy of 100%. It is also noted that the analysis did not performe any cross-validation as it is unnecessary in rain-forest method.     
