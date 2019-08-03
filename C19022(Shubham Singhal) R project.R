
# Please install libraries= DataExplorer,dummies,ggplot2,readr,dplyr,caret,randomForest,ROCR

data=read.csv(file='c:/C19022.csv')
str(data)
summary(data)
head(data)
dim(data)

#missing values
colSums(sapply(data,is.na))

#DATA EXPLORATION
library(ggplot2) # Data visualization
library(readr)
library(dplyr)

#numeric variables
numeric_var = select_if(data, is.numeric)
colSums(sapply(numeric_var, is.na))

#categorical variables
categorical_var = select_if(data, is.factor)
colSums(sapply(categorical_var, is.na))

#Replacing Revenue column boolean values with 1 and 0
data$Revenue=gsub('True',1,data$Revenue)
data$Revenue=gsub('False',0,data$Revenue)

#Conversion of datatypes
data$Region=as.character(data$Region)
data$OperatingSystems=as.character(data$OperatingSystems)
data$Browser=as.character(data$Browser)

# EDA using visualization
#Count of 0 and 1
ggplot(data, aes(x = Revenue)) + geom_bar(color='blue',fill = "#FF6666")
library(DataExplorer)
plot_str(data)
plot_missing(data)
plot_histogram(data)
plot_density(data)
plot_correlation(data, type = 'continuous')
plot_bar(data)
create_report(data)

x=select (data,-c(Revenue))
head(x)
dim(x)
y=data$Revenue
head(y)

# Creating dummies variable of categorical data
library(dummies)
X=dummy.data.frame(x)
head(X)
dim(X)

# Test - Train splitting
new=cbind(X,y)
dim(new)
head(new)
library(caret)
set.seed(3456)
trainIndex = createDataPartition(new$y, p = .8,
                                  list = FALSE,
                                  times = 1)
Train = new[ trainIndex,]
Test = new[-trainIndex,]

Train_x=select (Train,-c(y))
Test_x=select (Test,-c(y))
Train_y=Train$y
Test_y=Test$y

# Random forest classifier
library(randomForest)
rf=randomForest(Train_x, as.factor(Train_y), ntree=100, importance=TRUE)
rf
pred1=predict(rf,newdata=Test_x,type = "prob")
y_pred_num = ifelse(pred > 0.5, 1, 0)
y_pred = factor(y_pred_num, levels=c(0, 1))
#Accuracy
mean(y_pred == Test_y)

library(ROCR)
perf = prediction(pred1[,2],Test_y)

# 1. Area under curve
auc = performance(perf, "auc")
auc

# 2. True Positive and Negative Rate
pred3 = performance(perf, "tpr","fpr")

# 3. Plot the ROC curve
plot(pred3,main="ROC Curve for Random Forest",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")



# Logistic Regression
logitmod = glm(as.factor(y)~., family = "binomial", data=Train)
logitmod
pred = predict(logitmod, newdata = Test_x, type = "response")
y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
mean(y_pred == Test_y)
library(ROCR)
perf = prediction(pred,Test_y)

# 1. Area under curve
auc = performance(perf, "auc")
auc

# 2. True Positive and Negative Rate
pred3 = performance(perf, "tpr","fpr")

# 3. Plot the ROC curve
plot(pred3,main="ROC Curve for Logistic Regression",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")


#Hence, Random Forest is giving good accuracy = 0.875 with ROC AUC = 0.9164
#       Logistic Regression is giving good accuracy = 0.875 with ROC AUC = 0.8788
#   Thus, Random Forest model is preferred over Logistic regression for this dataset.
