set.seed(1)
df = read.csv("/Users/camilo/Desktop/Research/Lab/R/data/self/self.csv", header=T,as.is=T)
data <- subset(df,select=c(2:203))
data<-data[sample(nrow(data)),]
# Split train and test dataset
train <- data[1:1545,]
test <- data[1546: 2207,]
#Fit model
model <- glm(RESULT ~.,family=binomial(link='logit'),data=train)
# Summary model
#summary(model)
#Table deviance
#anova(model, test="Chisq")
# Model
fitted.results <- predict(model,newdata=subset(test,select=c(1:202)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$RESULT)
print(paste('Accuracy',1-misClasificError))

# Confusion Matrix
#library(caret)
#confusionMatrix(data=fitted.results, reference=test$RESULT)

## K-fold Cross Validation
#Randomly shuffle the data
#Create 10 equally size folds
folds <- cut(seq(1,nrow(data)),breaks=5,labels=FALSE)
#Perform 10 fold cross validation
for(i in 1:5){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- data[testIndexes, ]
  trainData <- data[-testIndexes, ]
  #Use the test and train data partitions however you desire...
  model <- glm(RESULT ~. - ID,family=binomial(link='logit'),data=trainData)
  fitted.results <- predict(model,newdata=subset(testData,select=c(1:202)),type='response')
  fitted.results <- ifelse(fitted.results > 0.5,1,0)
  misClasificError <- mean(fitted.results != test$RESULT)
  print(paste('Accuracy ',1-misClasificError))
}
# ROC
library(ROCR)
p <- predict(model, newdata=subset(test,select=c(1:202)), type="response")
pr <- prediction(p, test$RESULT)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
