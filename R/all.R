df = read.csv("/Users/camilo/Desktop/Research/Lab/R/data/improved/improved.csv", header=T,as.is=T)
data <- subset(df,select=c(2:202))
set.seed(1)
#Randomly shuffle the data
data<-data[sample(nrow(data)),]
# Split train and test dataset
train <- data[1:1545,]
test <- data[1546: 2207,]
#Fit model
model <- glm(RESULT ~ .,family=binomial(link='logit'),data=subset(train, select=c(-ID)))

# save this model
#save(model, file = "logitmodel.rda")
#saveRDS(model, "model.rds")
# ROC
p <- predict(model, newdata=subset(test,select=c(1:201)), type="response")
pr <- prediction(p, test$RESULT)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
print(paste('AUC:',auc))

# Model
fitted.results <- predict(model,newdata=subset(test,select=c(1:201)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$RESULT)
print(paste('Accuracy',1-misClasificError))

# Summary model
#summary(model)
#Table deviance
anova(model, test="Chisq")

## K-fold Cross Validation
#Create 5 equally size folds
folds <- cut(seq(1,nrow(data)),breaks=5,labels=FALSE)

#Perform 10 fold cross validation
library(ROCR)
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- data[testIndexes, ]
  trainData <- data[-testIndexes, ]
  print(paste("Test Data:", nrow(testData)))
  print(paste("Train Data:", nrow(trainData)))
  #Use the test and train data partitions however you desire...
  model <- glm(RESULT ~ .,family=binomial(link='logit'),data=subset(train, select=c(-ID)))
  fitted.results <- predict(model,newdata=subset(testData,select=c(1:201)),type='response')
  fitted.results <- ifelse(fitted.results > 0.55,1,0)
  misClasificError <- mean(fitted.results != test$RESULT)
  print(paste('Accuracy ',1-misClasificError))
  # Summary
  print(summary(model))
  #s <- summary(model)
  #filename <- paste("/Users/camilo/Desktop/Research/Lab/R/results/","summary",i,".txt", sep="")
  #capture.output(s, file = filename)
  #Anova
  #a <- anova(model, test="Chisq")
  #filename <- paste("/Users/camilo/Desktop/Research/Lab/R/results/","anova",i,".txt", sep="")
  #capture.output(s, file = filename)
}
