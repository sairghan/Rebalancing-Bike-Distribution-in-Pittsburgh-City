library(class) #for knn
library(car)  # for recode
library(plotly) #for bar chart
library('pROC') # for ROC curve
library(plyr) # for recoding data
library(ROCR) # for plotting roc
library(e1071) # for SVM
library(rpart) # for decision tree

set.seed(12345) # set the seed to get exactly the same results every time

do.classification <- function(train.set, test.set, 
                              cl.name, verbose=F) {
  ## use the raw probabilities to plot ROC
  switch(cl.name, 
         knn1 = { # Knn test k=1
           prob = knn(train.set[,-1], test.set[,-4], cl=train.set[,4], k = 1, prob=T)
           prob = attr(prob,"prob")
           prob
         },
         knn15 = { # Knn test k=15
           prob = knn(train.set[,-1], test.set[,-4], cl=train.set[,4], k = 15, prob=T)
           prob = attr(prob,"prob")
           prob
         },
         knn30 = { # Knn test k=30
           prob = knn(train.set[,-1], test.set[,-4], cl=train.set[,4], k = 30, prob=T)
           prob = attr(prob,"prob")
           prob
         },
         lr = { # logistic regression
           model = glm(y~., family=binomial(link="logit"), data=train.set)
           if (verbose) {
             print(summary(model))             
           }
           prob = predict(model, newdata=test.set, type="response") 
           prob
         },
         svm = { # Support vector machine
           model = svm(y~., data=train.set, probability=T)
           if (0) {
             tuned <- tune.svm(y~., data = train.set, 
                               kernel="radial", 
                               gamma = 10^(-6:-1), cost = 10^(-1:1))
             gamma = tuned[['best.parameters']]$gamma
             cost = tuned[['best.parameters']]$cost
             model = svm(y~., data = train.set, probability=T, 
                         kernel="radial", gamma=gamma, cost=cost)                        
           }
           prob = predict(model, newdata=test.set, probability=T)
           prob = attr(prob,"probabilities")
           prob = prob[,which(colnames(prob)==1)]/rowSums(prob)
           prob
         },
         dtree = { # decision tree
           model = rpart(y~., data=train.set)
           prob = predict(model, newdata=test.set)
           prob = prob[,2]/rowSums(prob) # renormalize the prob.
           prob
         }
  ) 
}



cv <- function(dataset, testingset, cl.name, prob.cutoff=0.5) {
  probs = NULL
  actuals = NULL
  train.set = dataset
  test.set = testingset
  prob = do.classification(train.set, test.set, cl.name)
  predicted = as.numeric(prob > prob.cutoff)
  actual = test.set$y
  confusion.matrix = table(actual,factor(predicted,levels=c(0,1)))
  confusion.matrix
  error = (confusion.matrix[1,2]+confusion.matrix[2,1]) / nrow(test.set)  
  probs = c(probs,prob)
  actuals = c(actuals,actual)
  
  ## plot ROC
  result = data.frame(probs,actuals)
  pred = prediction(result$probs,result$actuals)
  perf = performance(pred, "tpr","fpr")
  plot(perf)  
  
  ## get other measures by using 'performance'
  get.measure <- function(pred, measure.name='auc') {
    perf = performance(pred,measure.name)
    m <- unlist(slot(perf, "y.values"))
    m
  }
  err = mean(get.measure(pred, 'err'))
  precision = mean(get.measure(pred, 'prec'),na.rm=T)
  recall = mean(get.measure(pred, 'rec'),na.rm=T)
  fscore = mean(get.measure(pred, 'f'),na.rm=T)
  cat('error=',err,'precision=',precision,'recall=',recall,'f-score',fscore,'\n')
  auc = get.measure(pred, 'auc')
  cat('auc=',auc,'\n')
  m1 <- data.frame(err,precision,recall,fscore,auc)
  #return(m1) #get result 
  m2 <- c(m1,result)
  return(m2) 
}



my.classifier <- function(dataset, testingset, cl.name, do.cv=F) {
  n.obs <- nrow(dataset) # no. of observations in dataset
  n.cols <- ncol(dataset) # no. of predictors
  print(table(dataset$y)) # 2x2table
  if (do.cv) cv(dataset, testingset,cl.name)
}


### main ###
dataset <- read.table("C:/Users/sag163/Downloads/Training Dataset.csv", header=T, sep=",")
testingset <- read.table("C:/Users/sag163/Downloads/Testing Dataset.csv", header=T, sep=",")
names(dataset)[names(dataset)=="size"]="y";
names(testingset)[names(testingset)=="size"]="y";
for (difY in 1:nrow(dataset)){
  if(dataset$y[difY]<0){
    dataset$y[difY] = 0
  }else{
    dataset$y[difY] = 1
  } 
}
for (difZ in 1:nrow(testingset)){
  if(testingset$y[difZ]<0){
    testingset$y[difZ] = 0
  }else{
    testingset$y[difZ] = 1
  } 
}
dataset$y = as.factor(dataset$y)
dataset$weekday= as.factor(dataset$weekday)
dataset$hour= as.factor(dataset$hour)
dataset$StationNum = as.factor(dataset$StationNum)

testingset$y = as.factor(testingset$y)
testingset$weekday= as.factor(testingset$weekday)
testingset$hour= as.factor(testingset$hour)
testingset$StationNum = as.factor(testingset$StationNum)
#dataset[1:3,]
#testingset[1:3,]

classresult <- data.frame(err=double(),precision=double(),recall=double(),fscore=double(),auc=double(),stringsAsFactors=FALSE)
ROCcurve_lr <- data.frame(probs=double(8568),actuals=double(8568))
ROCcurve_knn1 <- data.frame(probs=double(8568),actuals=double(8568))
ROCcurve_knn15 <- data.frame(probs=double(8568),actuals=double(8568))
ROCcurve_knn30 <- data.frame(probs=double(8568),actuals=double(8568))
ROCcurve_svm <- data.frame(probs=double(8568),actuals=double(8568))
ROCcurve_dtree <- data.frame(probs=double(8568),actuals=double(8568))



result_knn1 <- my.classifier(dataset, testingset, cl.name='knn1',do.cv=T)

result_knn15 <- my.classifier(dataset, testingset, cl.name='knn15',do.cv=T)

result_knn30 <- my.classifier(dataset, testingset, cl.name='knn30',do.cv=T)

result_dtree <- my.classifier(dataset, testingset, cl.name='dtree',do.cv=T)


result_svm <- my.classifier(dataset, testingset, cl.name='svm',do.cv=T)


result_lr <- my.classifier(dataset, testingset, cl.name='lr',do.cv=T)


classresult[1,1] = result_lr$err
classresult[1,2] = result_lr$precision
classresult[1,3] = result_lr$recall
classresult[1,4] = result_lr$fscore
classresult[1,5] = result_lr$auc

classresult[2,1] = result_knn1$err
classresult[2,2] = result_knn1$precision
classresult[2,3] = result_knn1$recall
classresult[2,4] = result_knn1$fscore
classresult[2,5] = result_knn1$auc

classresult[3,1] = result_knn15$err
classresult[3,2] = result_knn15$precision
classresult[3,3] = result_knn15$recall
classresult[3,4] = result_knn15$fscore
classresult[3,5] = result_knn15$auc

classresult[4,1] = result_knn30$err
classresult[4,2] = result_knn30$precision
classresult[4,3] = result_knn30$recall
classresult[4,4] = result_knn30$fscore
classresult[4,5] = result_knn30$auc

classresult[5,1] = result_svm$err
classresult[5,2] = result_svm$precision
classresult[5,3] = result_svm$recall
classresult[5,4] = result_svm$fscore
classresult[5,5] = result_svm$auc

classresult[6,1] = result_dtree$err
classresult[6,2] = result_dtree$precision
classresult[6,3] = result_dtree$recall
classresult[6,4] = result_dtree$fscore
classresult[6,5] = result_dtree$auc
#classresult





####### generate table
rownames(classresult) <- c('lr','knn1','knn15','knn30','svm','dtree')
#classresult
classresultt = as.data.frame(t(classresult))
classresultt # get table



#write.csv(classresultt, file = "~/desktop/error table.csv", row.names = TRUE)

rnclassresult = row.names(classresult)
#rnclassresult


####### AUC bar chat

plot_ly(x = rnclassresult,y = classresult$auc,name = "AUC",type = "bar",marker = list(color = 'rgb(225,180,180)',
                                                                                      line = list(color = 'rgb(8,48,107)',width = 1.5)))
  layout(title = "AUC")


###### generate each ROC curve in same graph with legend
ROCcurve_lr[,1] <- result_lr$probs
ROCcurve_lr[,2] <- result_lr$actuals
ROCcurve_knn1[,1] <- result_knn1$probs
ROCcurve_knn1[,2] <- result_knn1$actuals
ROCcurve_knn15[,1] <- result_knn15$probs
ROCcurve_knn15[,2] <- result_knn15$actuals
ROCcurve_knn30[,1] <- result_knn30$probs
ROCcurve_knn30[,2] <- result_knn30$actuals
ROCcurve_svm[,1] <- result_svm$probs
ROCcurve_svm[,2] <- result_svm$actuals
ROCcurve_dtree[,1] <- result_dtree$probs
ROCcurve_dtree[,2] <- result_dtree$actuals

GG = result_svm$actuals
#write.table(GG, "~/desktop/mydata.txt", sep="\t")

Roc_lr=prediction(result_lr$probs,result_lr$actuals)
Roc1 = performance(Roc_lr, "tpr","fpr")
Roc_knn1=prediction(result_knn1$probs,result_knn1$actuals)
Roc2 = performance(Roc_knn1, "tpr","fpr")
Roc_knn15=prediction(result_knn15$probs,result_knn15$actuals)
Roc3 = performance(Roc_knn15, "tpr","fpr")
Roc_knn30=prediction(result_knn30$probs,result_knn30$actuals)
Roc4 = performance(Roc_knn30, "tpr","fpr")
Roc_svm=prediction(result_svm$probs,result_svm$actuals)
Roc5 = performance(Roc_svm, "tpr","fpr")
Roc_dtree=prediction(result_dtree$probs,result_dtree$actuals)
Roc6 = performance(Roc_dtree, "tpr","fpr")

plot(Roc1, col="red") 
plot(Roc2, add=TRUE, col="orange") 
plot(Roc3, add=TRUE, col="yellow") 
plot(Roc4, add=TRUE, col="green") 
plot(Roc5, add=TRUE, col="blue") 
plot(Roc6, add=TRUE, col="purple") 
legend("bottomright",c('lr','knn1','knn15','knn30','svm','dtree'),cex=0.7, fill=c("red","orange","yellow","green","blue","purple"))
