# The client has subscribed a term deposit or not.

bank1 <- read.csv("D:\\Data Science study\\assignment\\Sent\\6\\bank-full.csv",sep = ";")
View(bank1)

# The current data et is not proper for our analysis so lets make it presentable.

str(bank1)

colnames(bank1)

# Let's do cleansing of the data

bank <- bank1[,c(-3,-4,-10,-11,-12)]
bank$default <- NULL
bank$pdays <- NULL
str(bank)

is.na(bank)
sum(is.na(bank))

# Lets create the glm model for our data set

banks_model <- glm(y~., family = "binomial", data = bank)
summary(banks_model)

colnames(bank)

boxplot(bank$pdays)$out     # Not working
influence.measures(banks_model)    # Not working
summary(bank)

predict(banks_model,bank)  # predicting values 

prob <- banks_model$fitted.values   # probabilities of the final predictions

# lets plot the confusion matrix

confusion <- table(prob > 0.5, bank$y)
table(bank$y)   #calculating no. of 'yes' and 'no'

confusion

# Accuracy

Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy  #0.8934551

#Error

error <- 1-Accuracy
error   # 0.1065449


prop.table(table(bank$y))  # Accuracy is > % majority class hence its model is valid

prop.table(table(prob >0.5))

## ROC Curve
library(ROCR)
rocpred <- prediction(prob, bank$y)
rocpred

rocrperf <- performance(rocpred,'tpr', 'fpr')

str(rocrperf)

windows()
plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7),print.cutoffs.at=seq(0.1,by=0.1))

# Lets find the cut-off value, tpr, fpr

rocr_cutoff <- data.frame(cut_off=rocrperf@alpha.values[[1]],fpr=rocrperf@x.values,tpr=rocrperf@y.values)
colnames(rocr_cutoff) <- c("cut_off","FPR","TPR")
rocr_cutoff <- round(rocr_cutoff,2)   # rounding of values at 2 decimals.

# Lets sort the data 
 library(dplyr)
rocr_cutoff <- arrange(rocr_cutoff,desc(TPR))
View(rocr_cutoff)

install.packages("pROC")
library(pROC)
auc <- performance(rocpred,measure = "auc")
auc <- auc@y.values[[1]]
auc            # 0.7424884
str(auc)
