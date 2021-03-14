# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:28:36 2020

@author: Shubham
"""
# Problem Statement:- Whether the client has subscribed a term deposit or not Binomial ("yes" or "no")

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

bank1 = pd.read_csv("D:\\Data Science study\\assignment\\Sent\\6\\bank-full.csv",sep=';')

# Lets drop the unnecessary columns from the data

bank1.columns
bank = bank1.drop(['marital','education','day', 'month', 'duration','default','pdays'],axis=1)
bank
bank.columns

# plotting barplot for catagorical columns

sb.countplot(x='job',data=bank,palette="hls")
sb.countplot(x='housing',data=bank,palette="hls")
sb.countplot(x='loan',data=bank,palette="hls")
sb.countplot(x='contact',data=bank,palette="hls")
sb.countplot(x='campaign',data=bank,palette="hls")
sb.countplot(x='previous',data=bank,palette="hls")
sb.countplot(x='poutcome',data=bank,palette="hls")

bank.job.value_counts()
bank.describe  #getting summary of the dataframe
bank.info   ##getting summary of the dataframe
bank.dtypes
pd.crosstab(bank.job,bank.y).plot(kind="bar")
pd.crosstab(bank.housing,bank.y).plot(kind="bar")
pd.crosstab(bank.loan,bank.y).plot(kind="bar")
pd.crosstab(bank.contact,bank.y).plot(kind="bar")
pd.crosstab(bank.campaign,bank.y).plot(kind="bar")
pd.crosstab(bank.previous,bank.y).plot(kind="bar")
pd.crosstab(bank.poutcome,bank.y).plot(kind="bar")

# Lets see the box plonow
sb.boxplot(x="age",y="y",data=bank,palette='hls')
sb.boxplot(x="balance",y="y",data=bank,palette='hls')  # boxplot needs at least one numeric variable so we can only see these two plots

bank.isnull().sum()
bank.shape

# Now lets build our model
# First create the X & Y datasets

X=bank.drop(["y"],axis=1)
Y=bank.iloc[:,9]

#Lets create the dummies for the catagorical data

X_dummies = pd.get_dummies(X[["job","housing","loan","contact","poutcome"]])
# drop the original columns
X.drop(["job","housing","loan","contact","poutcome"],inplace=True,axis=1)
X_new = pd.concat([X,X_dummies],axis=1)


# Model
banks=LogisticRegression()
banks.fit(X_new,Y)
print(banks.intercept_,banks.coef_)
print(banks.intercept_)    #B0

prob = banks.predict_proba(X_new)    #Predicting probability values
y_pred = banks.predict(X_new)    #predicting output instead of the probabiulity values
bank["y_pred"] = y_pred
y_prob = pd.DataFrame(banks.predict_proba(X_new.iloc[:,:]))
new_bank = pd.concat([bank,y_prob],axis=1)
new_Y = pd.concat([Y,y_prob],axis=1)  #concating original output and output probabilities
Y_dummy = pd.get_dummies(Y)
# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print(confusion_matrix)
type(y_pred)

#   Accuracy
accuracy = sum(Y==y_pred)/new_bank.shape[0]
new_bank.shape(0)
accuracy    #  0.8934108955785096
pd.crosstab(y_pred,Y)

# ROC Curve
fpr,tpr,thresholds = metrics.roc_curve(Y_dummy.iloc[:,1:],y_prob.iloc[:,1:])  #only takes binary numerical columns
fig,ax = plt.subplots()
plt.plot(fpr,tpr);plt.xlabel("FalsePositive");plt.ylabel("TruePositive");

roc_auc = metrics.auc(fpr,tpr)
roc_auc   #0.7427136939531613
