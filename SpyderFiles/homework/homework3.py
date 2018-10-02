# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 12:28:44 2018

@author: Andrea
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

#-------------------------------------------------------

csv = pd.read_csv("C:/Users/Andrea/Downloads/train.csv")


#creating non-null
dat = pd.DataFrame(csv)

datCat = pd.DataFrame(dat[["Sex","Pclass","Survived","Embarked"]])
#,"Pclass", "Survived", "Embarked"
datNum = pd.DataFrame(dat[["Age","Fare"]])

datUse = pd.DataFrame(dat[["Age","Fare","Sex","Pclass", "Survived", "Embarked"]])

numDatArr = ["Age","Fare"]
catDatArr = ["Sex","Pclass","Survived","Embarked"]
#,"Pclass", "Survived", "Embarked"

#------------------------------------------------

#create a class to select colummns from a dataframe

class DataSelect(BaseEstimator,TransformerMixin):
    def __init__(self, colNames):
        self.colNames = colNames
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return  X[self.colNames]





#Getting rid of NA's for numerical data
pipeNumeric = Pipeline([
 ("select_cols", DataSelect(numDatArr)),
 ("remove_nas", Imputer(strategy="median")),
 ("z-scaling", StandardScaler())

 ])
pipNum = pipeNumeric.fit_transform(datNum)

#Need to deal with strings in categorical before putting in piepline so use get_dummies

class DumCat(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Adding dummies when needed"""
    def fit(self, X,y=None):
        self.dummyDat = pd.get_dummies(X) #Creating dataFrame to run the function
        return self
    def transform(self,X):
        return self.dummyDat #return the dummyDat that has the new columns

datTest = pd.DataFrame(dat[["Embarked"]])

DumCat().fit_transform(datTest) 
#to access, need to Class.fit_transform(dataWantChanged)


#Getting rid of NA's for categorical data
pipeCategory = Pipeline([
 ("select_cols", DataSelect(catDatArr)),
 ("get_dummies", DumCat()),
 ("remove_nas", Imputer(strategy="most_frequent")),
 ])
pipCat = pipeCategory.fit_transform(datCat)


#Now that the data is processed, need to join the Numerical and Categorical 

full_pipeline = FeatureUnion(transformer_list=[
         ('cat_data',pipeCategory),
        ('num_data',pipeNumeric)
        ])
united = full_pipeline.fit_transform(datUse)


#--------------------------------------------------

sns.set(style="whitegrid")
sns.countplot(x="Survived",data =dat, palette="Set3")
plt.title("Survival Count")


sns.barplot(x= "Survived", y="Age", hue ="Sex",data = dat, palette="Set3")
plt.title("Survival based on Age and Sex")



sns.countplot("Pclass", data=dat, hue="Sex",palette="Set3")
plt.title("Number of Passengers in each class by Gender")
plt.xlabel("Class");



sns.violinplot(x="Sex", y = "Age", hue ="Survived", data=dat, inner="box", palette="Set3", cut=2, linewidth=3)

#--------------------------------------------------
from sklearn.model_selection import train_test_split

x = united[:,[2,3,7]] #sex(2,3) and age(7)
y = united[:,1] #survived
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_test)
score = decisiontree.score(x_test,y_test)
score
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
y_pred = log_reg.predict(x_train)
log_scores = log_reg.score(x_test,y_test)
log_scores





