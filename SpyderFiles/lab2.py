# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:34:07 2018

@author: Andrea
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
input_file = "train.csv"
dat = pd.read_csv(input_file)
dat.shape
dat.count() #non-NaN values

dat.count()/len(dat) #%of non-NaN in column

#what fraction of passengers survived
dat["Survived"].value_counts() #like Table

#matplotlib barplot
dat["Survived"].value_counts().plot(kind="bar")
plt.title("Counts of Titanic Survival")

#matplotlib Histogram
plt.hist(dat["Age"].dropna())


#seaplot barplot
sns.countplot(x="Survived",data =dat)
plt.title("Counts of Titanic Survival")


#matplotlib num passengers each class
dat["Pclass"].value_counts().plot(kind="bar")

#seaborn num passengers each class
sns.countplot(x = "Pclass", data = dat)

#seaborn hist no curve
sns.distplot(dat['Age'].dropna(),hist=True,kde = False)
plt.title("Histogram of Titanic passenger ages")