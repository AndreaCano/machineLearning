# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:59:17 2018

@author: Andrea
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()

dat = iris["data"]

plt.scatter(dat[:,1],dat[:,3])
plt.grid()
plt.title('Petal Width by Sepal Width')
plt.ylabel('Petal width(cm)')
plt.xlabel('Sepal width(cm)')

dat1 = pd.DataFrame(data = dat, columns = iris["feature_names"])
dat1["target"] = iris["target"]
dat1['target']=dat1['target'].replace([0,1,2],iris["target_names"])


sns.set(style="whitegrid")
sns.countplot(x="target",data =dat1)
plt.title("Number of Examples per Species")
plt.xlabel("species")

sns.distplot(dat1['sepal width (cm)'],hist=True,kde = False)
plt.title("Histogram of Sepal Width")

sns.barplot(x='target', y='sepal length (cm)', data = dat1, estimator = np.mean)
plt.title("Avg. Sepal Length by Species")
plt.ylabel("mean(sepal length (cm))")
plt.xlabel("species")

sns.boxplot(x= 'target', y= 'sepal width (cm)', data = dat1)
plt.title("Boxplot of Sepal Width by Species")
plt.xlabel("species")

sns.violinplot(x='target',y='sepal width (cm)', data=dat1)
plt.title("Violinplot of Sepal Width by Species")
plt.xlabel("species")

sns.violinplot(x='target',y='sepal width (cm)', data=dat1, split= True, inner="stick")
plt.title("Violinplot of Sepal Width by Species")
plt.xlabel("species")