# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:49:09 2018

@author: Andrea
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA, KernelPCA


dat = pd.read_csv("C:/Users/Andrea/Downloads/train.csv")
df = pd.DataFrame(dat[["Age","SibSp","Fare"]])
df.fillna(df.mode().iloc[0],inplace=True) #replace NaNs with first row mode

df = pd.DataFrame(dat[["Age","SibSp","Fare"]])
imputer = Imputer(strategy="median")
imputer.fit(df)
x=imputer.transform(df)
df=pd.DataFrame(x,columns=df.columns)
