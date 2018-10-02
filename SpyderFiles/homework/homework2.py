# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 13:26:02 2018

@author: Andrea
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion, Pipeline

dat = pd.read_csv("C:/Users/Andrea/Downloads/train.csv")
df = pd.DataFrame(dat)