# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:55:28 2018

@author: Andrea
"""

import numpy as np 
from numpy import arange
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

dat = datasets.load_diabetes()
dat_x = dat.data
day_y = dat.target


 