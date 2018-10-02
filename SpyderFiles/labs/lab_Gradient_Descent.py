# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:48:50 2018

@author: Andrea
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from numpy import arange

def my_function(x):
  return (-1 * ((x-1)**2) +2)

x = np.arange(-1,4)

y = my_function(x)

#x = np.random.uniform(0, 5, 100)    # or use values equally spaced

#y = np.vectorize(my_function)(x)
plt.plot(x,y)


#define f_prime that finds derivative

eps = 1e-6
def gd(x, lr, steps):
    delta = f_prime(x) * lr
    if delta < eps or steps == 0: return x
    return gd(x + delta, lr, steps - 1)

gd(-1, 0.1, 1000)