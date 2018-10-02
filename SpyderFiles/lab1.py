# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import pandas as pd
input_file = "machine.csv"
dat = pd.read_csv(input_file)

#get dimensions
dat.shape

#indexing with Panda
df = pd.DataFrame(np.random.randn(6,4),
                  index = list('abcdef'),
                  columns = list('ABCD'))

#Shift and up arrow to highlight
#f9 
#index by label 
df.loc[['a','b'],:] #gives rows a and b,all columns
df.loc[:, 'A' : 'C'] #all columns

#index by position
df.iloc[:3] #rows up to 3,all columns
df.iloc[3:] #rows 3 and beyond, all columns
df.iloc[3:,[0,2]] #rows 3 and beyond, columns 0 and 2

#dictionary style indexing
df['D'] #column D
df[['A','C']] #columns A and C

df[1:3] #like a loop rows: 1 >= x < 3; all columns

#Remove first two columns
dat = dat.loc[:,'myct':] #all rows, that column and all after

