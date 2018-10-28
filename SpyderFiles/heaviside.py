# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:37:50 2018

@author: Andrea
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets



#                     0   if x < 0
#heaviside(x, h0) =  h0   if x == 0
#                    1   if x > 0


np.heaviside([-1.5, 0, 2.0], .5)
