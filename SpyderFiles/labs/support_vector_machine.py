# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:54:14 2018

@author: Andrea
"""

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

X = shuffle(X)
y = shuffle(y)


svm_clf = Pipeline((
       ("scaler", StandardScaler()),
       ("linear_svc", LinearSVC(C=1, loss="hinge")),
   ))
# fit the model
svm_clf.fit(X, y)

# make predictions

svm_clf.predict([[5.5, 1.7]])