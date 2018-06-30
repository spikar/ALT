# -*- coding: utf-8 -*-
"""
Created on Sun May 27 13:22:24 2018

@author: AJHUNJH
"""

# Auto Learning Tool - Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('C:\\Users\\AJHUNJH\\Downloads\\Concrete_Data.csv')

class mlr():
    
    def __init__(self, dataset, scaling, x_label, x_cat):
        self.training_data = dataset
        self.X = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values
        self.y = self.y.reshape(-1,1)
        self.x_label = x_label
        self.x_cat = x_cat
        self.scaling = scaling
        
        for i in x_label:
            self.X[:, i] = labelencoder.fit_transform(self.X[:, i])
    
        onehotencoder = OneHotEncoder(categorical_features = x_cat)
        self.X = onehotencoder.fit_transform(self.X).toarray()
        
        if scaling == 1:
            from sklearn.preprocessing import StandardScaler
            self.sc_X = StandardScaler()
            self.X = self.sc_X.fit_transform(self.X)
            self.sc_y = StandardScaler()
            self.y = self.sc_y.fit_transform(self.y)
            
        self.regressor = LinearRegression()
        self.regressor.fit(self.X, self.y)
            
    
    def predict(self,X_pred):
        
        for i in self.x_label:
            X_pred[:, i] = labelencoder.fit_transform(X_pred[:, i])
    
        onehotencoder = OneHotEncoder(categorical_features = self.x_cat)
        X_pred = onehotencoder.fit_transform(X_pred).toarray()
        
        if self.scaling == 1:
            X_pred = self.sc_X.transform(X_pred)
            y_pred = self.regressor.predict(X_pred)
            return self.sc_y.inverse_transform(y_pred)
        else:
            y_pred = self.regressor.predict(X_pred)
            return y_pred
            
            
model = mlr(dataset,1,[11],[8,9,10,11])

y_pred = model.predict(X)


















            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            