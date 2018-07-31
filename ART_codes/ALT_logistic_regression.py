# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 18:54:28 2018

@author: ajhunjh
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Importing the dataset
dataset = pd.read_csv('C:\\Users\\AJHUNJH\\Downloads\\Concrete_Data.csv')


class LR():
    
    def __init__(self, dataset, test_size):

        self.dataset = dataset
        self.X = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values

        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size, random_state = 0) 
        
    def feature_scaling(self):
        
        # Feature Scaling
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
    
    def train_model(self):
    
        # Fitting Logistic Regression to the Training set
        self.classifier = LogisticRegression(random_state = 0)
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self):
        # Predicting the Test set results
        self.y_pred = self.classifier.predict(self.X_test)
        
        
    def accuracy(self):    
        # Making the Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        
        













        