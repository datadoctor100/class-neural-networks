#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 08:22:46 2021

@author: operator
"""

# Import
from pylab import mpl, plt
mpl.rcParams['savefig.dpi'] = 500
mpl.rcParams['font.family'] = 'serif'
plt.style.use('seaborn')
mpl.rcParams['figure.figsize'] = [10, 6]
from operator import itemgetter
import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

class financial_network:
    
    def __init__(self, df, lags):
        
        self.df = df
        self.split_dat()
        self.nn = self.build_model()
        
    def split_dat(self):

        y = self.df['trades']
        x = self.df[[i for i in self.df.columns if i not in ['trades', 'signals_trade', 'total', 'ticker']]]

        self.xtrain, self.xval, ytrain, yval = train_test_split(x, y, random_state = 100, test_size = .2)
        
        self.ytrain = np_utils.to_categorical(ytrain)
        self.yval = np_utils.to_categorical(yval)

    def build_model(self):
        
        model = Sequential()
        model.add(Dense(12, input_dim = 4, activation = 'relu'))
        model.add(Dense(8, activation = 'relu'))
        model.add(Dense(2, activation = 'sigmoid'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        
        return model
    
    def evaluate_model(self, model):
        
        model.fit(self.xtrain, self.ytrain, epochs = 150, batch_size = 1000)
        
        scores = model.evaluate(self.xval, self.yval, verbose = 0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        
    def build_preds(self, model):
    
        preds = model.predict_classes(self.df[[i for i in self.df.columns if i not in ['trades', 'signals_trade', 'total', 'ticker']]])
        preds[0] = -1
        preds[-1] = 1
        
        return preds
    
class categorical_network:
    
    def __init__(self, df, y, loss_class, input_dim):
        
        self.df = df
        self.input_dim = input_dim
        
        self.y = self.df[y]
        self.x = self.df[[i for i in self.df.columns if i != y]]
        
        xtrain, xval, ytrain, yval = train_test_split(self.x, self.y, random_state = 100, test_size = .2)
        
        self.nn = self.build_model(loss_class)
        
        self.nn.fit(self.x, self.y, epochs = 100, batch_size = 1000, verbose = 0)
        
        self.preds = self.nn.predict_classes(xval)
        
        # Evaluate performance
        self.confusion_mat = metrics.confusion_matrix(yval, self.preds)
        self.auc = metrics.roc_auc_score(yval, self.preds)
        
    # Function to initialize network
    def build_model(self, loss_class):
        
        model = Sequential()
        model.add(Dense(12, input_dim = self.input_dim, activation = 'relu'))
        model.add(Dense(8, activation = 'relu'))
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(loss = loss_class, optimizer = 'adam', metrics = ['accuracy'])
        
        return model
