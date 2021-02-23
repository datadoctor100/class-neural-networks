#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:13:35 2021

@author: operator
"""

# Import
import os
os.chdir('/Users/operator/Documents')
import warnings
warnings.filterwarnings('ignore')
from nn import *
import pandas as pd

# Get 
df = pd.read_csv('/Users/operator/Documents/diabetes.csv')

print(f"Target Outcomes:  {df['Outcome'].unique()}")

# Model
nn = categorical_network(df, 'Outcome', 'binary_crossentropy', 8)

# Evaluate
print('Inspect Confusion Matrix for Predictions: ')
print(nn.confusion_mat)
print(f'ROC AUC Score for Model- {round(nn.auc, 3)}')
