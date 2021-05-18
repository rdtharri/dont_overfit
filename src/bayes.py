#%% Import Libraries

import gc
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import random, randint

#%% Import Data

train = pd.read_csv("C:/Users/rithar/OneDrive - SAS/Training/Don't Overfit/data/train.csv")
test= pd.read_csv("C:/Users/rithar/OneDrive - SAS/Training/Don't Overfit/data/test.csv")

#%% X,y Separation

scaler = StandardScaler()

train_id = train.id.values
train_target = train.target.values
train_x = train[[str(i) for i in range(300)]].values
train_x = scaler.fit_transform(train_x)

test_id = test.id.values
test_x = test[[str(i) for i in range(300)]].values
test_x = scaler.transform(test_x)

#%% Sampler

def data_sample(train_mat, col_ratio=0.5):
    
    #Random Columns
    random_columns = random(300)
    random_rows = randint(0,250,size=250)

    sample_x = train_mat[random_rows]
    sample_x = sample_x[:,random_columns>col_ratio]
    
    sample_y = train_target[random_rows]
    
    #sample
    return sample_x, sample_y,(random_columns > col_ratio)

#%% Test Sampler
    
sample_x, sample_y, columns = data_sample()

#%% Logistic Regressions

iterations = 1000
parm_estimates = np.full((iterations,train_x.shape[1]),np.nan)

for i in tqdm(range(iterations)):
    
    #Get Sample
    sample_x, sample_y, columns = data_sample(train_x)
    
    #Train Logistic Regresstion
    clf = LogisticRegression(class_weight='balanced', 
                             penalty='l1', 
                             C=0.1,
                             fit_intercept = False,
                             solver='liblinear')                             
    clf.fit(sample_x, sample_y)
    
    #Append Coefficients to Matrix
    parm_estimates[i,columns] = clf.coef_
    
#%% Summarize Parm Estimates
    
parm_means = np.zeros(train_x.shape[1])
parm_stds = np.zeros(train_x.shape[1])

for col in tqdm(range(train_x.shape[1])):
    parm_means[col] = np.nanmean(parm_estimates[:,col])
    parm_stds[col] = np.nanstd(parm_estimates[:,col])
    
plt.hist(parm_means)
plt.title("Parameter Histogram")
plt.show()

plt.hist(parm_stds)
plt.title("Std Histogram")
plt.show()
    
#%% Post-Process Parm Estimates    
    
for parm in range(parm_means.shape[0]):
    if parm_stds[parm] * 0.5 > parm_means[parm]:
        parm_means[parm] = 0
        
plt.hist(parm_means)
plt.title("Parameter Histogram")
plt.show()
    
#%% Use Parameter Estimates

#Parmaters to Column Matrix
beta = parm_means.reshape((300,1))

#Score Datasets
train_odds = np.exp(np.dot(train_x,beta))
train_predictions = train_odds / (1 + train_odds)
test_odds = np.exp(np.dot(test_x,beta))
test_predictions = test_odds / (1 + test_odds)

#Train AUC
train_auc = roc_auc_score(train_target, train_predictions)
print(train_auc)  
    
#%% Create Submission

submission = pd.DataFrame({'id' : test_id, 'target' : test_predictions.ravel()})
submission.to_csv('submissions/bayes_submission.csv',index = False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    