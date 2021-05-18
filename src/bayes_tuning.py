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

from sklearn.model_selection import KFold, StratifiedKFold

#%% Import Data

train = pd.read_csv("data/train.csv")
test= pd.read_csv("data/test.csv")

#%% X,y Separation

scaler = StandardScaler()

train_id = train.id.values
train_target = train.target.values
train_x = train[[str(i) for i in range(300)]].values
train_x = scaler.fit_transform(train_x)
train_x = np.concatenate((np.full((train_x.shape[0],1),1),train_x),axis=1)

test_id = test.id.values
test_x = test[[str(i) for i in range(300)]].values
test_x = scaler.transform(test_x)
test_x = np.concatenate((np.full((test_x.shape[0],1),1),test_x),axis = 1)

#%% Sampler

def data_sample(train_mat, col_ratio=0.5):
    
    #Random Columns
    random_columns = random(300)
    random_columns = np.concatenate((np.array([0]),random_columns),axis=0)
    random_rows = randint(0,train_mat.shape[0],size=250)

    sample_x = train_mat[random_rows]
    sample_x = sample_x[:,random_columns<col_ratio]
    
    sample_y = train_target[random_rows]
    
    #sample
    return sample_x, sample_y,(random_columns < col_ratio)


#%% Logistic Regressions

def bayes_log_predictions(train_data, train_target, test_data,
                          col_ratio  = 0.5,
                          iterations = 1000,
                          C          = 0.1,
                          cutoff     = 0.75,
                          retrain    = 1):

    parm_estimates = np.full((iterations,train_data.shape[1]),np.nan)
    int_estimates = np.zeros((iterations))

    for i in range(iterations):
        
        #Get Sample
        sample_x, sample_y, columns = data_sample(train_data,
                                                  col_ratio = col_ratio)
        
        #Train Logistic Regresstion
        clf = LogisticRegression(class_weight='balanced', 
                                 penalty='l1', 
                                 C=C,
                                 fit_intercept = False,
                                 solver='liblinear')                             
        clf.fit(sample_x, sample_y)
        
        #Append Coefficients to Matrix
        parm_estimates[i,columns] = clf.coef_
        int_estimates[i] = clf.intercept_
        
    int_mean = np.mean(int_estimates)
    parm_means = np.zeros(train_x.shape[1])
    parm_stds = np.zeros(train_x.shape[1])
    
    #Calculate Average, STDS
    for col in range(train_x.shape[1]):
        parm_means[col] = np.nanmean(parm_estimates[:,col])
        parm_stds[col] = np.nanstd(parm_estimates[:,col])

    #Cutoff (based on SD)
    for parm in range(parm_means.shape[0]):
        if parm_stds[parm] * cutoff > np.abs(parm_means[parm]):
            parm_means[parm] = 0
            
    #Optionally Retrain
    if retrain == 1:
        selected_parms = (np.abs(parm_means) > 0)
        
        train_data_sub = train_data[:,selected_parms]
        
        clf = LogisticRegression(class_weight='balanced', 
                                 penalty='l1', 
                                 C=C,
                                 fit_intercept = False,
                                 solver='liblinear')                             
        clf.fit(train_data_sub, train_target)
        
        int_mean = clf.intercept_
        parm_means = np.zeros((301))
        parm_means[selected_parms] = clf.coef_.ravel()

    #Parmaters to Column Matrix
    beta = parm_means.reshape((301,1))
    
    #Score Datasets
    train_odds = np.exp(np.dot(train_data,beta) + int_mean)
    train_predictions = train_odds / (1 + train_odds)
    test_odds = np.exp(np.dot(test_data,beta) + int_mean)
    test_predictions = test_odds / (1 + test_odds)
    
    #Train AUC
    #train_auc = roc_auc_score(train_target, train_predictions)
    #print("AUC on training: {}".format(train_auc)) 
    
    return train_predictions, test_predictions
    
#%% Test Function
    
train_preds, test_preds = bayes_log_predictions(train_x, train_target, test_x)
train_auc = roc_auc_score(train_target,train_preds)
print("Train AUC: {}".format(train_auc))

#%% Grid Search

col_ratios = [0.3,0.7, 1]
iterations = [1000]
Cs         = [0.1]
cutoff     = [0,0.1, 0.75, 1.5]
retrain    = [0, 1]

grid = np.array(np.meshgrid(col_ratios, iterations, Cs, cutoff,retrain)).T.reshape(-1,5)
cv_auc = np.zeros((grid.shape[0]))

for i, row in enumerate(grid):
    
    folds = 5
    valid_aucs = []

    #five-fold predictions are the best predictions
    kf = KFold(n_splits=folds,shuffle=True)
    
    for train_index, valid_index in kf.split(train_x,train_target):
        
        #Get Model Predictions
        train_preds, valid_preds = bayes_log_predictions(train_x[train_index], 
                                                         train_target[train_index], 
                                                         train_x[valid_index],
                                                         col_ratio  = row[0],
                                                         iterations = int(row[1]),
                                                         C          = row[2],
                                                         cutoff     = row[3],
                                                         retrain    = row[4])
        
        #Append Valid AUC
        val_auc = roc_auc_score(train_target[valid_index],valid_preds)
        print("   Fold AUC: {}".format(val_auc))
        valid_aucs.append(val_auc)

    #Print Average Valid AUC, append
    cv_auc[i] = np.mean(valid_aucs)
    print("Iteration {} CV AUC: {}".format(i,np.mean(valid_aucs)))

#%% Grid Results

grid_results = np.concatenate((grid,cv_auc.reshape(-1,1)),axis=1)

#%% Final Call

train_preds, test_predictions = bayes_log_predictions(train_x, 
                                                      train_target, 
                                                      test_x,
                                                      col_ratio  = 0.7,
                                                      iterations = 10000,
                                                      C          = 0.5,
                                                      cutoff     = 0.1,
                                                      retrain    = 1)



#%% Final Train Eval
train_auc = roc_auc_score(train_target,train_preds)
print(" Train AUC: {}".format(train_auc))

#%% Create Submission

submission = pd.DataFrame({'id' : test_id, 'target' : test_predictions.ravel()})
submission.to_csv('submissions/bayes_submission.csv',index = False)