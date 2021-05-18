#%% Import Libraries

import gc
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import random, randint

from sklearn.model_selection import KFold, StratifiedKFold

#%% Import Data

train = pd.read_csv("data/train.csv")
test= pd.read_csv("data/test.csv")

#%% X,y Separation

train_id = train.id.values
train_target = train.target.values
train_x = train[[str(i) for i in range(300)]].values

test_id = test.id.values
test_x = test[[str(i) for i in range(300)]].values

#%% Log Likelihood

def neg_log_like(y,x,beta,intercept=True):
    
    if intercept == True:
        x_int = np.concatenate((np.full((x.shape[0],1),1),x),axis = 1)
        odds = np.exp(np.matmul(x_int,beta))   
    else:
        odds = np.exp(np.matmul(x,beta))
    
    #Calculate Predictions
    y_pred = odds / (1 + odds)
    
    #Neg_Log_Like
    return log_loss(y,y_pred,normalize=False,eps=0.00001)

#%% Penalized Log Likelihood
    
def pen_log_like(y,x,beta,intercept=True,l1 = 0.1, l2 = 0):
    
    if intercept == True:
        x_int = np.concatenate((np.full((x.shape[0],1),1),x),axis = 1)
        odds = np.exp(np.matmul(x_int,beta))   
    else:
        odds = np.exp(np.matmul(x,beta))
    
    #Calculate Predictions
    y_pred = odds / (1 + odds)
    
    #Penalty
    pen = np.sum((beta ** 2)) * l2 + np.sum(np.abs(beta)) * l1
    
    #Neg_Log_Like
    return log_loss(y,y_pred,normalize=False,eps=0.00001) + pen

#%% Newton Raphson for Gibbs
    
def nr_mean(y,x,beta,index,intercept=True,eps=0.01,conv=0.0001,max_iter=1000):
    
    iteration = 1
    grad = 1000
    while (np.abs(grad) > conv) and (iteration < max_iter):
        beta_low, beta_mid, beta_high = beta.copy(), beta.copy(), beta.copy()
        
        beta_low[index] = beta_low[index] - np.array(eps)
        beta_high[index] = beta_high[index] + np.array(eps)
        
        #nn_low = neg_log_like(y,x,beta_low)
        #nn_mid = neg_log_like(y,x,beta_mid)
        #nn_high = neg_log_like(y,x,beta_high)
        
        nn_low = pen_log_like(y,x,beta_low)
        nn_mid = pen_log_like(y,x,beta_mid)
        nn_high = pen_log_like(y,x,beta_high)
        
        grad = (nn_high - nn_low)/(2 * eps)
        hess = (nn_high - 2 * nn_mid + nn_low)/(eps**2)
        
        delta = grad/hess
        beta[index] = beta[index] - delta
        
        #print((beta_low[index],beta_mid[index],beta_high[index]))
        #print((nn_low,nn_mid,nn_high))
        #print((grad,hess,delta))
        iteration = iteration + 1
        
    return(beta)
    
#%% Gibbs
    
#initialize beta
beta = np.zeros((train_x.shape[1]+1,1))

#store past betas
samples = 1000
beta_hist = np.zeros((samples,beta.shape[0]))

#Initial Log_like
neg_log_like(train_target,train_x,beta)

# Gibbs Sampling
for sample in tqdm(range(samples)):

    #print("Iteration: {}".format(sample))
    
    #BootStrap Sample
    #random_rows = randint(0,train_x.shape[0],size=250)
    #sample_x = train_x[random_rows]
    sample_x = train_x
    
    #Loop Through Betas, Sample Beta
    for beta_index in range(beta.shape[0]):
        
        #Mean (MLE)
        beta = nr_mean(train_target,train_x,beta,beta_index)
        
        #Error
        beta[beta_index] += np.random.normal(0,0.30)
        
    #Save Iteration
    beta_hist[sample,:] = beta.ravel()
    
    
#%% Post-Processing Betas
    
#Average Beta post-sample 500
betas_burned = beta_hist[500:,:]
beta_means = np.mean(betas_burned,axis=0)
beta_stds = np.std(betas_burned,axis=0)

#%% Prediction

#Train
train_x_int = np.concatenate((np.full((train_x.shape[0],1),1),train_x),axis = 1)
train_odds = np.exp(np.matmul(train_x_int,beta_means))
train_preds = train_odds / (1 + train_odds)
train_auc = roc_auc_score(train_target,train_preds)
print("Train AUC: {}".format(train_auc))

#Test
test_x_int = np.concatenate((np.full((test_x.shape[0],1),1),test_x),axis = 1)
test_odds = np.exp(np.matmul(test_x_int,beta_means))
test_preds = test_odds / (1 + test_odds)

submission = pd.DataFrame({'id' : test_id, 'target' : test_preds.ravel()})
submission.to_csv('submissions/gibbs_submission.csv',index = False)