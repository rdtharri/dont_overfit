#%% Import Libraries

import gc
import time
import os
from tqdm import tqdm


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

from keras.layers import Dropout, Add, Multiply
from keras.layers import Input, Dense, BatchNormalization, LSTM
from keras.layers import GRU, Conv1D, Flatten, MaxPooling1D
from keras.models import Model
import keras
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.regularizers import l1

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

#%% Polynomial Features

#%% Bar Graph of Target

target_bins, target_count = np.unique(train_target,return_counts=True)
plt.bar([0,1],height=target_count)
plt.title("Target Counts")
plt.show()

#%% T-SNE on train

#Fit Manifold
train_tsne = TSNE().fit_transform(train_x)

#Plot
plt.scatter(x=train_tsne[:,0],y=train_tsne[:,1],c=train_target)
plt.title("TSNE-2Comp")
plt.show()


#%% T-SNE on full

#full matrices
full_x = np.concatenate((train_x,test_x))
full_target = np.concatenate((train_target,np.zeros((test_x.shape[0]))))

#Fit Manifold
#full_tsne = TSNE().fit_transform(full_x)

#Plot
#plt.scatter(x=full_tsne[:,0],y=full_tsne[:,1],c=full_target)
#plt.title("TSNE-2Comp")
#plt.show()

#%% Neural Network

def nnet_model(depth=5):
    
    K.clear_session()
    
    inp = Input((train_x.shape[1],))
    #att = Dense(300, activation = "softmax")(inp)
    #x = Multiply()([inp,att])
    #x = Dense(600)(inp)
    x = Dense(1, activation = "sigmoid",
              kernel_regularizer = keras.regularizers.l1(0.1))(x)
    
    
    #x = Dense(5, activation = "tanh")(inp)
    #x = Dense(1, activation = "sigmoid")(x)
    
    
    #Residual Attempt
    
    #x = Dense(5)(inp)
    
    #for i in range(depth):
    #    x_res = Dropout(0.4)(x)
    #    x_res = Dense(10, activation = "relu")(x_res)
    #    x_res = BatchNormalization()(x_res)
    #    x_res = Dense(5, activation = "relu")(x_res)
    #    x_res = BatchNormalization()(x_res)
    #    x = Add()([x,x_res])
        
        
    #x = Dense(1,activation = "sigmoid")(x)
      
    model = Model(inputs = [inp], outputs = [x])
    model.compile(optimizer = 'adam', 
                  loss = "binary_crossentropy")

    return(model)

#%% Fit models
    
#pre-allocate memory for test predictions
folds = 10
valid_predictions = np.zeros((len(train)))
test_predictions = np.zeros((len(test_x),folds))

#five-fold predictions are the best predictions
kf = StratifiedKFold(n_splits=folds)
k = 0
for train_index, valid_index in kf.split(train_x,train_target,train_target):
    print("Fold: ",k)
    
    #call model
    model = nnet_model(depth=5)

    # fits the model on batches with real-time data augmentation:
    model.fit(x = train_x[train_index,:], y = train_target[train_index],
              validation_data = (train_x[valid_index,:], train_target[valid_index]),
              callbacks = [EarlyStopping(monitor='val_loss',patience = 5)],
              epochs = 100, shuffle = True, batch_size = 5, verbose = 2)

    #predict on test
    valid_predictions[valid_index] = np.ravel(model.predict(train_x[valid_index,:]))
    print("Batch {} AUC: {}".format(k+1,
          roc_auc_score(train_target[valid_index], 
                        valid_predictions[valid_index])))
    test_predictions[:,k] = np.ravel(model.predict(test_x))
    k = k + 1
    
# Validation AUC
valid_auc = roc_auc_score(train_target, valid_predictions)
print(valid_auc)    

#%% Create Submission    
submission = test_predictions.mean(axis = 1)
submission = pd.DataFrame({'ID' : test_id, 'target' : submission})
submission.to_csv('nnet_submission.csv',index = False)
