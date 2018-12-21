import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from aux import *
import time
# xgboost 

data, output, num_class = get_data()

# seed = 2732 # Smaller sample for tuning model parameters
# data, _, output, _ = train_test_split(data, output, test_size=0.95, random_state=seed)

# split 
seed = 1234
test_size = 0.3
train, test, y_train, y_test = train_test_split(data, output, test_size=test_size, random_state=seed)

param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.4
param['num_class'] = num_class
param['eval_metric'] = 'mlogloss'
param['max_depth'] = 6
param['min_child_weight'] = 1
param['gamma'] = 0 
param['reg_alfa'] = 0.05
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['silent']= 1

dtrain = xgb.DMatrix(train, label=y_train)
dtest = xgb.DMatrix(test, label=y_test)

num_rounds = 31

t0 = time.time()
# model = xgb.train(param, dtrain, num_rounds, [(dtrain,'train'),(dtest,'eval')], early_stopping_rounds=5);
xgb.cv(param, dtrain, num_rounds, nfold=5, verbose_eval=True);
t1 = time.time()
print(t1-t0)