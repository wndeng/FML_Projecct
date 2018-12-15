import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from aux import *

data, output, num_class = get_data_lgbm()

# seed = 2732 # Smaller sample for tuning model parameters
# data, _, output, _ = train_test_split(data, output, test_size=0.95, random_state=seed)

# split 
seed = 1234
test_size = 0.3
train, test, y_train, y_test = train_test_split(data, output, test_size=test_size, random_state=seed)

dtrain = lgb.Dataset(train, y_train, categorical_feature=[0,3,5])
dtest = lgb.Dataset(test, y_test, categorical_feature =[0,3,5], reference=dtrain)

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_leaves': 200,
    'learning_rate': 0.07,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'num_class': num_class,
    'min_data_in_leaf': 600,
    'verbose': -1,
}


gbm = lgb.cv(params, dtrain, num_boost_round=100, nfold=5, early_stopping_rounds=5, verbose_eval=True)
#gbm = lgb.train(params, dtrain, num_boost_round=100, valid_sets=dtest, early_stopping_rounds=5)