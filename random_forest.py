import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from aux import *
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

data, output, num_class = get_data()

# split 
seed = 1234
test_size = 0.3
train, test, y_train, y_test = train_test_split(data, output, test_size=test_size, random_state=seed)

seed = 4321

rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=seed)

rf.fit(train, y_train)

result = rf.predict_proba(test)

def obj_func(y_true, y_pred):
	total = 0
	loss = 0
	for prob_arr, ind in zip(y_pred, y_true):
		loss += math.log(max(min(prob_arr[ind], 1-10**(-15)), 10**(-15)))
		total += 1
	return (-1*loss/total)

scoring_func = make_scorer(obj_func, needs_proba=True)

print(cross_val_score(rf, train, y_train, cv=5, scoring=scoring_func))