import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from aux import *
import math

data, output, num_class = get_data()

	# seed = 2732 # Smaller sample for tuning model parameters
	# data, _, output, _ = train_test_split(data, output, test_size=0.95, random_state=seed)

# split 
seed = 1234
test_size = 0.3
train, test, y_train, y_test = train_test_split(data, output, test_size=test_size, random_state=seed)

model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=1, learning_rate=1)

model.fit(train, y_train)

result = model.staged_predict_proba(test)

for current_boost in result:
	total = 0
	loss = 0
	for prob_arr, ind in zip(current_boost, y_test):
		loss += math.log(prob_arr[ind])
		total += 1
	print(-1*loss/total) # Doesn't work, objective function is not similar to multiclass log loss