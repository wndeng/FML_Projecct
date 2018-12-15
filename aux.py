import numpy as np
import pandas as pd


def isWeekend(d):
	if(d == "Friday" or d == "Saturday" or d == "Sunday"):
		return True
	else:
		return False

def get_data(): # preprocess data
	data_raw = pd.read_csv("../data/train.csv") # Change for actual pwd

	data = data_raw
	data["PdDistrict"] = pd.Categorical(data.PdDistrict).codes
	data["Category"] = pd.Categorical(data.Category).codes
	output = data["Category"] # Output variable
	num_class = len(output.unique());

	# Drop output column as well as high categorical variables with high cardinality
	data.drop("Address", axis=1, inplace=True)
	data.drop("Descript", axis=1, inplace=True)
	data.drop("Resolution", axis=1, inplace=True)
	data.drop("Category", axis=1, inplace=True)


	# Feature extraction for time data
	date = pd.to_datetime(data.Dates).dt
	data["Month"] = date.month + 100
	data["Hour"] = date.hour
	data.drop("Dates", axis=1, inplace=True)

	# One hot encoding for categorical var "PdDistrict"
	encoding_district = pd.get_dummies(data["PdDistrict"])
	data = pd.concat([data, encoding_district], axis=1)
	data.drop("PdDistrict", axis=1, inplace=True)


	# Feature extraction for day (split days into two categories: either Weekend or Weekday)
	data["Weekend"] = [isWeekend(d) for d in data["DayOfWeek"]]
	data["WeekDay"] = ~data.Weekend
	data.drop("DayOfWeek", axis=1, inplace=True)

	# Feature extraction for Month
	encoding_month = pd.get_dummies(data["Month"])
	data = pd.concat([data, encoding_month], axis=1)
	data.drop("Month", axis=1, inplace=True)
	return data, output, num_class # returns features, output variable, and output class count

def get_data_lgbm(): # preprocess data
	data_raw = pd.read_csv("../data/train.csv") # Change for actual pwd

	data = data_raw
	data["PdDistrict"] = pd.Categorical(data.PdDistrict).codes
	data["Category"] = pd.Categorical(data.Category).codes
	output = data["Category"] # Output variable
	num_class = len(output.unique());

	# Drop output column as well as high categorical variables with high cardinality
	data.drop("Address", axis=1, inplace=True)
	data.drop("Descript", axis=1, inplace=True)
	data.drop("Resolution", axis=1, inplace=True)
	data.drop("Category", axis=1, inplace=True)

	# Feature extraction for time data
	date = pd.to_datetime(data.Dates).dt
	data["Month"] = date.month
	data["Hour"] = date.hour
	data.drop("Dates", axis=1, inplace=True)

	# Feature extraction for day (split days into two categories: either Weekend or Weekday)
	data["Weekend"] = [isWeekend(d) for d in data["DayOfWeek"]]
	data.drop("DayOfWeek", axis=1, inplace=True)

	return data, output, num_class # returns features, output variable, and output class count