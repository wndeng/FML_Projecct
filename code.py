import numpy as np
import pandas as pd
import xgboost as xgb
from aux import *

train_raw = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

output = train_raw["Category"] # Output variable

# print(train.columns)

train = train_raw

# Drop output column as well as high categorical variables with high cardinality
train.drop("Address", axis=1, inplace=True)
train.drop("Descript", axis=1, inplace=True)
train.drop("Resolution", axis=1, inplace=True)
train.drop("Category", axis=1, inplace=True)


# Feature extraction for time data
date = pd.to_datetime(train.Dates).dt
train["Month"] = date.month
train["Hour"] = date.hour
train.drop("Dates", axis=1, inplace=True)

# One hot encoding for categorical var "PdDistrict"
encoding_district = pd.get_dummies(train["PdDistrict"])
train = pd.concat([train, encoding_district], axis=1)
train.drop("PdDistrict", axis=1, inplace=True)


# Feature extraction for day (split days into two categories: either Weekend or Weekday)
train["Weekend"] = [isWeekend(d) for d in train["DayOfWeek"]]
train["WeekDay"] = ~train.Weekend
train.drop("DayOfWeek", axis=1, inplace=True)

# Feature extraction for Month
encoding_month = pd.get_dummies(train["Month"])
train = pd.concat([train, encoding_month], axis=1)
train.drop("Month", axis=1, inplace=True)

# Feature extraction for hour
train["EarlyMorning"] = [0 <= d < 6 for d in train["Hour"]]
train["LateMorning"] = [6 <= d < 12 for d in train["Hour"]]
train["EarlyNight"] = [12 <= d < 18 for d in train["Hour"]]
train["LateNight"] = [18 <= d < 24 for d in train["Hour"]]
train.drop("Hour", axis=1, inplace=True)

print(train.columns)
print(train.head())