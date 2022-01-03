import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# loading dataset
df_dev = pd.read_csv("../datasets/NYC_Airbnb/development.csv")
df_eval = pd.read_csv("../datasets/NYC_Airbnb/evaluation.csv")
df_dev.drop(columns=["name", "host_name"], inplace=True)
df_eval.drop(columns=["name", "host_name"], inplace=True)

# fixing NaN values
df_dev.fillna(
    value = {
        "last_review": "2008-08-11",
        "reviews_per_month": 0
    },
    inplace=True
)
df_eval.fillna(
    value = {
        "last_review": "2008-08-11",
        "reviews_per_month": 0
    },
    inplace=True
)

# converting last_review feature to a numerical value representing days from the newest review
newest_review = datetime.today()
df_dev["last_review"] = pd.to_datetime(df_dev["last_review"]).apply(lambda x: abs((newest_review - x).days))
df_eval["last_review"] = pd.to_datetime(df_eval["last_review"]).apply(lambda x: abs((newest_review - x).days))

# one-hot encoding of categorical features
df_dev = pd.get_dummies(df_dev, columns=["room_type", "neighbourhood_group", "neighbourhood"], prefix=["room_type=", "neig_group=", "neig="])
df_eval = pd.get_dummies(df_eval, columns=["room_type", "neighbourhood_group", "neighbourhood"], prefix=["room_type=", "neig_group=", "neig="])

# getting dataset as numpy arrays
X_dev = df_dev.drop(columns=["id", "price"], inplace=False).values
y_dev = df_dev["price"].values

model = RandomForestRegressor(n_estimators=500, max_features="sqrt", max_depth=30, n_jobs=16)
model.fit(X_dev, y_dev)
