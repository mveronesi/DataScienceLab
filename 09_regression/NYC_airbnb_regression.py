import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# loading dataset
df_dev = pd.read_csv("../datasets/NYC_Airbnb/development.csv")
# df_eval = pd.read_csv("../datasets/NYC_Airbnb/evaluation.csv")
df_dev.drop(columns=["name", "host_name", "neighbourhood"], inplace=True)

# fixing NaN values
df_dev.fillna(
    value = {
        "last_review": "1971-01-01",
        "reviews_per_month": 0
    },
    inplace=True
)

# converting last_review feature to a numerical value representing days from the newest review
last_reviews = pd.to_datetime(df_dev["last_review"])
newest_review = datetime.today()
df_dev["last_review"] = last_reviews.apply(lambda x: abs((newest_review - x).days))

# one-hot encoding of categorical features
df_dev = pd.get_dummies(df_dev, columns=["room_type", "neighbourhood_group"], prefix=["room_type=", "neig_group="])

# getting dataset as numpy arrays
X_dev = df_dev.drop(columns=["id", "price"], inplace=False).values
y_dev = df_dev["price"].values

X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.2)

# separating categorical features from numerical ones
X_train_categorical = X_train[:, -8:]
X_test_categorical = X_test[:, -8:]
X_train_numerical = X_train[:, :-8]
X_test_numerical = X_test[:, :-8]

# feature normalization
scaler = StandardScaler()
scaler.fit(X_train_numerical)
X_train_numerical = scaler.transform(X_train_numerical)
X_test_numerical = scaler.transform(X_test_numerical)

# recomposing X_train and X_test
X_train = np.hstack((X_train_numerical, X_train_categorical,))
X_test = np.hstack((X_test_numerical, X_test_categorical,))

# testing random forest
model = RandomForestRegressor(n_estimators=500, max_features="sqrt", max_depth=20, n_jobs=16)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(score)
