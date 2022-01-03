from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# generate dataset with 2000 samples and 100 features
X, y = make_regression(n_samples=2000, n_features=100, n_informative=80, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, shuffle=True)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
R2 = r2_score(y_test, y_pred)
print("R^2: ", R2)
MAE = mean_absolute_error(y_test, y_pred)
print("MAE: ", MAE)
MSE = mean_squared_error(y_test, y_pred)
print("MSE: ", MSE)
