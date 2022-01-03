import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def inject_noise(y):
    return y + np.random.normal(0, 50, size=y.size)

# defining functions
f1 = lambda x: x * np.sin(x) + 2*x
f2 = lambda x: 10 * np.sin(x) + np.power(x, 2)
f3 = lambda x: np.sign(x) * (np.power(x, 2) + 300) + 20 * np.sin(x)

functions = [f1, f2, f3]
NOISE = False

# setting dataset parameters
tr = 20
n_samples = 100
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Original functions")

for i in range(len(functions)):
    function_name = "FUNCTION " + str(i+1)
    print(function_name)
    f = functions[i]
    # generating dataset
    X = np.linspace(-tr, tr, n_samples)
    y = inject_noise(f(X)) if NOISE == True else f(X)
    
    # plotting function
    ax[i].plot(X, y)
    ax[i].set_title(function_name)

    # splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, shuffle=True)
    
    # sorting test set
    y_test = y_test[X_test.argsort()]
    X_test.sort()

    X_train = X_train.reshape(-1, 1)
    Y_train = X_train.reshape(-1, 1)
    X_test = X_train.reshape(-1, 1)
    y_test = X_train.reshape(-1, 1)

    # trying with linear regressor
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    print("LINEAR REGRESSION SCORES:")
    R2 = r2_score(y_test, y_pred)
    print("R^2: ", R2)
    MAE = mean_absolute_error(y_test, y_pred)
    print("MAE: ", MAE)
    MSE = mean_squared_error(y_test, y_pred)
    print("MSE: ", MSE)

    # plotting predicted functions
    ax[i].scatter(X_test, y_pred, c='red', label="linear regression predictions")
    ax[i].legend()

    # LINEAR REGRESSION WORKS FINE ONLY FOR FUNCTION 1

    # trying with polynomial regression with Lasso
    poly_lasso_reg = make_pipeline(PolynomialFeatures(5), Lasso(alpha=0.5))
    poly_lasso_reg.fit(X_train, y_train)
    y_pred = poly_lasso_reg.predict(X_test)
    print("POLYNOMIAL LASSO REGRESSION SCORES:")
    R2 = r2_score(y_test, y_pred)
    print("R^2: ", R2)
    MAE = mean_absolute_error(y_test, y_pred)
    print("MAE: ", MAE)
    MSE = mean_squared_error(y_test, y_pred)
    print("MSE: ", MSE)

    # plotting predicted functions
    ax[i].scatter(X_test, y_pred, c='green', label="polynomial lasso regression predictions")
    ax[i].legend()

plt.tight_layout()
plt.show()
