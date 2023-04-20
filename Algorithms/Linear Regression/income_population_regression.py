import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Creating Sample dataset
X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
Y = np.array([2, 3, 5, 4, 5])

# Importing Linear Regression model
model = LinearRegression()
model.fit(X, Y)

# Print the coefficients of the model
print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)

# Predicting the output
x_test = np.array([6]).reshape((-1, 1))
y_pred = model.predict(x_test)
print('Predicted Output: ', y_pred)