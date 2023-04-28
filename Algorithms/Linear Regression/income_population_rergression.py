import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Importing Dataset
df = pd.read_csv('income_population.csv')

# Preparing the data
X = df[['Year', 'Population']].values
Y = df['Income'].values

# Linear Regression model
model = LinearRegression()

# Fit the data(train the model)
model.fit(X, Y)

# Predicting the output
new_data = pd.DataFrame({'Year': [2015], 'Population': [150]})
predicted_income = model.predict(new_data[['Year', 'Population']])
print(predicted_income)






