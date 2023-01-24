# Non Linear Regression

Non-Linear Regression: 

Logit function and interpretation 

Types of error measures (ROCR), 

Logistic Regression in classification —>

Logistic regression is **a statistical analysis method to predict a binary outcome, such as yes or no, based on prior observations of a data set**
. A logistic regression model predicts a dependent data variable by analyzing the relationship between one or more existing independent variables. ([https://www.techtarget.com/searchbusinessanalytics/definition/logistic-regression#:~:text=Logistic regression is a statistical,or more existing independent variables](https://www.techtarget.com/searchbusinessanalytics/definition/logistic-regression#:~:text=Logistic%20regression%20is%20a%20statistical,or%20more%20existing%20independent%20variables).)

Sigmoid function helps to convert an independent variable (x) into expression of probability.

![sigmoid_function.svg](Non%20Linear%20Regression%200c008995bfc6401d97ebdb019dd0258b/sigmoid_function.svg)

S(x) denotes probability.

**Types of logistic regression**

1. binary Logistic Regression
2. **Multinomial logistic regression**
3. Ordinal logistic Regression

[https://www.ibm.com/in-en/topics/logistic-regression#:~:text=Logistic regression estimates the probability,bounded between 0 and 1](https://www.ibm.com/in-en/topics/logistic-regression#:~:text=Logistic%20regression%20estimates%20the%20probability,bounded%20between%200%20and%201).

### **Binary Logistic Regression Major Assumptions**

1. The dependent variable should be dichotomous in nature (e.g., presence vs. absent).
2. There should be no outliers in the data, which can be assessed by converting the continuous predictors to standardized scores, and removing values below -3.29 or greater than 3.29.
3. There should be no high correlations ([multicollinearity](https://www.statisticssolutions.com/multicollinearity/)) among the predictors. This can be assessed by a correlation matrix among the predictors. Tabachnick and Fidell (2013) suggest that as long correlation coefficients among independent variables are less than 0.90 the assumption is met.

([https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/what-is-logistic-regression/](https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/what-is-logistic-regression/))

The multiple **binary logistic regression model** is the following:

![Untitled](Non%20Linear%20Regression%200c008995bfc6401d97ebdb019dd0258b/Untitled.png)

Refer —> [https://online.stat.psu.edu/stat462/node/207/](https://online.stat.psu.edu/stat462/node/207/)