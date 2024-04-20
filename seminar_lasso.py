# Importing all the needed packages and the python file to run the DM-tests
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets

from dm_test import dm_test

data = pd.read_csv('/content/H1_gdp.csv')

target = data['Actual']
features = data.loc[:, 'ID_1':'ID_95']
Y = np.array(target)
X = np.array(features)
T,K = X.shape
X_bar = np.mean(X, axis=1)

Y_transform = Y - X_bar

#Applying KFold cross validation using Lasso model evaluate the lambda perforamnce
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def cross_validate_lasso(X, y, lambdas, n_splits=5):
    kf = KFold(n_splits=n_splits)
    average_mse_scores = []

    for alpha in lambdas:
        mse_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = Lasso(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)

        average_mse_scores.append(np.mean(mse_scores))

    # Find the lambda with the minimum average MSE
    min_mse_index = np.argmin(average_mse_scores)
    optimal_lambda = lambdas[min_mse_index]
    return optimal_lambda

lambdas = np.logspace(-50, 50, 100)

optimal_lambda = cross_validate_lasso(X, Y, lambdas)

print(optimal_lambda)

lasso = Lasso(alpha=optimal_lambda)
lasso.fit(X, Y)


# Get the coefficients of lasso model after fitting it and filtering the forecasters corresponding with coefficient 0
coef = lasso.coef_
beta = coef[coef !=0]
indices = np.nonzero(coef)[0]


print(indices)

#Making a new subset X matrix only containing the 'survivor' forecasters
indices_to_keep = [2, 9, 21]
X_subset = X[:, indices_to_keep]
X_bar_2 = np.mean(X_subset, axis=1)
Y_transform_2 = Y - X_bar_2


# Creating matrices and vectors to calculate RMSE and use it for the models using RMSE as step 2
rmse = np.zeros(3)
denominator = 0
weight = np.zeros(3)
Y_subset_weight = np.zeros(T)
X_subset_multiply_weight = np.zeros((T,3))
c = 0.5

for i in range(3):
  rmse[i] = math.sqrt(np.square(np.subtract(X_subset[:,i],Y)).mean())
  denominator = denominator + math.exp(c/rmse[i])


for i in range(3):
  weight[i] = math.exp(c/rmse[i]) / denominator
  X_subset_multiply_weight = weight[i] * X_subset[:,i]


Y_subset_weight = Y - X_subset_multiply_weight[0] - X_subset_multiply_weight[1] - X_subset_multiply_weight[2]

X_multiply_beta = np.zeros((T,K))

for i in range(K):
  X_multiply_beta[:,i] = coef[i] * X[:,i]

#Applying KFold cross validation using Ridge model evaluate the lambda perforamnce

def cross_validate_ridge(X, y, lambdas, n_splits=5):
    kf = KFold(n_splits=n_splits)
    average_mse_scores = []

    for alpha in lambdas:
        mse_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)

        average_mse_scores.append(np.mean(mse_scores))

    # Find the lambda with the minimum average MSE
    min_mse_index = np.argmin(average_mse_scores)
    optimal_lambda = lambdas[min_mse_index]
    return optimal_lambda

#Applying KFold cross validation using Elastic Net model evaluate the lambda perforamnce

def cross_validate_en(X, y, lambdas, n_splits=5):
    kf = KFold(n_splits=n_splits)
    average_mse_scores = []

    for alpha in lambdas:
        mse_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = ElasticNet(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)

        average_mse_scores.append(np.mean(mse_scores))

    # Find the lambda with the minimum average MSE
    min_mse_index = np.argmin(average_mse_scores)
    optimal_lambda = lambdas[min_mse_index]
    return optimal_lambda

#Tuning the lambda parameter for the second time, using it to run the step 2 models.
optimal_lambda_2 = cross_validate_en(X_subset, Y, lambdas)


# for Egalitarian models use X_subset and normal Y as input, for RMSE models use X_Subset and Y - Y_subset_weight as input
#Change cross_validate_x with x refering to the varying step 2 model (with x = {lasso, ridge, en})

print(optimal_lambda_2)

#Step 1 Lasso/Step 2 eLasso
model_Lasso_ela = Lasso (alpha = optimal_lambda_2, max_iter = 1000000)
model_Lasso_ela.fit(X_subset, Y_transform_2)
coef_lasso_ela = model_Lasso_ela.coef_

coef_lasso_ela_transform = coef_lasso_ela + 1/3
prediction_elalasso = np.dot(X_subset,coef_lasso_ela_transform)

#Step 1 Lasso/Step 2 eLasso
from sklearn import metrics

from sklearn.metrics import mean_squared_error

rmse_elasso = mean_squared_error(Y, prediction_elalasso, squared=False)
print(rmse_elasso)

#Step 1 Lasso/Step 2 eRidge
model_Ridge_ela = Ridge(alpha = optimal_lambda_2, max_iter = 1000000)
model_Ridge_ela.fit(X_subset, Y_transform_2)
coef_Ridge_ela = model_Ridge_ela.coef_

coef_Ridge_ela_transform = coef_Ridge_ela + 1/3
prediction_elaridge = np.dot(X_subset,coef_Ridge_ela_transform)

#Step1 Lasso/Step 2 eRidge
rmse_ridge = mean_squared_error(Y, prediction_elaridge, squared=False)
print(rmse_ridge)

#Step 1Lasso/Step 2 eEN
model_EN_ela = ElasticNet(alpha = optimal_lambda_2, max_iter = 1000000)
model_EN_ela.fit(X_subset, Y_transform_2)
coef_EN_ela = model_EN_ela.coef_

coef_EN_ela_transform = coef_EN_ela + 1/3
prediction_ela_en= np.dot(X_subset,coef_EN_ela_transform)

#Step 1 Lasso/Step 2 eEN
mse_ela_en = mean_squared_error(Y, prediction_ela_en, squared=False)
print(mse_ela_en)

#Step 1 Lasso/Step 2 RMSELasso
model_lasso_rmse = Lasso(alpha = optimal_lambda_2, max_iter = 1000000)
model_lasso_rmse.fit(X_subset, Y_subset_weight)
coef_lasso_rmse = model_lasso_rmse.coef_

prediction_lasso_rmse= np.dot(X_subset,coef_lasso_rmse)

#Step 1 Lasso/Step 2 RMSELasso
mse_lasso_mrse = mean_squared_error(Y, prediction_lasso_rmse, squared=False)
print(mse_lasso_mrse)

#Step 1 Lasso/Step 2 RMSERidge
model_ridge_rmse = Ridge(alpha = optimal_lambda_2, max_iter = 1000000)
model_ridge_rmse.fit(X_subset, Y_subset_weight)
coef_ridge_rmse = model_ridge_rmse.coef_

prediction_ridge_rmse= np.dot(X_subset,coef_ridge_rmse)

#Step 1 Lasso/Step 2 RMSERidge
mse_ridge_mrse = mean_squared_error(Y, prediction_ridge_rmse, squared=False)
print(mse_ridge_mrse)

#Step 1 Lasso/Step 2 RMSEEN
model_en_rmse = ElasticNet(alpha = optimal_lambda_2, max_iter = 1000000)
model_en_rmse.fit(X_subset, Y_subset_weight)
coef_en_rmse = model_en_rmse.coef_

prediction_en_rmse= np.dot(X_subset,coef_ridge_rmse)

#Step 1 Lasso/Step 2 RMSEEN
mse_en_mrse = mean_squared_error(Y, prediction_en_rmse, squared=False)
print(mse_en_mrse)

#Lasso and Simple Average
mse_lasso_average = mean_squared_error(Y, X_bar_2, squared=False)
print(mse_lasso_average)

#Performing the DM-tests. Change the prediction_x_rmse with varying x to the model you would like to compare to.
X_mean = np.mean(X, axis=1)

rt = dm_test(Y,prediction_en_rmse,X_mean,h = 1, crit="MSE")
print(rt)
