# SGD-Regressor-for-Multivariate-Linear-Regression
### NAME: PAVITHRA
### REG NO: 212223220073


## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select features and targets, and split into training and testing sets.

2.Scale both X (features) and Y (targets) using StandardScaler.

3.Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.

4.Predict on test data, inverse transform the results, and calculate the mean squared error.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:Pavithra.S 
RegisterNumber:212223220073  
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
df.head()

x=df.drop(columns=['AveOccup','HousingPrice'])

y=df[['AveOccup','HousingPrice']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)


multi_output_sgd=MultiOutputRegressor(sgd)


multi_output_sgd.fit(x_train,y_train)

y_pred=multi_output_sgd.predict(x_test)


y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)

mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])
```
## Output:
![image](https://github.com/user-attachments/assets/b98271c9-4eb8-479f-a739-b1d572e9e912)

 ## MSE & PREDICTION:

 ![image](https://github.com/user-attachments/assets/3e816cda-bae9-4986-8a2e-b73298137c62)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
