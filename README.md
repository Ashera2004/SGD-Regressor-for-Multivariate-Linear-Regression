# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Algorithm for Predicting House Price and Number of Occupants Using SGD Regressor  

1. **Data Collection & Preprocessing**  
   - Collect housing data, including features such as size, number of rooms, location, and past prices.  
   - Handle missing values, normalize/standardize numerical data, and encode categorical data if necessary.  
   - Split the dataset into training and testing sets.  

2. **Initialize the SGD Regressor Model**  
   - Import `SGDRegressor` from `sklearn.linear_model`.  
   - Set hyperparameters such as the learning rate, maximum iterations, and penalty terms for regularization.  

3. **Train the Model**  
   - Fit the model to the training data using Stochastic Gradient Descent (SGD).  
   - The model will iteratively update weights using small batches of data to minimize the error.  

4. **Evaluate Model Performance**  
   - Use Mean Squared Error (MSE) or R² score to assess model accuracy.  
   - Fine-tune hyperparameters if needed (e.g., adjusting the learning rate or number of iterations).  

5. **Make Predictions**  
   - Provide new input data (house features) to the trained model.  
   - Output the predicted house price and estimated number of occupants.  


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#load the california housing dataset
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

# use the first 3 features as inputs
X= df.drop(columns=['AveOccup','HousingPrice']) #data[:,:3] #Features: 'MedInc', 'HouseAge', 'AveRooms'
#Use 'medHouseVal' and 'Aveoccup' as output variables
Y=df[['AveOccup','HousingPrice']]  #np.column_stack((data.target,data.data[:,6]))  #Targets:'MedHouseVal', 'AveOccup'

#split the data into training and testing sets
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2,random_state=42)

#scale the features and target variables
scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

#initialize the SGDRegressor
sgd=SGDRegressor(max_iter=1000, tol=1e-3)

#Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd= MultiOutputRegressor(sgd)

#train the model
multi_output_sgd.fit(X_train,Y_train)

#predict on the test data
Y_pred= multi_output_sgd.predict(X_test)

#inverse transform the predictions to get them back to the original scale
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)

#evaluate the model using mean squared error
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)

#optionally, print some predictions
print("\nPredictions:\n",Y_pred[:5]) #print first 5 predictions

Developed by: A S Siddarth
RegisterNumber: 212224040316
*/
```

## Output:

![Screenshot 2025-03-30 180519](https://github.com/user-attachments/assets/8d93162b-6df7-4ef4-80cb-3a23298f0222)

![Screenshot 2025-03-30 180530](https://github.com/user-attachments/assets/acb55be5-6210-4300-b7ab-734f8a9b7ef2)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
