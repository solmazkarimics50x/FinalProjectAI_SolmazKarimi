import pandas as pd
import numpy as np
import missingno as ms
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.constants import degree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error



df = pd.read_csv("auto-mpg.data", sep ='\\s+', header = None)
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']#
# print(df.head().to_string())
# print(df.dtypes)
# Convert 'horsepower' to numeric, forcing errors to NaN
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
# Convert categorical variables to numerical (if necessary)
df['car_name'] = df['car_name'].astype('category').cat.codes  # Convert car names to categorical codes
# print(df.dtypes)
# print(df.isnull().sum())
df = df.dropna(subset=['horsepower'])  # Drop rows with missing horsepower
# print(df.isnull().sum())
# print(df.dtypes)
print(df.head().to_string())
print(df.to_csv("auto-mpg.csv", index = False))
# print(df.head().to_string())
# Visualize missing data (if any)
# ms.matrix(df)
# plt.show()
# print(df.corr().to_string())
#
# sns.heatmap(df.corr())
# plt.show()
# # Step 4: Define features and target variable
X = df.drop('mpg', axis=1)  # Features
# X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']]
y = df['mpg']  # Target variable
# # Step 5: Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# # Convert the scaled features back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
# # Print the first few rows of the scaled DataFrame
# # print(X_scaled_df.head().to_string())
# # Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# # Step 6: Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)
# # Step 7: Make predictions
y_pred = model.predict(X_test)
# # Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Predicted MPG values:", y_pred)
print("Root Mean Squared Error (RMSE):", rmse) #3.308563538792321
test_residual = y_pred - y_test
print(test_residual)
sns.scatterplot( x = y_test , y = test_residual)
plt.axhline(y = 0 , color = 'red' , linestyle = '--')
plt.show()
final_model = LinearRegression()
final_model.fit(X_scaled,y)
y_hat = final_model.predict(X_scaled)

from joblib import  dump , load
dump(final_model, "FuelCar.joblib")






