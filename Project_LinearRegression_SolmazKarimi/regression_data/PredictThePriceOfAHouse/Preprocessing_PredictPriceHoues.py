import pandas as pd
import numpy as np
import missingno as ms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import lineStyles
from sklearn.preprocessing import MinMaxScaler


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_excel("Real estate valuation data set.xlsx")
#, usecols= ['X2 house age',
      # 'X3 distance to the nearest MRT station',
      # 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude','Y house price of unit area']
# print(df.head().to_string())
df.to_csv("Real estate valuation data set.csv", index= False)
print(df.head().to_string())


# Check for missing valuesPriceHouse
# print("Missing values in each column:")
# print(df.isnull().sum())
# Visualize missing values (if any)
# ms.matrix(df)
# plt.show()
# # Separate features and target variable
# X = df[['X2 house age', 'X3 distance to the nearest MRT station',
#          'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
# y = df['Y house price of unit area']
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# # Pipeline with scaling and polynomial features (degree 2)
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('regressor', LinearRegression())
# ])
# # Train model
# pipeline.fit(X_train, y_train)
# # Predict
# y_train_pred = pipeline.predict(X_train)
# y_test_pred = pipeline.predict(X_test)
# # Metrics
# def print_metrics(y_true, y_pred, dataset_name):
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     print(f"{dataset_name} RMSE: {rmse:.2f}")
#     print(f"{dataset_name} MAE: {mae:.2f}")
#     print(f"{dataset_name} R^2: {r2:.3f}\n")
#
# print_metrics(y_train, y_train_pred, "Train")
# print_metrics(y_test, y_test_pred, "Test")

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# Split the data into training and testing sets
# برش افقی
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=True, random_state=42, test_size=0.3)
# Print the first few rows of the training set
# print(X_train[:5])  # Display the first 5 rows of the training features
# print(X_test[:5])
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
# print(y_pred)
from sklearn.metrics import mean_squared_error , mean_absolute_error

MAE = mean_absolute_error(y_test ,y_pred)
MSE = mean_squared_error(y_test , y_pred)
RMSE = np.sqrt(MSE)
# Print the error metrics
# print("Mean Absolute Error (MAE):", MAE)# 6.151521273181801
# print("Mean Squared Error (MSE):", MSE)#74.05926303148075
# print("Root Mean Squared Error (RMSE):", RMSE)#8.605769171403608
#
# test_residual = y_pred - y_test
# # print(test_residual)
#
# # # sns.scatterplot(x= y_test , y = test_residual)
# # # plt.axhline( y = 0 , color = 'red', linestyle = "--")
# # # plt.show()
# # Final_model = LinearRegression()
# # Final_model.fit(X_scaled , y)
# # y_hat = Final_model.predict(X_scaled)
# #
# # from joblib import dump
# #
# # dump(Final_model, "PredictPriceHouse.joblib")
#
# # print(y_hat)
# #
# # # Optional: Visualize the predictions vs actual values
# # plt.figure(figsize=(10, 6))
# # plt.scatter(y_test, y_pred, alpha=0.7)
# # plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
# # plt.xlabel('Actual Prices')
# # plt.ylabel('Predicted Prices')
# # plt.title('Actual vs Predicted House Prices')
# # plt.show()
#
#
# # print(y_hat)
# # newdata = [[19.5, 390.5684,6,24.97937,121.54243]]
# # print(Final_model.predict(newdata))
