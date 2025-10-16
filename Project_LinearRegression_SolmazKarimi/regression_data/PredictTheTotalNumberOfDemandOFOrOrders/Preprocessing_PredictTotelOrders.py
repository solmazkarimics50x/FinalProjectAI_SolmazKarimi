import pandas as pd
import numpy as np
import missingno as ms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv("Daily_Demand_Forecasting_Orders.csv", sep=";")
print(df.head().to_string())
df.to_csv("Daily_Demand_Forecasting_TotalOrders.csv", index = False)
print(df.head().to_string())
# print(df.dtypes)
# print(df.describe().to_string())
# print(df.isnull().sum())

# ms.matrix(df)
# plt.show()
# print(df.corr().to_string())
# sns.heatmap(df.corr())
# plt.show()



X =df.drop(['Target (Total orders)'] , axis =1 )
y =df['Target (Total orders)']
print(df['Target (Total orders)'])
mms = MinMaxScaler()
X_scale = mms.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train, y_test = train_test_split(X_scale,y , shuffle= True,random_state= 42 , test_size= 0.3)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_predic= lr.predict(X_test)
# print(y_predic)
from sklearn.metrics import mean_squared_error , mean_absolute_error
MAE = mean_absolute_error(y_test ,y_predic)
MSE = mean_squared_error(y_test , y_predic)
RMSE = np.sqrt(MSE)
# Print the error metrics
# print("Mean Absolute Error (MAE):", MAE)# 3.315866100213801e-14
# print("Mean Squared Error (MSE):", MSE)#3.904335573573861e-27
# print("Root Mean Squared Error (RMSE):", RMSE)#6.24846827116363e-14
# test_residual = y_predic - y_test
# sns.scatterplot( x= y_test , y = test_residual)
# plt.axhline(y =0 , color ='red', linestyle = '--')
# plt.show()

final_model = LinearRegression()
final_model.fit(X_scale, y)
y_hat = final_model.predict(X_scale)

from joblib import dump , load
dump(final_model, "PredictTotalOrder.joblib")

# loaded_model = load('PredictTotalOrder.joblib')

# print(y_hat)
# Optionally, you can also evaluate the final model
final_MAE = mean_absolute_error(y, y_hat)
final_MSE = mean_squared_error(y, y_hat)
final_RMSE = np.sqrt(final_MSE)
# print("Final Model Mean Absolute Error (MAE):", final_MAE)#1.2410813117943082e-13
# print("Final Model Mean Squared Error (MSE):", final_MSE)#2.7599613537332464e-26
# print("Final Model Root Mean Squared Error (RMSE):", final_RMSE)#1.6613131413834197e-13

# --- 6. Extract coefficients and create DataFrame ---
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': final_model.coef_,
    'Absolute_Coefficient': np.abs(final_model.coef_)
}).sort_values('Absolute_Coefficient', ascending=False)
# --- 7. Create bar chart ---
# plt.figure(figsize=(12, 8))
# # Choose colors based on the sign of the coefficients
# colors = ['#2ecc71' if coef > 0 else '#e74c3c' for coef in coefficients['Coefficient']]
# # Draw horizontal bar chart
# bars = plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors, edgecolor='black')
# # Add numerical values to the bars
# for bar in bars:
#     width = bar.get_width()
#     label_pos = width * 1.02 if width > 0 else width * 0.98
#     plt.text(label_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')
# # # --- 8. Customize the chart ---
# # plt.title('Coefficients of the Linear Regression Model')
# # plt.xlabel('Coefficient')
# # plt.ylabel('Features')
# # plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Vertical line at zero
# # plt.grid(axis='x', linestyle='--', alpha=0.7)
# # plt.show()