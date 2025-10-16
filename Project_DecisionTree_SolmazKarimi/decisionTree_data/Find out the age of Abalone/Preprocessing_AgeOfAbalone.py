import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms



df = pd.read_csv("abalone.data.csv", header= None)
df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
# print(df.head().to_string())
# print(df['Sex'].unique())
# Perform one-hot encoding on the 'Sex' column
df_encoded = pd.get_dummies(df, columns=['Sex']) #
# Display the resulting DataFrame
# print(df_encoded.head().to_string())
# print(df_encoded.dtypes)
# print(df_encoded.columns)
# Specify the new order of columns
new_order = ['Sex_F', 'Sex_I', 'Sex_M', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
# # Reorder the DataFrame
df_reordered = df_encoded[new_order]
# df_reordered = df_reordered.drop(['Sex_I'], axis = 1)
# Display the resulting DataFrame
# print(df_reordered.head().to_string())
# print(df_reordered.isnull().sum())

# Save to CSV file
df_reordered.to_csv("preprocessed_abalone_data.csv", index=False)
df = pd.read_csv("preprocessed_abalone_data.csv")
print(df.head().to_string())

# ms.matrix(df_reordered)
# plt.show()
# print(df_reordered.corr().to_string())
# sns.heatmap(df_reordered.corr())
# plt.show()
# print(df_reordered.dtypes)
# print(df_reordered.describe().to_string())
# # print(df_reordered.shape)
# print("\nData quality check Diameter:")
# print(df['Diameter'].describe())
# print("\nMissing values:", df['Diameter'].isnull().sum())
# if df['Diameter'].std() < 0.1:
#     print("\nWarning: Low variance (may not be useful for clustering)") #std= 0.099240# correlation :Rings VS 0.574660
# print("\nData quality check Height:")
# print(df['Height'].describe())
# print("\nMissing values:", df['Height'].isnull().sum())
# if df['Height'].std() < 0.1:
#     print("\nWarning: Low variance (may not be useful for clustering)") #std= 0.041827# correlation :Rings VS 0.557467

# # Create a figure with subplots
# fig, axes = plt.subplots(5, 2, figsize=(12, 12))  # 3 rows, 2 columns#,constrained_layout=True
# # Flatten the axes array for easy iteration
# axes = axes.flatten()
# # List of columns to plot
# columns = ['Sex_M', 'Sex_I', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
# # Loop through the columns and create histograms
# for i, column in enumerate(columns):
#     sns.histplot(data=df_reordered, x=column, ax=axes[i], kde=True)
#     axes[i].set_title(f'Histogram of {column}')
#     axes[i].set_xlabel(column)
#     axes[i].set_ylabel('Frequency')
#
#     # Calculate skewness
#     skew_value = df_reordered[column].skew()
#
#     # Annotate the skewness value on the plot
#     axes[i].text(0.95, 0.95, f'Skew: {skew_value:.2f}', transform=axes[i].transAxes,
#                  fontsize=12, verticalalignment='top', horizontalalignment='right',
#                  bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
#
# plt.subplots_adjust(hspace=1.3, wspace=0.3)  # Adjust the spacing between rows and columns
# plt.show()

# print(df_reordered.head().to_string())
# برش عمودی
X = df_reordered.drop('Rings', axis = 1)
y = df['Rings']

# برش افقی
from sklearn.model_selection import train_test_split

X_trian , X_test, y_train  , y_test =  train_test_split(X,y, shuffle= True , random_state= 42 , test_size= 0.3)
# print(X_trian.head().to_string())
# print((X_test.head().to_string()))
# print( X_trian.shape)
# print(X_test.shape)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_trian,y_train)
y_prediction = lr.predict(X_test)
# print(y_prediction)
from sklearn.metrics import mean_squared_error, mean_absolute_error

MAE = mean_absolute_error(y_test, y_prediction)
MSE = mean_squared_error(y_test, y_prediction)
RMSE = np.sqrt(MSE)
# print(MAE)#1.5832377389437269
# print(MSE)#4.784788642609111
# print(RMSE)#2.1874159738397063
test_residual = y_prediction - y_test
# print(test_residual)
# sns.scatterplot(x = y_test ,y = test_residual )
# plt.axhline(y = 0 , color = 'red', linestyle = '--')
# plt.show()
final_model = LinearRegression()
final_model.fit(X.values,y)
y_hat = final_model.predict(X.values)

from joblib import dump
dump(final_model, "abalone_model.joblib")
#print(y_hat)
# New sample for testing the model
# new_sample = [[False, True, 0.400, 0.320, 0.100, 0.3000, 0.1200, 0.0600, 0.090]]  # [Sex_M, Sex_I, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight]
# # Make a prediction using the final model
# predicted_rings = final_model.predict(new_sample)
# print(predicted_rings)
# newdata = [[1 , 0 , 0.5 ,0.4,0.1,0.5,0.23,0.1, 0.2]]
# print(final_model.predict(newdata))

#Variance Inflation Factor (VIF):
# from statsmodels.stats.outliers_influence import variance_inflation_factor
#
#
# def calc_vif(X):
#     vif = pd.DataFrame()
#     vif["variables"] = X.columns
#     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#     return vif
#
#
# # Calculate VIF for your features
# vif_results = calc_vif(X_trian)
# print(vif_results)

#Correlation Matrix:
# plt.figure(figsize=(10, 8))
# sns.heatmap(X_trian.corr(), annot=True, fmt=".2f", cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()
# #Model Validation
# #Cross-Validation:
# from sklearn.model_selection import cross_val_score
#
# scores = cross_val_score(lr, X_trian, y_train, cv=5, scoring='neg_mean_squared_error')
# rmse_scores = np.sqrt(-scores)
# print("Cross-validated RMSE scores:", rmse_scores)
# print("Mean RMSE:", rmse_scores.mean())
#
# #Feature Importance
# feature_importance = pd.DataFrame({'Feature': X_trian.columns, 'Coefficient': lr.coef_})
# print(feature_importance.sort_values(by='Coefficient', ascending=False))

# #Permutation Importance
# from sklearn.inspection import permutation_importance
#
# result = permutation_importance(lr, X_trian, y_train, n_repeats=10, random_state=42)
# sorted_idx = result.importances_mean.argsort()
#
# plt.barh(X_trian.columns[sorted_idx], result.importances_mean[sorted_idx])
# plt.xlabel("Permutation Importance")
# plt.title("Feature Importance")
# plt.show()
