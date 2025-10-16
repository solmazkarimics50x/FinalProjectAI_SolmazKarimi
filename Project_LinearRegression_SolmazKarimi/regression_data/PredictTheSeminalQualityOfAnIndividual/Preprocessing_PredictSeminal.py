import pandas as pd
import numpy as np
import missingno as ms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv("fertility_Diagnosis.txt", header =None)
df.columns  =['Season','Age' ,'Childish Diseases','Accident or serious trauma','Surgical intervention','High fevers in last year', 'Frequency of alcohol consumption','Smoking Habit','Number of hours spent sitting per day','Output']
# print(df.head().to_string())
# df.to_csv("fertility_Diagnosis.csv", index=False)
# print(df.head().to_string())


# print(df.columns)
# print(df.head().to_string())
# print

# Create dummy variables for the 'Output' column, dropping the first category
print(df['Output'].unique())
df = pd.get_dummies(data=df, columns=['Output'])
df.to_csv("fertility_Diagnosis.csv", index=False)
print(df.head().to_string())
# print(df.dtypes)
# print(df.describe().to_string())
# print(df.isnull().sum())
# print(df.corr().to_string())
# ms.matrix(df)
# plt.show()
# sns.heatmap(df.corr())
# plt.show()
#
X = df.drop(['Output_O'], axis= 1)
y = df['Output_O']

# print(X.shape)

from sklearn.model_selection import train_test_split


X_train,X_test , y_train ,y_test = train_test_split( X, y ,shuffle= True, random_state= 42, test_size= 0.3)

# print(X_train.shape)
# print(X_test.shape)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred =  lr.predict(X_test)
# print(y_pred)

from sklearn.metrics import mean_absolute_error ,mean_squared_error
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
# print(MAE)#0.20402335440265046
# print(MSE)#0.10848619826034982
# print(RMSE)#0.32937243093548346

test_residual = y_pred - y_test
# print(test_residual)
# sns.scatterplot( x = y_test , y = test_residual)
# plt.axhline(y = 0 , color = 'red' , linestyle = '--')
# plt.show()
final_model = LinearRegression()
final_model.fit(X,y)

y_hat = final_model.predict(X)
# newdata = [[1 , 0.64 , 1 ,1,1,0,0.2,0, 0.32]]
# print(final_model.predict(newdata))
from joblib import dump , load
dump(final_model, "PredictSeminal.joblib")
#
# from joblib import load
#
# loaded_model = load("PredictSeminal.joblib")
# newdata = [[1 , 0.64 , 1 ,1,1,0,0.2,0, 0.32]]
# print(loaded_model.predict(newdata))



# # Convert categorical target variable to numerical
# label_encoder = LabelEncoder()
# df['Output'] = label_encoder.fit_transform(df['Output'])  # 'N' -> 0, 'O' -> 1
# # Define features and target
# X = df.drop('Output', axis=1)
# y = df['Output']
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Train a logistic regression model (for binary classification)
# model = LogisticRegression()
# model.fit(X_train, y_train)
# # Evaluate the model
# accuracy = model.score(X_test, y_test)
# print(f"Model accuracy: {accuracy:.2f}")