import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msg

df = pd.read_csv("wifi_localization.txt", sep ="\t", header = None)
df.columns = ['Wifi 1','Wifi 2','Wifi 3','Wifi 4','Wifi 5','Wifi 6','Wifi 7','Room']
print(df.head().to_string())
df.to_csv("wifi_localization.csv" , index = False)
print(df.head().to_string())
# print(df.dtypes)
# print(df.shape)
# print(df.describe().to_string())
# print(df.info())
# print(df.isna().sum())
# msg.matrix(df)
# plt.show()
# # Create a figure with subplots
# fig, axes = plt.subplots(4, 2, figsize=(12, 12))  # 3 rows, 2 columns#,constrained_layout=True
# # Flatten the axes array for easy iteration
# axes = axes.flatten()
# # List of columns to plot
# columns = ['Wifi 1','Wifi 2','Wifi 3','Wifi 4','Wifi 5','Wifi 6','Wifi 7']
# # Loop through the columns and create histograms
# for i, column in enumerate(columns):
#     sns.histplot(data=df, x=column, ax=axes[i], kde=True)
#     axes[i].set_title(f'Histogram of {column}')
#     axes[i].set_xlabel(column)
#     axes[i].set_ylabel('Frequency')
#
#     # Calculate skewness
#     skew_value = df[column].skew()
#     mean_value = df[column].mean()
#
#     # Annotate the skewness value on the plot
#     axes[i].text(0.95, 0.95, f'Skew: {skew_value:.2f}', transform=axes[i].transAxes,
#                  fontsize=12, verticalalignment='top', horizontalalignment='right',
#                  bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
#     axes[i].text(0.30, 0.95, f'Mean: {mean_value:.2f}', transform=axes[i].transAxes,
#                  fontsize=12, verticalalignment='top', horizontalalignment='right',
#                  bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
#
# plt.subplots_adjust(hspace=1.3, wspace=0.3)  # Adjust the spacing between rows and columns
# plt.show()
#df.columns = ['Wifi 1','Wifi 2','Wifi 3','Wifi 4','Wifi 5','Wifi 6','Wifi 7']

def remove_outliers(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 -Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[ (df[column] >= lower_bound) &(df[column] <= upper_bound) ])

ru_df = remove_outliers(df,'Wifi 1')
ru_df = remove_outliers(ru_df,'Wifi 2')
ru_df = remove_outliers(ru_df,'Wifi 3')
ru_df = remove_outliers(ru_df,'Wifi 4')
ru_df = remove_outliers(ru_df,'Wifi 5')
ru_df = remove_outliers(ru_df,'Wifi 6')
ru_df = remove_outliers(ru_df,'Wifi 7')
# plt.subplot(2,2,1)
# sns.histplot(data= df ,x ='Wifi 1')
# plt.title('Befor Remove Outliers  data Wifi 1')
#
# plt.subplot(2,2,2)
# sns.histplot(data= ru_df ,x ='Wifi 1')
# plt.title('After Remove Outliers  data  Wifi 1 ')
#
# plt.subplot(2,2,3)
# sns.histplot(data= df ,x ='Wifi 2')
# plt.title('Befor Remove Outliers  data Wifi 2 ')
#
# plt.subplot(2,2,4)
# sns.histplot(data= ru_df ,x ='Wifi 2')
# plt.title('After Remove Outliers  data Wifi 2 ')
# plt.tight_layout()
# plt.show()
#
# plt.subplot(2,2,1)
# sns.histplot(data= df ,x ='Wifi 3')
# plt.title('Befor Remove Outliers  data Wifi 3 ')
#
# plt.subplot(2,2,2)
# sns.histplot(data= ru_df ,x ='Wifi 3')
# plt.title('After Remove Outliers  data Wifi 3 ')
#
# plt.subplot(2,2,3)
# sns.histplot(data= df ,x ='Wifi 4')
# plt.title('Befor Remove Outliers  data Wifi 4 ')
#
# plt.subplot(2,2,4)
# sns.histplot(data= ru_df ,x ='Wifi 4')
# plt.title('After Remove Outliers  data Wifi 4 ')
#
# plt.tight_layout()
# plt.show()
#
# plt.subplot(2,3,1)
# sns.histplot(data= df ,x ='Wifi 5')
# plt.title('Befor Remove Outliers  data Wifi 5 ')
#
# plt.subplot(2,3,2)
# sns.histplot(data= ru_df ,x ='Wifi 5')
# plt.title('After Remove Outliers  data Wifi 5 ')
#
# plt.subplot(2,3,3)
# sns.histplot(data= df ,x ='Wifi 6')
# plt.title('Befor Remove Outliers  data Wifi 6 ')
#
# plt.subplot(2,3,4)
# sns.histplot(data= ru_df ,x ='Wifi 6')
# plt.title('After Remove Outliers  data Wifi 6 ')
#
# plt.subplot(2,3,5)
# sns.histplot(data= df ,x ='Wifi 7')
# plt.title('Befor Remove Outliers  data Wifi 7 ')
#
# plt.subplot(2,3,6)
# sns.histplot(data= ru_df ,x ='Wifi 7')
# plt.title('After Remove Outliers  data Wifi 7 ')
#
# plt.tight_layout()
# plt.show()
# Classification :KNN
X = df.drop('Room',axis = 1)
y =df['Room']
# print(X.head(10).to_string())
from sklearn.model_selection import train_test_split

X_train,X_test , y_train ,y_test = train_test_split(X,y,test_size=0.3,shuffle=True ,random_state= 42)
# print(len(X_train) , len(y_train))
# print(len(X_test) , len(y_test))

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_Train_Scaled = mms.fit_transform(X_train)
X_Test_Scaled = mms.transform(X_test)

from joblib import dump
# After fitting the scaler
dump(mms, "scaler_wifi.joblib")

#Model KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_Train_Scaled, y_train)
y_pred = knn_model.predict(X_Test_Scaled)
# y_pred = knn_model.predict_proba(X_Test_Scaled)

# print(y_pred)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test ,y_pred))
# # eblow method
# test_error_rates = []
#
# for k in range(2, 16):
#     knn_model = KNeighborsClassifier(n_neighbors=k)
#     knn_model.fit(X_Train_Scaled, y_train)
#     y_pred = knn_model.predict(X_Test_Scaled)
#     test_error = 1 - accuracy_score(y_test, y_pred)
#     test_error_rates.append(test_error)
#
# print(test_error_rates)
# plt.plot(range(2, 16), test_error_rates)
# plt.ylabel('Error Rate')
# plt.xlabel('K Value')
#
# plt.show()

final_model = KNeighborsClassifier()
final_model.fit(X,y)
y_hat = final_model.predict(X)

from joblib import dump
dump(final_model, "wifi_model.joblib")


