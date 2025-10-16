import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msg



df = pd.read_csv("car.data", header = None)
df.columns = ['buying','maint','doors','persons','lug_boot','safety','class']
print(df.head().to_string())
# print(df.describe().to_string())
df.to_csv("car.csv" , index = False)

print(df['buying'].unique())
df['buying'] = df['buying'].map({'vhigh' : 4 , 'high': 3,'med' : 2,'low' : 1 })
print(df['maint'].unique())
df['maint'] = df['maint'].map({'vhigh' : 4 , 'high': 3,'med' : 2,'low' : 1})
print(df['doors'].unique())
df['doors'] = df['doors'].map({'2':2 , '3':3 , '4':4,'5more': 5})
print(df['persons'].unique())
df['persons'] = df['persons'].map({'2':2 , '4':4,'more': 5})
print(df['lug_boot'].unique())
df['lug_boot'] = df['lug_boot'].map({'small' : 0, 'med': 1, 'big':2 })
print(df['safety'].unique())
df['safety'] = df['safety'].map({'low' : 0, 'med': 1, 'high':2 })
print(df['class'].unique())
df['class'] = df['class'].map({'unacc' : 0 , 'acc': 1,'vgood' : 2,'good' : 3})
# Classification :KNN
X = df.drop('class',axis = 1)
y =df['class']


from sklearn.model_selection import train_test_split
X_train ,X_test ,y_train, y_test = train_test_split(X, y ,shuffle= True, random_state= 42, test_size= 0.3 )

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors= 5)
knn_model.fit(X_train,y_train)

y_pred = knn_model.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test , y_pred)) #k=5 ,accuracy = 0.9460500963391136 &k=3 ,accurecy :0.9152215799614644 &k=4 , accurecy :0.9210019267822736
# eblow method
test_error_rates = []

for k in range(2, 16):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_pred)
    test_error_rates.append(test_error)

print(test_error_rates)
plt.plot(range(2, 16), test_error_rates)
plt.ylabel('Error Rate')
plt.xlabel('K Value')
plt.show()

final_model = KNeighborsClassifier(n_neighbors=5)
final_model.fit(X,y)
y_pred = final_model.predict(X)

from joblib import  dump
dump(final_model, "PredictAcceptabilityCar.joblib")









# print(df.head().to_string())
# print(df.dtypes)
# print(df.shape)
# msg.matrix(df)
# # plt.show()
# print(df.describe().to_string())
# print(df.info())
# print(df.isna().sum())
# # Create a figure with subplots
# fig, axes = plt.subplots(4, 2, figsize=(12, 12))  # 3 rows, 2 columns#,constrained_layout=True
# # Flatten the axes array for easy iteration
# axes = axes.flatten()
# # List of columns to plot
# columns = ['buying','maint','doors','persons','lug_boot','safety']
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

