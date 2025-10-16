import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msg



df = pd.read_csv("Qualitative_Bankruptcy.data.txt" ,header = None)

df.columns = ['Industrial Risk','Management Risk','Financial Flexibility','Credibility',
              'Competitiveness','Operating Risk','Class']
print(df.head().to_string())
df.to_csv("Qualitative_Bankruptcy.csv" , index = False)
# print()
# print(df['Industrial Risk'].unique())
df['Industrial Risk'] = df['Industrial Risk'].map({'P':2 ,'N':0 ,'A':1 })
# print(df['Management Risk'].unique())
df['Management Risk'] = df['Management Risk'].map({ 'P':2 ,'N':0 ,'A':1})
# print(df['Financial Flexibility'].unique())
df['Financial Flexibility'] = df['Financial Flexibility'].map({'A':1,'P':2 ,'N':0 })
# print(df['Credibility'].unique())
df['Credibility'] =df['Credibility'].map({'A':1,'P':2 ,'N':0})
# print(df['Competitiveness'].unique())
df['Competitiveness'] =df['Competitiveness'].map({'A':1,'P':2 ,'N':0})
# print(df['Operating Risk'].unique())
df['Operating Risk'] =df['Operating Risk'].map({'P':2 ,'N':0 ,'A':1})
# print(df['Class'].unique())
df['Class'] =df['Class'].map({'NB':0, 'B':1})

# print(df.isna().sum())
# print(df.describe())



#Classification KNN
X= df.drop('Class', axis= 1)
y = df['Class']
###
from sklearn.model_selection import train_test_split

X_train ,X_test ,y_train ,y_test = train_test_split(X,y,shuffle =True,random_state= 42 , test_size= 0.3)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,y_train)

y_pred = knn_model.predict(X_test)
# print(y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test ,y_pred)) # k= 5 , accuracy = 1.0

# elbow method
test_error_rates = []

for k in range(2,16):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train,y_train)
    y_pred = knn_model.predict(X_test)
    test_error = 1 - accuracy_score(y_test,y_pred)
    test_error_rates.append(test_error)

print(test_error_rates)
plt.plot(range(2,16),test_error_rates)
plt.ylabel('Error Rate')
plt.xlabel('K Value')
plt.show()

final_model = KNeighborsClassifier(n_neighbors= 5)
final_model.fit(X,y)
y_hat =final_model.predict(X)

from joblib import dump
dump(final_model , 'PredictBankruptcy.joblib')


















# print(df.head().to_string())
# print(df.dtypes)
# print(df.shape)
# print(df.describe().to_string())
# print(df.info())
# print(df.isna().sum())
# msg.matrix(df)
# plt.show()
