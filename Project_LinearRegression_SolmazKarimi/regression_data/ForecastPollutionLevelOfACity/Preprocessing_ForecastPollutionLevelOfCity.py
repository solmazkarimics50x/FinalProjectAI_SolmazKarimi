import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from joblib import dump , load

# file_path = "PRSA_data_2010.1.1-2014.12.31.csv"
def load_data(file_path):
    df = pd.read_csv(file_path, usecols =['pm2.5','DEWP','TEMP','PRES', 'cbwd','Iws','Is','Ir'])
    df.to_csv("PRSA_data.csv", index =False)
    return df

def visualize_missing_data(df):
    ms.matrix(df)
    plt.show()

def impute_missing_values (df):
    # df = df.dropna(subset = ['pm2.5'])
    df['pm2.5'] = df['pm2.5'].fillna(df['pm2.5'].mean())
    return df
def visualize_impute_missing_data(df):
    ms.matrix(df)
    plt.show()
def map_categorical_features(df):
    df['cbwd'] = df['cbwd'].map({'NW':0, 'cv':1, 'NE':2, 'SE':3})
    return df

def remove_outliers(df, column):

    Q1 , Q3 = np.array(df[column].quantile([0.25,0.75]))
    IQR = Q3 -Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[ (df[column] >= lower_bound) &  (df[column] <= upper_bound) ]

def remove_outliers_from_dataframe(df):
    for column in ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws']:  # ,,'Is','Ir'
        df = remove_outliers(df, column)
    return df


def scale_features(X):
    mms = MinMaxScaler()
    X_scaled = mms.fit_transform(X)
    dump(mms, "scaler.joblib")  # Save the fitted scaler
    return X_scaled

def train_test_split_data(X,y):
    return train_test_split(X, y, shuffle=True, random_state=42, test_size=0.2)

def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def evaluate_model(y_test, y_pred):
    MAE = mean_absolute_error(y_test,y_pred)
    MSE = mean_squared_error(y_test,y_pred)
    RMSE = np.sqrt(MSE)
    Accuracy = r2_score(y_test, y_pred)

    return  MAE,MSE,RMSE,Accuracy

def plot_coefficients (model , features_name):
    coefficients = pd.DataFrame({
        'Feature': features_name,
        'Coefficients': model.coef_ ,
        'Absolute Coefficients': np.abs(model.coef_)
    }).sort_values('Absolute Coefficients',ascending = False )

    plt.figure(figsize= (12,8))
    colors = ['#2ecc71' if coef > 0 else '#e74c3c' for coef in coefficients['Coefficients'] ]
    # y = coefficients['Feature']  & width = coefficients['Coefficients'] & height = 0.8
    #plt.barh(y,width,...)
    bars = plt.barh(coefficients['Feature'], coefficients['Coefficients'],color = colors , edgecolor = 'black')
    # plt.text(x,y,s=str,va=)
    #x= lable_pos
    #y= bar.get_y() + bar.get_height() /2
    # str = f'{width:.2f}'
    for bar in bars:
        width = bar.get_width()
        label_pos = width * 1.02 if width > 0 else width *0.98
        plt.text (label_pos, bar.get_y() + bar.get_height() /2 , f'{width:.2f}' , va = 'center' )

    plt.title('Coefficients of the Linear Regression Model')
    plt.xlabel('Coefficients')
    plt.ylabel('Features')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()


def main():
    # Load data
    df = load_data("PRSA_data_2010.1.1-2014.12.31.csv")
    print(df.describe().to_string())

    # Visualize missing data
    visualize_missing_data(df)

    # Impute missing values
    df = impute_missing_values(df)

    # Visualize impute missing data
    visualize_impute_missing_data(df)

    # Map categorical features
    df = map_categorical_features(df)

    # Remove outliers
    df = remove_outliers_from_dataframe(df)

    # Prepare features and target variable
    X = df.drop('pm2.5', axis=1)
    y = df['pm2.5']

    # Scale features
    X_scaled = scale_features(X)



    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

    # Train linear regression model
    lr = train_linear_regression(X_train, y_train)

    # Make predictions
    y_pred_lr = lr.predict(X_test)

    # Evaluate model
    MAE, MSE, RMSE,Accuracy = evaluate_model(y_test, y_pred_lr)
    print("Mean Absolute Error (MAE):", MAE)
    print("Mean Squared Error (MSE):", MSE)
    print("Root Mean Squared Error (RMSE):", RMSE)
    print("R² Score (Coefficient of Determination):",Accuracy)
    # Train final model on the entire dataset
    final_model = LinearRegression()
    final_model.fit(X_scaled, y)
    y_hat = final_model.predict(X_scaled)

    dump(final_model, "ForecastPollutionLevel.joblib")

    # Evaluate final model
    final_MAE, final_MSE, final_RMSE,final_Accuracy = evaluate_model(y, y_hat)
    print("Final Model Mean Absolute Error (MAE):", final_MAE)
    print("Final Model Mean Squared Error (MSE):", final_MSE)
    print("Final Model Root Mean Squared Error (RMSE):", final_RMSE)
    print("Final Model R² Score (Coefficient of Determination):", final_Accuracy)

    dump(final_Accuracy, "Accuracy.joblib")

    # Plot coefficients
    plot_coefficients(final_model, X.columns)

if __name__ == "__main__":
    main()







# test_residual = y_pred - y_test
# sns.scatterplot(x = y_test , y = test_residual)
# plt.axhline (y = 0, color = 'red',linestyle = '--')
# plt.show()





# def remove_outliers(df, column):
#
#     Q1 , Q3 = np.array(df[column].quantile([0.25,0.75]))
#     IRQ = Q3 -Q1
#     lower_bound = Q1 - 1.5 * IRQ
#     upper_bound = Q3 + 1.5 * IRQ
#     return df[ (df[column] >= lower_bound) &  (df[column] <= upper_bound) ]


# print(df.describe().to_string())
# plt.figure(figsize =(12,6))
# plt.subplot(1,2,1)
# sns.boxplot(data =df[['pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir']])
# plt.title("Before Outlier Removal")
# plt.xticks(rotation =45)


# for column in ['pm2.5','DEWP','TEMP','PRES','Iws']:#,,'Is','Ir'
#     df = remove_outliers(df,column)

# plt.subplot(1,2,2)
# sns.boxplot(df[['pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir']])
# plt.title("After Outliers Removal")
# plt.xticks(rotation = 45)
# plt.tight_layout()
# plt.show()




