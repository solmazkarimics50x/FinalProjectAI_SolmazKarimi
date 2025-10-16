import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preProcessing_classificationKNN(df: pd.DataFrame, uselessCols: list, dataset_name: str, outlier_threshold:float, scaler='zscore', main_form=None, target_col=None):
    """
    Preprocesses the input DataFrame by scaling, removing outliers, and dropping useless columns.
    Parameters:
    - df: pd.DataFrame - The input DataFrame to be processed.
    - uselessCols: list - A list of column names to be dropped from the DataFrame.
    - dataset_name: str - The name of the dataset, which may affect processing.
    - outlier_threshold: float - The threshold for identifying outliers using the IQR method.
    - scaler: str - The scaling method to apply ('zscore', 'minmax', or 'None').
    - main_form: object - A reference to the main GUI class for updating status messages.
    Returns:
    - clean_df: pd.DataFrame - The cleaned and preprocessed DataFrame.
    - numeric_cols: list - The names of the numeric columns in the DataFrame.
    """

    def scalingData(df: pd.DataFrame, scalerMethod: str, target_col: str = None):
        """
        Scales the numeric columns of the DataFrame based on the specified scaling method.
        Parameters:
        - df: pd.DataFrame - The DataFrame to scale.
        - scalerMethod: str - The scaling method to apply.
        Returns:
        - df_scaled: pd.DataFrame - The scaled DataFrame.
        - numeric_cols: list - The names of the numeric columns.
        """
        # Check if no scaling is to be applied
        if scalerMethod == 'None':
            numeric_cols = df.select_dtypes(include='number').columns
            if target_col is not None and target_col in numeric_cols:
                numeric_cols = numeric_cols.drop(target_col)
            main_form.add_status("-No scaling applied")
            return df, numeric_cols

        # Initialize the appropriate scaler based on the method
        elif scalerMethod == 'zscore':
            scalerModel = StandardScaler()
        elif scalerMethod == 'minmax':
            scalerModel = MinMaxScaler()

        # Identify numeric columns and scale them
        numeric_cols = df.select_dtypes(include='number').columns
        if target_col is not None and target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scalerModel.fit_transform(df[numeric_cols])
        main_form.add_status(f"-Applied {scalerMethod} scaling to numeric columns (excluding target)")
        return df_scaled , numeric_cols

    def remove_outliers(df: pd.DataFrame, numeric_cols:list):
        """
        Removes outliers from the DataFrame based on the IQR method.
        Parameters:
        - df: pd.DataFrame - The DataFrame from which to remove outliers.
        - numeric_cols: list - The names of the numeric columns to check for outliers.
        Returns:
        - df_clean: pd.DataFrame - The DataFrame with outliers removed.
        """
        df_clean = df.copy()
        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)

        # Calculate IQR and identify outliers for each numeric column
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Get records that are outliers in any column
        records_to_remove = outlier_mask.any(axis=1)
        outlier_count = records_to_remove.sum()
        main_form.add_status(f"-Removed {outlier_count} outliers")
        df_clean = df_clean[~records_to_remove]

        return df_clean

    # Drop specified useless columns if any
    if uselessCols and uselessCols != ['']:
        df.drop(columns=uselessCols, inplace=True)
        main_form.add_status(f"-Dropped columns: {uselessCols}")

    # Drop rows with NaN values
    if df.isna().any().any():
        df.dropna(inplace=True)
        main_form.add_status(f"-Dropped NaN values")

    # Specific dataset processing based on the dataset name:
    if dataset_name == 'Chance_Bankruptcy' :


        #map_categorical_features
        # print(df['Industrial Risk'].unique())
        df['Industrial Risk'] = df['Industrial Risk'].map({'P': 2, 'N': 0, 'A': 1})
        # print(df['Management Risk'].unique())
        df['Management Risk'] = df['Management Risk'].map({'P': 2, 'N': 0, 'A': 1})
        # print(df['Financial Flexibility'].unique())
        df['Financial Flexibility'] = df['Financial Flexibility'].map({'A': 1, 'P': 2, 'N': 0})
        # print(df['Credibility'].unique())
        df['Credibility'] = df['Credibility'].map({'A': 1, 'P': 2, 'N': 0})
        # print(df['Competitiveness'].unique())
        df['Competitiveness'] = df['Competitiveness'].map({'A': 1, 'P': 2, 'N': 0})
        # print(df['Operating Risk'].unique())
        df['Operating Risk'] = df['Operating Risk'].map({'P': 2, 'N': 0, 'A': 1})
        # print(df['Class'].unique())
        df['Class'] = df['Class'].map({'NB': 0, 'B': 1})
        main_form.add_status(f"-map_categorical_features for Industrial Risk & Management Risk & Financial Flexibility & Credibility & Competitiveness & Operating Risk & Class columns")

        scaled, numeric_cols = scalingData(df, scaler, target_col='Class')


    elif dataset_name == 'WIFI_Signal_Strength':
        scaled, numeric_cols = scalingData(df, scaler, target_col=target_col)

    elif dataset_name == "Predict_Acceptability_Car" :
        # map_categorical_features
        # print(df['buying'].unique())
        df['buying'] = df['buying'].map({'vhigh': 4, 'high': 3, 'med': 2, 'low': 1})
        # print(df['maint'].unique())
        df['maint'] = df['maint'].map({'vhigh': 4, 'high': 3, 'med': 2, 'low': 1})
        # print(df['doors'].unique())
        df['doors'] = df['doors'].map({'2': 2, '3': 3, '4': 4, '5more': 5})
        # print(df['persons'].unique())
        df['persons'] = df['persons'].map({'2': 2, '4': 4, 'more': 5})
        # print(df['lug_boot'].unique())
        df['lug_boot'] = df['lug_boot'].map({'small': 0, 'med': 1, 'big': 2})
        # print(df['safety'].unique())
        df['safety'] = df['safety'].map({'low': 0, 'med': 1, 'high': 2})
        # print(df['class'].unique())
        df['class'] = df['class'].map({'unacc': 0, 'acc': 1, 'vgood': 2, 'good': 3})
        main_form.add_status(
            f"-map_categorical_features for buying & maint & doors & persons & lug_boot & safety & class columns")
        scaled, numeric_cols = scalingData(df, scaler, target_col='class')






    # elif dataset_name == 'Dow_Jones':
    #     # Process specific columns for the Dow_Jones dataset
    #     for col in ['close', 'open', 'high', 'low']:
    #         if col in df.columns:
    #             df[col] = df[col].replace('[\\$,]', '', regex=True).str.strip().astype(float)
    #         else:
    #             print(f"Column {col} is empty or not found in the DataFrame.")
    #
    #     # Create dummy variables for the 'stock' column
    #     stock_dummies = pd.get_dummies(df['stock'], prefix='stock')
    #     main_form.add_status('-get_dummies for stock column')
    #     df = pd.concat([df, stock_dummies], axis=1)
    #     df.drop(columns=['stock', 'date'], inplace=True)
    #     main_form.add_status(f"-Dropped columns: {['stock', 'date']}")
    #     scaled, numeric_cols = scalingData(df, scaler)
    #
    # elif dataset_name == 'Wholesale_Customers':
    #     scaled, numeric_cols = scalingData(df, scaler)
    #
    # elif dataset_name == 'Travel_Reviews':
    #     scaled, numeric_cols = scalingData(df, scaler)
    #
    # elif dataset_name == 'Electric_Consumption':
    #     scaled, numeric_cols = scalingData(df, scaler)

    # Remove outliers from the scaled DataFrame
    clean_df = remove_outliers(scaled, numeric_cols=numeric_cols)

    return clean_df, numeric_cols