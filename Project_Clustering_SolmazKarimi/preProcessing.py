import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preProcessing(df: pd.DataFrame, uselessCols: list, dataset_name: str, outlier_threshold:float, scaler='zscore', main_form=None):
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

    def scalingData(df: pd.DataFrame, scalerMethod: str):
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
            main_form.add_status("-No scaling applied")
            return df, numeric_cols

        # Initialize the appropriate scaler based on the method
        elif scalerMethod == 'zscore':
            scalerModel = StandardScaler()
        elif scalerMethod == 'minmax':
            scalerModel = MinMaxScaler()

        # Identify numeric columns and scale them
        numeric_cols = df.select_dtypes(include='number').columns
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scalerModel.fit_transform(df[numeric_cols])
        main_form.add_status(f"-Applied {scalerMethod} scaling to numeric columns")
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
    if dataset_name == 'User_Knowledge':
        scaled, numeric_cols = scalingData(df, scaler)

    elif dataset_name == 'Liver_Disorders':
        scaled, numeric_cols = scalingData(df, scaler)

    elif dataset_name == 'Dow_Jones':
        # Process specific columns for the Dow_Jones dataset
        for col in ['close', 'open', 'high', 'low']:
            if col in df.columns:
                df[col] = df[col].replace('[\\$,]', '', regex=True).str.strip().astype(float)
            else:
                print(f"Column {col} is empty or not found in the DataFrame.")

        # Create dummy variables for the 'stock' column
        stock_dummies = pd.get_dummies(df['stock'], prefix='stock')
        main_form.add_status('-get_dummies for stock column')
        df = pd.concat([df, stock_dummies], axis=1)
        df.drop(columns=['stock', 'date'], inplace=True)
        main_form.add_status(f"-Dropped columns: {['stock', 'date']}")
        scaled, numeric_cols = scalingData(df, scaler)

    elif dataset_name == 'Wholesale_Customers':
        scaled, numeric_cols = scalingData(df, scaler)

    elif dataset_name == 'Travel_Reviews':
        scaled, numeric_cols = scalingData(df, scaler)

    elif dataset_name == 'Electric_Consumption':
        scaled, numeric_cols = scalingData(df, scaler)

    # Remove outliers from the scaled DataFrame
    clean_df = remove_outliers(scaled, numeric_cols=numeric_cols)

    return clean_df, numeric_cols