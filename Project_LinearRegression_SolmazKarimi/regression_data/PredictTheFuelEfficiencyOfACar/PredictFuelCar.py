import pandas as pd
import numpy as np
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error
from joblib import load
from PIL import ImageTk, Image




# Load and preprocess the dataset
df = pd.read_csv("auto-mpg.data", sep=r'\s+', header=None)
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df.dropna(inplace=True)
df['car_name'] = df['car_name'].astype('category').cat.codes
# Define features and target variable
X = df.drop('mpg', axis=1)
y = df['mpg']
# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Load the pre-trained model
loaded_model = load("FuelCar.joblib")
# Get min/max values for scaling reference
feature_ranges = pd.DataFrame({
    'Feature': X.columns,
    'Min': scaler.data_min_,
    'Max': scaler.data_max_

})


def predict_mpg():
    try:
        input_data = pd.DataFrame({
            'cylinders': [float(cylinders_entry.get())],
            'displacement': [float(displacement_entry.get())],
            'horsepower': [float(horsepower_entry.get())],
            'weight': [float(weight_entry.get())],
            'acceleration': [float(acceleration_entry.get())],
            'model_year': [int(model_year_entry.get())],
            'origin': [int(origin_var.get())],
            'car_name': [int(car_name_entry.get())]
        }, columns=X.columns)
        # Scale using the pre-fitted scaler
        input_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = loaded_model.predict(input_scaled)
        result_label.config(text=f"Predicted MPG: {prediction[0]:.1f}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers in all fields")
def show_coefficients():
    # Get coefficients from the loaded model
    coefficients = loaded_model.coef_
    features = X.columns
    # Create a bar chart of the coefficients
    plt.figure(figsize=(10, 6))
    plt.barh(features, coefficients, color='skyblue')
    plt.xlabel('Coefficient Value')
    plt.title('Coefficients of the Linear Regression Model')
    plt.axvline(0, color='red', linestyle='--')  # Add a vertical line at x=0
    plt.show()

# Main window
root = tk.Tk()
root.title("Car MPG Predictor")
root.geometry("630x650")

# # Main container frame
# main_frame = ttk.Frame(root)
# main_frame.pack(fill='both', expand=True)#, padx=20, pady=20
# # Left frame for inputs
# input_frame = ttk.Frame(main_frame)
# input_frame.pack(side='left', padx=20, pady=20, fill='both')
# # Right frame for car image
# image_frame = ttk.Frame(main_frame)
# image_frame.pack(side='right', padx=20, pady=20)

# Feature scales
ttk.Label(root, text="Feature Scales (Min/Max)", font=('Arial', 10, 'bold')).pack(pady=5)
for _, row in feature_ranges.iterrows():
    ttk.Label(root,
             text=f"{row['Feature']}: {row['Min']:.1f} to {row['Max']:.1f}",
             font=('Arial', 8)).pack()
# Input fields with defaults
ttk.Label(root, text="\nEnter Car Details:", font=('Arial', 10, 'bold')).pack(pady=10)
defaults = {'cylinders': 4, 'displacement': 200, 'horsepower': 100,
           'weight': 3000, 'acceleration': 12, 'model_year': 76}
fields = []
for i, feature in enumerate(X.columns):
    frame = ttk.Frame(root)
    frame.pack(fill='x', padx=20, pady=5)

    ttk.Label(frame, text=f"{feature.capitalize()}:", width=15, anchor='w').pack(side='left')

    if feature == 'origin':
        origin_var = tk.StringVar(value="1")
        ttk.OptionMenu(frame, origin_var, "1", "1", "2", "3").pack(side='left')
    else:
        entry = ttk.Entry(frame, width=20)
        entry.pack(side='left', fill='x', expand=True)#
        entry.insert(0, str(defaults.get(feature, 0)))
        fields.append(entry)
cylinders_entry, displacement_entry, horsepower_entry, \
    weight_entry, acceleration_entry, model_year_entry, car_name_entry = fields
# Car name encoding info
ttk.Label(root, text="Car Name Encoding:", font=('Arial', 8)).pack()
ttk.Label(root, text="0=American, 1=European, 2=Japanese", font=('Arial', 8)).pack()


ttk.Button(root, text="Predict MPG", command=predict_mpg, width=40).pack(pady=10)
result_label = ttk.Label(root, text="", font=('Arial', 10, 'bold'))
result_label.pack()
# Show Coefficients button
ttk.Button(root, text="Show Coefficients", command=show_coefficients, width=40).pack(pady=10)

# Calculate RMSE for display
y_pred = loaded_model.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y, y_pred))
# RMSE Label
rmse_label = ttk.Label(root, text=f"RMSE: {rmse:.2f}", font=('Arial', 10, 'bold'), foreground='red')
rmse_label.pack(pady=10)

root.mainloop()

