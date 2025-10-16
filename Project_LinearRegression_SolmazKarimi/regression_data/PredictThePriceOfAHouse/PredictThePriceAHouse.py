import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tkinter import *
import tkinter as tk
from tkinter import messagebox, ttk
from joblib import load
import matplotlib.pyplot as plt



# Load the dataset
df = pd.read_excel("Real estate valuation data set.xlsx", usecols=['X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores', 'X5 latitude', 'X6 longitude', 'Y house price of unit area'])
# Separate features and target variable
X = df[['X2 house age', 'X3 distance to the nearest MRT station',
         'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
y = df['Y house price of unit area']
# Scale the features
mms = MinMaxScaler()
X_scaled = mms.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=True, random_state=42, test_size=0.3)

# Load the pre-trained model
loaded_model = load("PredictPriceHouse.joblib")


# Function to predict house price
def predict_price():
    try:
        # Get input values
        house_age = float(entry_age.get())
        distance_to_mrt = float(entry_distance.get())
        num_convenience_stores = int(entry_stores.get())
        latitude = float(entry_latitude.get())
        longitude = float(entry_longitude.get())

        # Prepare input for prediction as a DataFrame
        input_features = pd.DataFrame([[house_age, distance_to_mrt, num_convenience_stores, latitude, longitude]],
                                      columns=['X2 house age',
                                               'X3 distance to the nearest MRT station',
                                               'X4 number of convenience stores',
                                               'X5 latitude',
                                               'X6 longitude'])
        # Scale the input features
        input_scaled = mms.transform(input_features)
        # Make prediction
        predicted_price = loaded_model.predict(input_scaled)
        # Show the result
        # Display result
        price_display.config(text=f"{predicted_price[0]:.4f} NTD per Ping")

        # Calculate MSE and R²
        y_pred = loaded_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Update error display
        error_display.config(text=f"RMSE: {rmse:.4f}, R²: {r2:.4f}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

def show_coefficients():
    # Get the coefficients from the model
    coefficients = loaded_model.coef_
    feature_names = ['X2 house age', 'X3 distance to the nearest MRT station',
                     'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
    # Create a bar chart for the coefficients
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coefficients, color='skyblue')
    plt.xlabel('Coefficient Value')
    plt.title('Linear Regression Coefficients')
    plt.axvline(0, color='red', linestyle='--')  # Add a vertical line at x=0
    plt.show()

def create_spinbox(parent, from_, to, default, increment, row):
    frame = Frame(parent,bg='#73C2FB')
    frame.grid(row=row, column=1, sticky="ew")

    # Entry field
    entry_var = tk.StringVar(value=str(default))
    entry = Entry(frame, textvariable=entry_var, width=20)
    entry.pack(side="left", fill="x", expand=True)

    # Buttons frame
    btn_frame = ttk.Frame(frame)
    btn_frame.pack(side="left")

    # Up button
    def increment_value():
        try:
            current = float(entry_var.get())
            new_val = min(to, current + increment)
            entry_var.set(f"{new_val:.6f}" if isinstance(increment, float) else str(int(new_val)))
        except ValueError:
            entry_var.set(str(default))

    btn_up = ttk.Button(btn_frame, text="▲", width=2, command=increment_value)
    btn_up.pack()

    # Down button
    def decrement_value():
        try:
            current = float(entry_var.get())
            new_val = max(from_, current - increment)
            entry_var.set(f"{new_val:.6f}" if isinstance(increment, float) else str(int(new_val)))
        except ValueError:
            entry_var.set(str(default))

    btn_down = ttk.Button(btn_frame, text="▼", width=2, command=decrement_value)
    btn_down.pack()

    return entry_var



# Create the GUI
root = tk.Tk()
root.title("House Price Prediction")
root.geometry("800x600")



main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)
# Title
title_label = ttk.Label(main_frame, text="House Price Predictor", font=('Helvetica', 16, 'bold'))
title_label.grid(row=0, column=0, columnspan=3, pady=10)

subtitle_label = ttk.Label(main_frame, text="predict the price of a house based on the market historical data set of real estate valuations ",
                           font=('Helvetica', 10))
subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))

# Input Parameters Frame
input_frame = LabelFrame(main_frame, text="Input Parameters", padx=10, font=('Arial', 12, 'bold'),
                         bg='#73C2FB')  ##, padding=15
input_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

# Create input fields using spinboxes
Label(input_frame, text="House Age (years):",bg='#73C2FB').grid(row=0, column=0, sticky="e")
entry_age = create_spinbox(input_frame,1, 100, 19.5, 0.1, 0)
Label(input_frame, text="Distance to MRT Station (meters):",bg='#73C2FB').grid(row=1, column=0, sticky="e")
entry_distance = create_spinbox(input_frame, 1, 1000, 390.5684, 0.0001, 1)
Label(input_frame, text="Number of Convenience Stores:",bg='#73C2FB').grid(row=2, column=0, sticky="e")
entry_stores = create_spinbox(input_frame, 0, 10, 6, 1, 2)
Label(input_frame, text="Latitude (degrees):",bg='#73C2FB').grid(row=3, column=0, sticky="e")
entry_latitude = create_spinbox(input_frame, -90, 90, 24.97937, 0.00001, 3)
Label(input_frame, text="Longitude (degrees):",bg='#73C2FB').grid(row=4, column=0, sticky="e")
entry_longitude = create_spinbox(input_frame, -180, 180, 121.54243, 0.00001, 4)
# Predict buttoninput_frame
btn_predict = ttk.Button(input_frame, text="Predict House Price", command=predict_price, width= 30, padding= 12)
btn_predict.grid(row=5, columnspan=2, pady=10)

# Show Coefficients button
btn_show_coefficients = ttk.Button(input_frame, text="Show Coefficients", command=show_coefficients, width=30, padding=12)
btn_show_coefficients.grid(row=6, columnspan=2, pady=10)

# Results Frame
results_frame = LabelFrame(main_frame, text="Prediction Results", padx=10, font=('Arial', 12, 'bold'),
                           bg='#98CEEE')  # ,padding=15
results_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

result_label = Label(results_frame, text="Predicted House Price: ", font=('Helvetica', 12), bg='#98CEEE')
result_label.pack(pady=5)

price_display = Label(results_frame, text="--", font=('Helvetica', 24, 'bold'), foreground="blue", bg='#98CEEE')
price_display.pack(pady=20)

error_label = Label(results_frame, text="mean_squared_error: ", font=('Helvetica', 12), bg='#98CEEE')
error_label.pack(pady=5)

# Error Display
error_display = Label(results_frame, text="", font=('Helvetica', 12), foreground="red", bg='#98CEEE')
error_display.pack(pady=5)




# Configuration for grid layout
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)
input_frame.columnconfigure(2, weight=1)


# Run the GUI
root.mainloop()

