import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# Load the trained model and scaler
model = joblib.load("wifi_model.joblib")
scaler = joblib.load("scaler_wifi.joblib")
# Function to predict room and display results
def predict():
    try:
        # Get input values from the user
        wifi1 = float(wifi1_var.get())
        wifi2 = float(wifi2_var.get())
        wifi3 = float(wifi3_var.get())
        wifi4 = float(wifi4_var.get())
        wifi5 = float(wifi5_var.get())
        wifi6 = float(wifi6_var.get())
        wifi7 = float(wifi7_var.get())
        # Create a sample input for prediction as a DataFrame
        sample = pd.DataFrame([[wifi1, wifi2, wifi3, wifi4, wifi5, wifi6, wifi7]],
                              columns=['Wifi 1', 'Wifi 2', 'Wifi 3', 'Wifi 4', 'Wifi 5', 'Wifi 6', 'Wifi 7'])
        # Scale the input
        sample_scaled = scaler.transform(sample)

        # Predict the room
        prediction = model.predict(sample_scaled)
        probabilities = model.predict_proba(sample_scaled)
        # Display the prediction
        result_frame.config(text=f"Predicted Room: {prediction[0]}")

        # Display the accuracy and error
        accuracy = np.max(probabilities)  # Highest probability
        error = 1 - accuracy  # Error calculation
        accuracy_frame.config(text=f"Accuracy: {accuracy:.2f}\nError: {error:.2f}")
    except Exception as e:
        messagebox.showerror("Input Error", str(e))
# Function to show probability percentage chart
def show_probability_chart():
    try:
        # Get input values from the user
        wifi1 = float(wifi1_var.get())
        wifi2 = float(wifi2_var.get())
        wifi3 = float(wifi3_var.get())
        wifi4 = float(wifi4_var.get())
        wifi5 = float(wifi5_var.get())
        wifi6 = float(wifi6_var.get())
        wifi7 = float(wifi7_var.get())
        # Create a sample input for prediction as a DataFrame
        sample = pd.DataFrame([[wifi1, wifi2, wifi3, wifi4, wifi5, wifi6, wifi7]],
                              columns=['Wifi 1', 'Wifi 2', 'Wifi 3', 'Wifi 4', 'Wifi 5', 'Wifi 6', 'Wifi 7'])

        # Scale the input
        sample_scaled = scaler.transform(sample)

        # Predict probabilities
        probabilities = model.predict_proba(sample_scaled)[0]
        # Plot the probabilities
        plt.figure(figsize=(8, 4))
        sns.barplot(x=model.classes_, y=probabilities)
        plt.title('Probability Percentages for Predicted Room')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.show()
    except Exception as e:
        messagebox.showerror("Input Error", str(e))
# Main window setup
root = tk.Tk()
root.title("WiFi Room Prediction")
root.geometry('400x400')
# Input fields for WiFi signals
wifi1_var = tk.StringVar()
wifi2_var = tk.StringVar()
wifi3_var = tk.StringVar()
wifi4_var = tk.StringVar()
wifi5_var = tk.StringVar()
wifi6_var = tk.StringVar()
wifi7_var = tk.StringVar()
# Create input labels and entries using grid
tk.Label(root, text="Wifi 1:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
tk.Entry(root, textvariable=wifi1_var).grid(row=0, column=1, padx=10, pady=5)
tk.Label(root, text="(Range: 0 to 100)").grid(row=0, column=2, padx=10, pady=5, sticky='w')
tk.Label(root, text="Wifi 2:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
tk.Entry(root, textvariable=wifi2_var).grid(row=1, column=1, padx=10, pady=5)
tk.Label(root, text="(Range: 0 to 100)").grid(row=1, column=2, padx=10, pady=5, sticky='w')
tk.Label(root, text="Wifi 3:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
tk.Entry(root, textvariable=wifi3_var).grid(row=2, column=1, padx=10, pady=5)
tk.Label(root, text="(Range: 0 to 100)").grid(row=2, column=2, padx=10, pady=5, sticky='w')
tk.Label(root, text="Wifi 4:").grid(row=3, column=0, padx=10, pady=5, sticky='w')
tk.Entry(root, textvariable=wifi4_var).grid(row=3, column=1, padx=10, pady=5)
tk.Label(root, text="(Range: 0 to 100)").grid(row=3, column=2, padx=10, pady=5, sticky='w')
tk.Label(root, text="Wifi 5:").grid(row=4, column=0, padx=10, pady=5, sticky='w')
tk.Entry(root, textvariable=wifi5_var).grid(row=4, column=1, padx=10, pady=5)
tk.Label(root, text="(Range: 0 to 100)").grid(row=4, column=2, padx=10, pady=5, sticky='w')
tk.Label(root, text="Wifi 6:").grid(row=5, column=0, padx=10, pady=5, sticky='w')
tk.Entry(root, textvariable=wifi6_var).grid(row=5, column=1, padx=10, pady=5)
tk.Label(root, text="(Range: 0 to 100)").grid(row=5, column=2, padx=10, pady=5, sticky='w')
tk.Label(root, text="Wifi 7:").grid(row=6, column=0, padx=10, pady=5, sticky='w')
tk.Entry(root, textvariable=wifi7_var).grid(row=6, column=1, padx=10, pady=5)
tk.Label(root, text="(Range: 0 to 100)").grid(row=6, column=2, padx=10, pady=5, sticky='w')


# Create the Predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=7, column=0, columnspan=3, pady=10)
# Create the Show Probability Percentage button
show_prob_button = tk.Button(root, text="Show Probability Percentage", command=show_probability_chart)
show_prob_button.grid(row=8, column=0, columnspan=3, pady=10)
# Frame to display results
result_frame = tk.Label(root, text="", font=('Arial', 14))
result_frame.grid(row=9, column=0, columnspan=3, pady=10)
# Frame to display accuracy and error
accuracy_frame = tk.Label(root, text="", font=('Arial', 12))
accuracy_frame.grid(row=10, column=0, columnspan=3, pady=10)

# Run the application
root.mainloop()





