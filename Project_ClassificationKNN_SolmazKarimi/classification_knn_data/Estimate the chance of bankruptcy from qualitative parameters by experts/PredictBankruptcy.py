import tkinter as tk
from tkinter import messagebox , ttk
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk

# Load the trained model
model = joblib.load('PredictBankruptcy.joblib')
# Function to predict and display results
def predict():
    try:
        # Get input values from the user
        industrial_risk = int(industrial_risk_var.get())
        management_risk = int(management_risk_var.get())
        financial_flexibility = int(financial_flexibility_var.get())
        credibility = int(credibility_var.get())
        competitiveness = int(competitiveness_var.get())
        operating_risk = int(operating_risk_var.get())
        # Create a sample input for prediction as a DataFrame
        sample = pd.DataFrame({
            'Industrial Risk': [industrial_risk],
            'Management Risk': [management_risk],
            'Financial Flexibility': [financial_flexibility],
            'Credibility': [credibility],
            'Competitiveness': [competitiveness],
            'Operating Risk': [operating_risk]
        })
        # Predict the class and probabilities
        prediction = model.predict(sample)
        probabilities = model.predict_proba(sample)
        # Display the prediction
        result = "Bankruptcy" if prediction[0] == 1 else "Not Bankruptcy"

        # Display the error
        error = (1 - probabilities[0][prediction[0]])
        accuracy = 1 - error  # Accuracy percentage
        result_frame.config(text=f"Prediction: {result}\nPrediction Error: {error:.2f} "
                                 f"\nAccouracy: {accuracy:.2f}")


    except Exception as e:
        messagebox.showerror("Input Error", str(e))


# Function to show probability percentage chart
def show_probability_chart():
    try:
        # Get input values from the user
        industrial_risk = int(industrial_risk_var.get())
        management_risk = int(management_risk_var.get())
        financial_flexibility = int(financial_flexibility_var.get())
        credibility = int(credibility_var.get())
        competitiveness = int(competitiveness_var.get())
        operating_risk = int(operating_risk_var.get())

        # Create a sample input for prediction as a DataFrame
        sample = pd.DataFrame({
            'Industrial Risk': [industrial_risk],
            'Management Risk': [management_risk],
            'Financial Flexibility': [financial_flexibility],
            'Credibility': [credibility],
            'Competitiveness': [competitiveness],
            'Operating Risk': [operating_risk]
        })

        # Predict probabilities
        probabilities = model.predict_proba(sample)
        # Plot the probabilities
        plt.figure(figsize=(8, 4))
        sns.barplot(x=['Not Bankruptcy', 'Bankruptcy'], y=probabilities[0])
        plt.title('Probability Percentages')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.show()

    except Exception as e:
        (messagebox.showerror("Input Error", str(e)))

# Function to clear all input fields
def clear_entries():
    industrial_risk_var.set("")
    management_risk_var.set("")
    financial_flexibility_var.set("")
    credibility_var.set("")
    competitiveness_var.set("")
    operating_risk_var.set("")
    result_frame.config(text="")  # Clear the result display

# Main window setup
root = tk.Tk()
root.title("Bankruptcy Prediction")
root.geometry('600x550')
root.resizable(0,0)
x = int(root.winfo_screenwidth() / 2 - 600 / 2)
y = int(root.winfo_screenheight() / 2 - 550 / 2)
root.geometry(f'+{x}+{y}')
# Main container frame
main_frame = ttk.Frame(root, padding=20)
main_frame.grid(row=0, column=0, sticky='nsew')
# Configure grid weights
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
main_frame.grid_rowconfigure(7, weight=1)  # Extra row for spacing
main_frame.grid_columnconfigure(2, weight=1)  # Extra column for spacing
# Header with image
title_label = ttk.Label(main_frame, text="Bankruptcy Predictor", font=('Helvetica', 16, 'bold'))
title_label.grid(row=0, column=0, columnspan=3, pady=10)

img = Image.open("image.jpg").resize((450, 150), Image.LANCZOS)
img = ImageTk.PhotoImage(img)
img_label = ttk.Label(main_frame, image=img)
img_label.grid(row=1, column=0, columnspan=2, padx=10, sticky='n')
# Input Parameters Frame
input_frame = tk.Label(main_frame, padx=10, font=('Arial', 12, 'bold'), bg='#73C2FB')
input_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")
result_frame = tk.Label(main_frame, padx=10, font=('Arial', 14, 'bold'))
result_frame.grid(row=2, column=1, padx=10, pady=10, sticky="w")
# Create input fields
tk.Label(input_frame, bg='#73C2FB', text="Industrial Risk (0, 1, 2):").grid(row=3, column=0)
industrial_risk_var = tk.StringVar()
tk.Entry(input_frame, textvariable=industrial_risk_var).grid(row=3, column=1)
tk.Label(input_frame,bg='#73C2FB', text="Management Risk (0, 1, 2):").grid(row=4, column=0)
management_risk_var = tk.StringVar()
tk.Entry(input_frame, textvariable=management_risk_var).grid(row=4, column=1)
tk.Label(input_frame,bg='#73C2FB', text="Financial Flexibility (0, 1, 2):").grid(row=5, column=0)
financial_flexibility_var = tk.StringVar()
tk.Entry(input_frame, textvariable=financial_flexibility_var).grid(row=5, column=1)
tk.Label(input_frame,bg='#73C2FB', text="Credibility (0, 1, 2):").grid(row=6, column=0)
credibility_var = tk.StringVar()
tk.Entry(input_frame, textvariable=credibility_var).grid(row=6, column=1)
tk.Label(input_frame,bg='#73C2FB', text="Competitiveness (0, 1, 2):").grid(row=7, column=0)
competitiveness_var = tk.StringVar()
tk.Entry(input_frame, textvariable=competitiveness_var).grid(row=7, column=1)
tk.Label(input_frame,bg='#73C2FB', text="Operating Risk (0, 1, 2):").grid(row=8, column=0)
operating_risk_var = tk.StringVar()
tk.Entry(input_frame, textvariable=operating_risk_var).grid(row=8, column=1)
# Create the Predict button
predict_button = tk.Button(main_frame, text="Predict", command=predict, width=20)
predict_button.grid(row=4, columnspan=2, pady= 20)

# Create the Show Probability Percentage button
show_prob_button = tk.Button(main_frame, text="Show Probability Percentage", command=show_probability_chart, width=30)
show_prob_button.grid(row=5, columnspan=2, pady=10)

# Create the Clear button
clear_button = tk.Button(main_frame, text="Clear", command=clear_entries, width=20)
clear_button.grid(row=6, columnspan=2, pady=10)


# Run the application
root.mainloop()
