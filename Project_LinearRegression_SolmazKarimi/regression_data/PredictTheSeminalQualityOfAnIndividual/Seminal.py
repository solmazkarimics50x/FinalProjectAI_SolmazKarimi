import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import load





class FertilityPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predict the Seminal Quality of an Individual")
        self.root.geometry("750x650")
        # Load and prepare the dataset
        self.load_data()
        self.load_model()
        self.create_widgets()
        self.style_ui()

    def load_data(self):
        # Load the dataset
        df = pd.read_csv("fertility_Diagnosis.txt", header=None)
        df.columns = ['Season', 'Age', 'Childish Diseases', 'Accident or serious trauma',
                      'Surgical intervention', 'High fevers in last year',
                      'Frequency of alcohol consumption', 'Smoking Habit',
                      'Number of hours spent sitting per day', 'Output']
        # Create dummy variables for the 'Output' column
        df = pd.get_dummies(data=df, columns=['Output'], drop_first=True)
        # Separate features and target variable
        self.X = df.drop(['Output_O'], axis=1)
        self.y = df['Output_O']
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, shuffle=True, random_state=42, test_size=0.3
        )
    def load_model(self):
        # Load the pre-trained model
        self.model = load("PredictSeminal.joblib")
        # Calculate RMSE for the model using the test set
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.rmse = np.sqrt(mse)

    def style_ui(self):
        self.root.configure(bg='#f0f8ff')
        style = ttk.Style()
        style.configure('TFrame', background='#f0f8ff')
        style.configure('TLabel', background='#f0f8ff', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
    def create_widgets(self):
        # Header frame
        header_frame = ttk.Frame(self.root)
        header_frame.pack(pady=20)
        ttk.Label(
            header_frame,
            text="Seminal Quality Prediction",
            style='Header.TLabel'
        ).pack()
        ttk.Label(
            header_frame,
            text="Enter patient information to predict seminal quality"
        ).pack(pady=10)
        # Input frame
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=20, padx=30, fill=tk.BOTH, expand=True)
        # Feature inputs with example default values
        features = [
            ("Season (normalized -1 to 1)", "-0.33"),
            ("Age (normalized 0-1)", "0.64"),
            ("Childish Diseases (0: no, 1: yes)", "1"),
            ("Accident or serious trauma (0: no, 1: yes)", "0"),
            ("Surgical intervention (0: no, 1: yes)", "0"),
            ("High fevers in last year (0: no, 1: yes)", "1"),
            ("Frequency of alcohol consumption (normalized 0-1)", "0.5"),
            ("Smoking Habit (-1: no, 0: sometimes, 1: yes)", "0"),
            ("Number of hours spent sitting per day (normalized 0-1)", "0.5")
        ]
        self.entries = {}
        for i, (feature, default_value) in enumerate(features):
            frame = ttk.Frame(input_frame)
            frame.grid(row=i, column=0, sticky="ew", pady=5)
            ttk.Label(frame, text=feature.split("(")[0], width=40, anchor="e").pack(side=tk.LEFT, padx=5)
            entry = ttk.Entry(frame, width=15)
            entry.pack(side=tk.LEFT, padx=5)
            entry.insert(0, default_value)  # Set default value
            self.entries[feature.split()[0]] = entry
        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        ttk.Button(
            button_frame,
            text="Predict Seminal Quality",
            command=self.predict,
            width=25
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            button_frame,
            text="Clear Fields",
            command=self.clear_fields,
            width=15
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            button_frame,
            text="Show Coefficients",
            command=self.show_coefficients,
            width=25
        ).pack(side=tk.LEFT, padx=10)
        # Results frame
        results_frame = ttk.Frame(self.root)
        results_frame.pack(pady=20)
        ttk.Label(
            results_frame,
            text="Prediction Results",
            style='Header.TLabel'
        ).pack(pady=10)
        self.results_text = tk.Text(
            results_frame,
            height=4,
            width=80,
            wrap=tk.WORD,
            font=('Helvetica', 10)
        )
        self.results_text.pack()

    def predict(self):
        try:
            feature_values = []
            for feature, entry in self.entries.items():
                value = float(entry.get())
                # Validate some features
                if feature == "Smoking Habit" and not (-1 <= value <= 1):
                    raise ValueError("Smoking Habit must be -1, 0 or 1")
                if feature in ["Childish Diseases", "Accident or serious trauma", "Surgical intervention",
                               "High fevers in last year"] and value not in [0, 1]:
                    raise ValueError(f"{feature} must be 0 or 1")
                feature_values.append(value)
            # Create a DataFrame for the input data
            input_data = pd.DataFrame([feature_values], columns=self.X.columns)
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            # Interpret result
            probability = f"{round(prediction * 100, 1)}%"
            # Display results with conditional formatting
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Predicted Probability of Abnormal Seminal Quality: {probability}\n\n")
            if prediction > 0.3:
                self.results_text.insert(tk.END, "Interpretation: High probability of abnormal seminal quality", 'warning')
            else:
                self.results_text.insert(tk.END, "Interpretation: Likely normal seminal quality", 'normal')
            # Display RMSE
            self.results_text.insert(tk.END, f"Model RMSE: {round(self.rmse, 2)}\n")
            self.results_text.tag_config('warning', foreground='red')
            self.results_text.tag_config('normal', foreground='green')
            self.results_text.config(state=tk.DISABLED)
        except ValueError as e:(
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}"))

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

    def show_coefficients(self):
        # Get the coefficients from the model
        coefficients = self.model.coef_
        feature_names = self.X.columns
        # Create a bar chart for the coefficients
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, coefficients, color='skyblue')
        plt.xlabel('Coefficient Value')
        plt.title('Linear Regression Coefficients')
        plt.axvline(0, color='red', linestyle='--')  # Add a vertical line at x=0
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = FertilityPredictorApp(root)
    root.mainloop()
        