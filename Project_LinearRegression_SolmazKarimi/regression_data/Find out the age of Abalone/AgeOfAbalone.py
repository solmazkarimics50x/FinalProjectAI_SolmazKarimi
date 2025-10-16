from tkinter import *
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
from joblib import  load
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class AbalonePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Abalone Age Predictor")
        self.root.geometry("800x600")
        # Load the model
        self.load_model()
        # # Load and prepare the data
        # self.load_data()
        # self.train_model()
        self.setup_ui()

    def load_model(self):
        try:
            # Load the model
            self.model = load('abalone_model.joblib')

            # Load training data and calculate RMSE
            try:
                df = pd.read_csv("abalone.data.csv", header=None)
                df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                              'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
                df_encoded = pd.get_dummies(df, columns=['Sex'], drop_first=True)
                new_order = ['Sex_M', 'Sex_I', 'Length', 'Diameter', 'Height',
                             'Whole weight', 'Shucked weight', 'Viscera weight',
                             'Shell weight', 'Rings']
                df_reordered = df_encoded[new_order]

                X_train = df_reordered.drop('Rings', axis=1)
                y_train = df_reordered['Rings']

                # Calculate RMSE using the loaded model
                y_pred = self.model.predict(X_train)
                self.rmse = np.sqrt(mean_squared_error(y_train, y_pred))
                #print(f"Model loaded. Training RMSE: {self.rmse:.2f}")

            except Exception as e:
                print(f"Could not calculate RMSE: {str(e)}")
                self.rmse = "N/A"  # Set default if RMSE calculation fails

        except FileNotFoundError:
            messagebox.showerror("Error", "Model file not found. Please train the model first.")
            raise




    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        # Title
        title_label = ttk.Label(main_frame, text="Abalone Age Predictor", font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        subtitle_label = ttk.Label(main_frame, text="Predict the age of abalone based on physical measurements", font=('Helvetica', 10))
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        # Input Parameters Frame
        input_frame = LabelFrame(main_frame, text="Input Parameters", padx=10,font=('Arial',12,'bold'),bg='#73C2FB')##, padding=15
        input_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        # Sex selection
        self.sex_var = tk.StringVar(value="male")
        Label(input_frame, text="Sex:",bg='#73C2FB').grid(row=0, column=0, sticky="w", pady=5)
        Radiobutton(input_frame, text="Male", variable=self.sex_var, value="male",bg='#73C2FB').grid(row=0, column=1, sticky="w")
        Radiobutton(input_frame, text="Female", variable=self.sex_var, value="female",bg='#73C2FB').grid(row=0, column=2, sticky="w")
        Radiobutton(input_frame, text="Infant", variable=self.sex_var, value="infant",bg='#73C2FB').grid(row=0, column=3, sticky="w")
        # Feature sliders
        self.features = [
            ("Length (0.1-0.8)", "length", 0.1, 0.8, 0.455),
            ("Diameter (0.1-0.8)", "diameter", 0.1, 0.8, 0.365),
            ("Height (0.01-0.2)", "height", 0.01, 0.2, 0.095),
            ("Whole Weight (0.1-2.5)", "whole_weight", 0.1, 2.5, 0.514),
            ("Shucked Weight (0.01-1.0)", "shucked_weight", 0.01, 1.0, 0.2245),
            ("Viscera Weight (0.01-0.5)", "viscera_weight", 0.01, 0.5, 0.101),
            ("Shell Weight (0.01-0.5)", "shell_weight", 0.01, 0.5, 0.15)
        ]
        self.sliders = {}
        self.value_labels = {}
        for i, (label_text, name, min_val, max_val, default_val) in enumerate(self.features):
            Label(input_frame, text=label_text,bg='#73C2FB').grid(row=i + 1, column=0, sticky="w", pady=5)

            # Value display
            self.value_labels[name] = Label(input_frame, text=f"{default_val:.3f}",bg='#73C2FB')
            self.value_labels[name].grid(row=i + 1, column=1, sticky="e", padx=5)
            # Slider
            self.sliders[name] = ttk.Scale(
                input_frame,
                from_=min_val,
                to=max_val,
                value=default_val,
                command=lambda val, n=name: self.update_slider_value(n, val),
                length=300
            )
            self.sliders[name].grid(row=i + 1, column=2, columnspan=3, sticky="ew")
        # Predict button
        predict_btn = ttk.Button(input_frame, text="Predict Age", command=self.predict, width= 30,padding= 15)
        predict_btn.grid(row=len(self.features) + 2, column=0, columnspan=4, pady=40)

        # Show Coefficients button
        coeff_btn = ttk.Button(input_frame, text="Show Coefficients", command=self.show_coefficients, width=30)
        coeff_btn.grid(row=len(self.features) + 3, column=0, columnspan=4, pady=10)

        # Results Frame
        results_frame = LabelFrame(main_frame, text="Prediction Results",padx=10,font=('Arial',12,'bold'),bg='#98CEEE' )#,padding=15
        results_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        self.result_label = Label(results_frame, text="Predicted Age (Rings): ", font=('Helvetica', 12),bg='#98CEEE')
        self.result_label.pack(pady=5)

        self.age_display = Label(results_frame, text="--", font=('Helvetica', 24, 'bold'), foreground="blue",bg='#98CEEE')
        self.age_display.pack(pady=20)

        self.rmse_display = Label(results_frame, text="",bg='#98CEEE', font=('Helvetica', 12))
        self.rmse_display.pack(pady=5)
        # Configuration for grid layout
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(2, weight=1)

    def update_slider_value(self, name, value):
        # Get the slider object
        slider = self.sliders[name]
        # Get the current value and update the label
        current_value = float(slider.get())
        self.value_labels[name].config(text=f"{current_value:.3f}")

    def predict(self):
        try:
            # Get sex values
            sex_m = 1 if self.sex_var.get() == "male" else 0
            sex_i = 1 if self.sex_var.get() == "infant" else 0
            # Get other feature values
            features = [
                sex_m,
                sex_i,
                float(self.sliders["length"].get()),
                float(self.sliders["diameter"].get()),
                float(self.sliders["height"].get()),
                float(self.sliders["whole_weight"].get()),
                float(self.sliders["shucked_weight"].get()),
                float(self.sliders["viscera_weight"].get()),
                float(self.sliders["shell_weight"].get())
            ]

            # Convert features to a NumPy array for prediction
            features_array = np.array(features).reshape(1, -1)  # Reshape for a single sample
            # Make prediction
            prediction = self.model.predict(features_array)

            # # Create a DataFrame with the same feature names used during training
            # feature_names = ['Sex_M', 'Sex_I', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
            #                  'Viscera weight', 'Shell weight']
            # features_df = pd.DataFrame([features], columns=feature_names)
            # # Make prediction
            # prediction = self.model.predict(features_df)

            # Display result
            self.age_display.config(text=f"{prediction[0]:.1f}")

            #Display RMSE
            self.rmse_display.config(text=f"RMSE: {self.rmse:.2f}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    def show_coefficients(self):
        # Get coefficients from the model
        coefficients = self.model.coef_
        feature_names = ['Sex_M', 'Sex_I', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                         'Viscera weight', 'Shell weight']
        # Create a bar chart of the coefficients
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, coefficients, color='skyblue')
        plt.xlabel('Coefficient Value')
        plt.title('Coefficients of the Linear Regression Model')
        plt.axvline(0, color='red', linestyle='--')  # Add a vertical line at x=0
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = AbalonePredictorApp(root)
    root.mainloop()