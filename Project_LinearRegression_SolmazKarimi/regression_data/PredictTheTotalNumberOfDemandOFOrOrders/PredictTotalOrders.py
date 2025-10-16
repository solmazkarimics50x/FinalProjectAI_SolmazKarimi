import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from  tkinter import  *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk  # Import PIL for image handling



# Load the model
loaded_model = load('PredictTotalOrder.joblib')
# Load the coefficients for visualization
df = pd.read_csv("Daily_Demand_Forecasting_Orders.csv", sep=";")
X = df.drop(['Target (Total orders)'], axis=1)
mms = MinMaxScaler()
X_scaled = mms.fit_transform(X)
final_model = LinearRegression()
final_model.fit(X_scaled, df['Target (Total orders)'])





# Function to predict total orders
def predict_orders():
    try:
        week_of_month = float(entry_week_of_month.get())
        day_of_week = float(entry_day_of_week.get())
        non_urgent_order = float(entry_non_urgent_order.get())
        urgent_order = float(entry_urgent_order.get())
        order_type_a = float(entry_order_type_a.get())
        order_type_b = float(entry_order_type_b.get())
        order_type_c = float(entry_order_type_c.get())
        fiscal_sector_orders = float(entry_fiscal_sector_orders.get())
        traffic_controller_orders = float(entry_traffic_controller_orders.get())
        banking_orders_1 = float(entry_banking_orders_1.get())
        banking_orders_2 = float(entry_banking_orders_2.get())
        banking_orders_3 = float(entry_banking_orders_3.get())
        # Prepare input for prediction
        # Prepare input for prediction as a DataFrame
        input_data = pd.DataFrame([[week_of_month, day_of_week, non_urgent_order, urgent_order,
                                     order_type_a, order_type_b, order_type_c,
                                     fiscal_sector_orders, traffic_controller_orders,
                                     banking_orders_1, banking_orders_2, banking_orders_3]],
                                   columns=X.columns)  # Use the same columns as the original data
        # Scale the input data
        input_scaled = mms.transform(input_data)

        # Predict
        prediction = loaded_model.predict(input_scaled)

        # Calculate RMSE
        y_hat = final_model.predict(X_scaled)
        final_MSE = np.mean((y_hat - df['Target (Total orders)']) ** 2)
        final_RMSE = np.sqrt(final_MSE)
        # Show prediction and RMSE
        result_label.config(text=f"Predicted Total Orders: {prediction[0]:.2f}\n" +
                                 "\n RMSE: ",
                            fg="black")
        rmse_label = Label(result_frame, text=f"{final_RMSE:.2f}",
                              fg="red", font=('Arial', 12),bg='#73C2FB')
        rmse_label.pack()
        btn_show_coefficients.grid(row=14, column=0, columnspan=2, pady=5)
        # result_label.config(text=f"Predicted Total Orders: {prediction[0]:.2f}\nRMSE: {final_RMSE:.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Function to display coefficients
def show_coefficients():
    # Create a new window for the plot
    plot_window = tk.Toplevel(root)
    plot_window.title("Model Coefficients")
    plot_window.geometry("900x600")
    plot_window.resizable(True, True)

    # Calculate coefficients
    coef_data = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': final_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in coef_data['Coefficient']]
    bars = ax.barh(coef_data['Feature'], coef_data['Coefficient'], color=colors, height=0.6)

    # Style the plot
    ax.set_title("Feature Impact on Total Orders", pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    fig.tight_layout()

    # Embed in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Add close button
    ttk.Button(plot_window, text="Close", command=plot_window.destroy) \
        .pack(pady=10, padx=20, side=tk.BOTTOM)

# Create the main window
root = tk.Tk()
root.title("Total Orders Prediction")
root.geometry("700x600")
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)
# Title and Subtitle Frame
title_frame = ttk.Frame(main_frame)
title_frame.grid(row=0, column=0, columnspan=3, pady=10)
# Title
title_label = ttk.Label(main_frame, text="Total Number Orders Predictor", font=('Helvetica', 16, 'bold'))
title_label.grid(row=0, column=0, columnspan=3, padx= 200, pady= 20)

# Load and display TotalOrdersimage
total_order_image = Image.open("image.jpg")  # Update with your image path
total_order_image = total_order_image.resize((120, 100), Image.LANCZOS)  # Resize image if necessary
total_order_photo = ImageTk.PhotoImage(total_order_image)
total_order_label = ttk.Label(title_frame, image=total_order_photo)
total_order_label.grid(row=0, column=0, columnspan=1,rowspan= 2, padx=(5, 0))
# Subtitle
subtitle_label = ttk.Label(title_frame, text="Predict total demand and analyze model behavior", font=('Helvetica', 14 ))
subtitle_label.grid(row=1, column=1, columnspan=3, padx=10, pady=(20, 0))

# Input Parameters Frame
# Input Parameters Frame
input_frame = LabelFrame(main_frame, text="Input Parameters", padx=10, font=('Arial', 12, 'bold'), bg='#73C2FB')
input_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")



# Create input fields
Label(input_frame, text="Week of the Month (1-5):",bg='#73C2FB').grid(row=2, column=0)
entry_week_of_month = tk.Entry(input_frame)
entry_week_of_month.grid(row=2, column=1)
entry_week_of_month.insert(0, "1")  # Default value

Label(input_frame, text="Day of the Week (2-6):",bg='#73C2FB').grid(row=3, column=0)
entry_day_of_week = tk.Entry(input_frame)
entry_day_of_week.grid(row=3, column=1)
entry_day_of_week.insert(0, "2")  # Default value

Label(input_frame, text="Non-Urgent Order:",bg='#73C2FB').grid(row=4, column=0)
entry_non_urgent_order = tk.Entry(input_frame)
entry_non_urgent_order.grid(row=4, column=1)
entry_non_urgent_order.insert(0, "171.297")  # Default value

Label(input_frame, text="Urgent Order:",bg='#73C2FB').grid(row=5, column=0)
entry_urgent_order = tk.Entry(input_frame)
entry_urgent_order.grid(row=5, column=1)
entry_urgent_order.insert(0, "127.667")  # Default value

Label(input_frame, text="Order Type A:",bg='#73C2FB').grid(row=6, column=0)
entry_order_type_a = tk.Entry(input_frame)
entry_order_type_a.grid(row=6, column=1)
entry_order_type_a.insert(0, "41.542")  # Default value

Label(input_frame, text="Order Type B:",bg='#73C2FB').grid(row=7, column=0)
entry_order_type_b = tk.Entry(input_frame)
entry_order_type_b.grid(row=7, column=1)
entry_order_type_b.insert(0, "113.294")  # Default value

Label(input_frame, text="Order Type C:",bg='#73C2FB').grid(row=8, column=0)
entry_order_type_c = tk.Entry(input_frame)
entry_order_type_c.grid(row=8, column=1)
entry_order_type_c.insert(0, "162.284")  # Default value

Label(input_frame, text="Fiscal Sector Orders:",bg='#73C2FB').grid(row=9, column=0)
entry_fiscal_sector_orders = tk.Entry(input_frame)
entry_fiscal_sector_orders.grid(row=9, column=1)
entry_fiscal_sector_orders.insert(0, "18.156")  # Default value

Label(input_frame, text="Traffic Controller Orders:",bg='#73C2FB').grid(row=10, column=0)
entry_traffic_controller_orders = tk.Entry(input_frame)
entry_traffic_controller_orders.grid(row=10, column=1)
entry_traffic_controller_orders.insert(0, "49971")  # Default value

Label(input_frame, text="Banking Orders (1):",bg='#73C2FB').grid(row=11, column=0)
entry_banking_orders_1 = tk.Entry(input_frame)
entry_banking_orders_1.grid(row=11, column=1)
entry_banking_orders_1.insert(0, "33703")  # Default value

Label(input_frame, text="Banking Orders (2):",bg='#73C2FB').grid(row=12, column=0)
entry_banking_orders_2 = tk.Entry(input_frame)
entry_banking_orders_2.grid(row=12, column=1)
entry_banking_orders_2.insert(0, "69054")  # Default value
Label(input_frame, text="Banking Orders (3):",bg='#73C2FB').grid(row=13, column=0)
entry_banking_orders_3 = tk.Entry(input_frame)
entry_banking_orders_3.grid(row=13, column=1)
entry_banking_orders_3.insert(0, "18423")  # Default value



# Create buttons
btn_predict = tk.Button(main_frame, text="Predict Total Orders", command=predict_orders, width =40)
btn_predict.grid(row=12, column=0, columnspan=2, pady=5)
btn_show_coefficients = tk.Button(main_frame, text="Show Coefficients", command=show_coefficients, width= 40 )
btn_show_coefficients.grid(row=13, column=0, columnspan=2, pady=5)
# Result Frame
result_frame = LabelFrame(main_frame, text="Prediction Result", padx=10, font=('Arial', 12, 'bold'), bg='#73C2FB')
result_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
# Result Label
result_label = tk.Label(result_frame, text="",bg='#73C2FB', font=('Arial', 12))
result_label.pack(padx=10, pady=10)
# Run the application
root.mainloop()


