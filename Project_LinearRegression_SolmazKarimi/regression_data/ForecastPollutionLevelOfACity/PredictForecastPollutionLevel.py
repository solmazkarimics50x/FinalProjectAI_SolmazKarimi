import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
from tkinter import ttk , filedialog

from tkinter import messagebox as msg
from PIL import Image ,ImageTk
import os
import webbrowser
from joblib import load



model = load("ForecastPollutionLevel.joblib")
scaler = load("scaler.joblib")  # Load the fitted scaler
Accuracy = load("Accuracy.joblib")

def open_readme():
    readme_path = os.path.abspath("README.md")
    if readme_path :
        webbrowser.open(f'file://{readme_path}')
    else:
        msg.showerror("File NOT Found", "README.md file not found.")

def predict_pollution_level():
    # Get user input
    input_data = [

        float(ent_dewp.get()),  # DEWP
        float(ent_temp.get()),  # TEMP
        float(ent_pres.get()),  # PRES
        float(ent_cbwd.get()), # cbwd
        float(ent_iws.get()),  # Iws
        float(ent_is.get()),  # Is
        float(ent_ir.get())  # Ir
    ]
    # Use the same feature names as during training
    columns = ['DEWP', 'TEMP', 'PRES','cbwd', 'Iws', 'Is', 'Ir']
    input_df = pd.DataFrame([input_data], columns=columns)
    # Scale the input data
    input_scale = scaler.transform(input_df)

    # Make Prediction
    prediction = model.predict(input_scale)

    #Display the result
    lbl_predict.config(text = f'Predict: {prediction[0]:.2f}')
    lbl_Accuracy.config(text=f'Accuracy: {Accuracy :.2f}%')

def plot_importance_features():
    features = ['DEWP', 'TEMP', 'PRES','cbwd', 'Iws', 'Is', 'Ir']
    coefficients = model.coef_
    df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients,
        'AbsCoefficient': np.abs(coefficients)
    }).sort_values('AbsCoefficient', ascending=False)

    colors = ['#2ecc71' if coef > 0 else '#e74c3c' for coef in df['Coefficient']]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df['Feature'], df['Coefficient'], color=colors, edgecolor='black')
    for bar in bars:
        width = bar.get_width()
        label_pos = width * 1.02 if width > 0 else width * 0.98
        plt.text(label_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Feature Importance (Coefficients)")
    plt.xlabel("Coefficient Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




#
# def load_form_main():
main_form =tk.Tk()
main_form.title('Regression Model')
main_form.geometry('700x590')
main_form.resizable(0,0)
x = int(main_form.winfo_screenwidth() /2 - 700/2)
y = int(main_form.winfo_screenheight() /2 - 590/2)
main_form.geometry(f'+{x}+{y}')

# Title & Description --------------------------------------------------------
title_frame=Frame(main_form,width=100,height=200,padx=10,pady=10)
title_frame.grid(row=0,column=0,columnspan=2,padx=20,pady=10,sticky='n')

title = Label(title_frame,text='Predict Forecate Pollution Level Of City',font='Calibri 14',fg='dark blue')
title.grid(row=0,column=0,padx=10,pady=0,sticky='n')

input_frame= tk.LabelFrame(main_form,text = 'InputField' ,  relief= 'sunken')
input_frame.grid(row =1 ,column= 0 , padx= (20,0), pady=(10,0) , sticky= 'nsew')

lbl_dewp = tk.Label(input_frame, text = 'DEWP: (0, -11, 8) ')
lbl_dewp.grid(row = 0 , column = 0 , padx = 5 , pady = 5 , sticky ='w')
ent_dewp_txt = tk.StringVar()
# ent_dewp_txt.set("0, -11, 8")  # Default value
ent_dewp = tk.Entry(input_frame, width = 20, textvariable=ent_dewp_txt )
ent_dewp.grid(row = 0 , column = 1 ,padx =(10,10), sticky = 'e')
lbl_temp = tk.Label(input_frame, text = 'TEMP:(27.000000, -3.000000, 2.000000) ')
lbl_temp.grid(row = 1 , column = 0 , padx = 5 , pady = 5 , sticky ='w')
ent_temp_txt = tk.StringVar()
# ent_temp_txt.set("27.000000, -3.000000, 2.000000")
ent_temp = tk.Entry(input_frame, width = 20, textvariable=ent_temp_txt )
ent_temp.grid(row = 1 , column = 1 ,padx =(10,10), sticky = 'e')
lbl_pres = tk.Label(input_frame, text = 'PRES: (1025.000000, 997.000000, 1044.000000) ')
lbl_pres.grid(row = 2 , column = 0 , padx = 5 , pady = 5 , sticky ='w')
ent_pres_txt = tk.StringVar()
# ent_pres_txt.set("1025.000000, 997.000000, 1044.000000")
ent_pres = tk.Entry(input_frame, width = 20, textvariable=ent_pres_txt )
ent_pres.grid(row = 2 , column = 1 ,padx =(10,10), sticky = 'e')
lbl_cbwd = tk.Label(input_frame, text = "cbwd: ({'NW':0, 'cv':1, 'NE':2, 'SE':3}) ")
lbl_cbwd.grid(row = 3 , column = 0 , padx = 5 , pady = 5 , sticky ='w')
ent_cbwd_txt = tk.StringVar()
# ent_cbwd_txt.set("1025.000000, 997.000000, 1044.000000")
ent_cbwd = tk.Entry(input_frame, width = 20, textvariable=ent_cbwd_txt )
ent_cbwd.grid(row = 3 , column = 1 ,padx =(10,10), sticky = 'e')
lbl_iws = tk.Label(input_frame, text = 'Iws:(24.15, 27.28, 16.10) ')
lbl_iws.grid(row = 4 , column = 0 , padx = 5 , pady = 5 , sticky ='w')
ent_iws_txt = tk.StringVar()
# ent_iws_txt.set("24.15, 27.28, 16.10")
ent_iws = tk.Entry(input_frame, width = 20, textvariable=ent_iws_txt )
ent_iws.grid(row = 4 , column = 1 ,padx =(10,10), sticky = 'e')
lbl_is = tk.Label(input_frame, text = 'Is:(5, 10, 22) ')
lbl_is.grid(row = 5, column = 0 , padx = 5 , pady = 5 , sticky ='w')
ent_is_txt = tk.StringVar()
# ent_is_txt.set("5, 10, 22")
ent_is = tk.Entry(input_frame, width = 20, textvariable=ent_is_txt )
ent_is.grid(row = 5 , column = 1 ,padx =(10,10), sticky = 'e')
lbl_ir = tk.Label(input_frame, text = 'Ir:(11, 19, 17) ')
lbl_ir.grid(row = 6, column = 0 , padx = 5 , pady = 5 , sticky ='w')
ent_ir_txt = tk.StringVar()
# ent_ir_txt.set("11, 19, 17")
ent_ir = tk.Entry(input_frame, width = 20, textvariable=ent_ir_txt )
ent_ir.grid(row = 6 , column = 1 ,padx =(10,10), sticky = 'e')





image_frame = tk.Frame(main_form , relief= 'groove',bd=2)
image_frame.grid(row = 1 , column = 1 ,padx= 40, pady= 5, sticky = 'e')
# Load and display image.jpg automatically
img = Image.open("image.jpg")
img = img.resize((200, 150)) # Resize to fit the label
photo = ImageTk.PhotoImage(img)
lbl_image = tk.Label(image_frame,image = photo)
lbl_image.image = photo
lbl_image.grid(row = 0 , column = 1 , padx = 10 , pady = 10)
bt_predict = ttk.Button(main_form , text='Prediction', width = 20, command=predict_pollution_level ) # open_read_me
bt_predict.grid(row = 2 , column= 0,padx = 10 , pady = 5, sticky= 'n')
bt_reed_me = ttk.Button(main_form , text='Read MarkDown', width = 20, command=open_readme )
bt_reed_me.grid(row = 2 , column= 1,padx = 10 , pady = 5, sticky= 'n')

result_frame = LabelFrame(main_form,text = 'Result Predict' , relief= 'groove', font= 'Arial 12', fg= 'blue')
result_frame.grid(row = 3 , column = 0,columnspan= 1 ,padx = (10,0) , pady = (10,0), sticky= 'snew')
lbl_predict = Label(result_frame, text = 'Predict: ', font = 'Arial 10'  , fg = 'black')
lbl_predict.grid(row = 0 , column = 0 , sticky = 'w')

lbl_Accuracy = Label(result_frame, text = 'Accouracy: ', font = 'Arial 10'  , fg = 'black')
lbl_Accuracy.grid(row = 1 , column = 0 , sticky = 'w')

bt_importance_feature = ttk.Button(main_form, text = 'Plot Importance Feature', width = 30,command=plot_importance_features)
bt_importance_feature.grid(row = 4 , column = 0, columnspan= 2 ,padx = (10,0) , pady = (10,20) ,sticky = 'sn')
# bt_importance_feature.config(command=plot_importance_features)




main_form.mainloop()




