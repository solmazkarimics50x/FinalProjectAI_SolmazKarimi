import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from tkinter import messagebox as msg
from PIL import Image,ImageTk
import pathlib
import webbrowser
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os


from .preProcessing_regression import preProcessing_regression





class MainFormRegression:
    def __init__(self, parent):
        """
        Initializes the main application window and sets up the GUI components.
        """
        self.parent = parent  # Store reference to parent window
        # self.root = tk.Tk()
        self.root = tk.Toplevel(parent)
        self.root.title("Regression...") # Set the window title
        self.root.resizable(0,0) # Disable window resizing
        self.root.geometry("1080x430") # Set the window size#("580x430")
        # Center the window on the screen
        x = int(self.root.winfo_screenwidth()/2 - 1080 / 2 )
        y = int(self.root.winfo_screenheight() / 2 - 430 / 2)
        self.root.geometry(f'+{x}+{y}')
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle window close event

        # Create a style for the GUI
        self.style = ttk.Style()
        # Configure button styles
        self.style.map("TButton",
                       background = [("active","#4CFE50"),("!active", "SystemButtonFace")],
                       foreground= [("active","blue"),("!active","black")]
                       )
        self.df = pd.DataFrame() # Initialize an empty DataFrame for data storage
        # Define paths for datasets
        self.dataSet_path=  {
            # 'User_Knowledge': './Project_Clustering_SolmazKarimi/clustering_data/User Knowledge Modeling/',
            'Age_Abalone': './Project_LinearRegression_SolmazKarimi/regression_data/Find out the age of Abalone/',
            'PriceHouse': './Project_LinearRegression_SolmazKarimi/regression_data/PredictThePriceOfAHouse/',
            'SeminalQuality': './Project_LinearRegression_SolmazKarimi/regression_data/PredictTheSeminalQualityOfAnIndividual/',
            'FuelEfficiencyCar': './Project_LinearRegression_SolmazKarimi/regression_data/PredictTheFuelEfficiencyOfACar/',
            'TotalDemandOrders': './Project_LinearRegression_SolmazKarimi/regression_data/PredictTheTotalNumberOfDemandOFOrOrders/',
            'PollutionLevelCity' : './Project_LinearRegression_SolmazKarimi/regression_data/ForecastPollutionLevelOfACity/'
        }
        # Define filenames for datasets
        self.dataSet_file = {
            # 'User_Knowledge': 'Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.csv',
            'Age_Abalone': 'preprocessed_abalone_data.csv',
            'PriceHouse': 'Real estate valuation data set.csv',
            'SeminalQuality': 'fertility_Diagnosis.csv',
            'FuelEfficiencyCar': 'auto-mpg.csv',
            'TotalDemandOrders': 'Daily_Demand_Forecasting_TotalOrders.csv',
            'PollutionLevelCity': 'PRSA_data.csv'

        }
        # Create various frames for the GUI
        self.create_input_frame()
        self.create_image_frame()
        self.create_progress_frame()
        self.create_results_frame()
        self.create_table_frame()



    def run(self):
        """
                Starts the main event loop of the Tkinter application.
        """
        self.root.mainloop()

    def create_input_frame(self):
        """
        Creates the input frame for selecting datasets and parameters.
        """
        self.input_frame = tk.Frame(self.root , bg='#73C2FB')
        self.input_frame.grid(row = 0, column = 1 , padx=(5,0), pady = (10,0) ,sticky="snew")
        # Label and combobox for dataset selection
        self.combo_label = tk.Label(self.input_frame, text = "Select Dataset:", bg='#73C2FB')
        self.combo_label.grid(row = 0, column= 0 , padx =5 , pady= 5 , sticky= "w")
        self.dataset_entry = tk.StringVar() # Variable to hold the selected dataset
        self.dataset_combo = ttk.Combobox(self.input_frame, textvariable= self.dataset_entry , state= 'readonly',width = 30)
        self.dataset_combo["value"] = list(self.dataSet_file.keys()) # Populate combobox with dataset names
        self.dataset_combo.current= "" # Set default selection
        self.dataset_combo.grid(row= 0 , column = 1 , padx= 5 , pady= 5 , sticky = "e")
        self.dataset_combo.bind('<<ComboboxSelected>>',self.on_dataset_selected ) # Bind selection event

        self.useless_columns_lbl = tk.Label(self.input_frame , text = " Useless Columns: ",bg='#73C2FB')
        self.useless_columns_lbl.grid(row = 1 , column = 0 , padx = 5 , pady = 5 , sticky= "w")
        self.uselessCols_var = tk.StringVar()
        self.useless_columns_entry = ttk.Entry(self.input_frame , textvariable= self.uselessCols_var , width = 30)
        self.useless_columns_entry.grid(row = 1 , column = 1 , padx = 5 , pady = 5 , sticky = "ew")

        self.target_column_lbl = tk.Label(self.input_frame , text = " Target Column: ",bg='#73C2FB')
        self.target_column_lbl.grid(row = 2 , column = 0 , padx = 5 , pady = 5 , sticky= "w")
        self.target_column_var = tk.StringVar()
        self.target_column_entry = ttk.Entry(self.input_frame, textvariable= self.target_column_var , width = 30)
        self.target_column_entry.grid(row = 2 , column = 1 , padx = 5 , pady = 5 , sticky = "ew")

        # Radio buttons for scaling options
        self.scaleing_frame = tk.Frame(self.input_frame,bg='#73C2FB')
        self.scaleing_frame.grid(row=3, column=0, columnspan=2, sticky='nsew')
        self.scaling_var = tk.StringVar(value='zscore') # Default scaling method
        self.zscore_radio = tk.Radiobutton(self.scaleing_frame, text='Zscore Scaling', variable=self.scaling_var, value='zscore',bg='#73C2FB')
        self.minmax_radio = tk.Radiobutton(self.scaleing_frame, text='MinMax Scaling', variable=self.scaling_var, value='minmax',bg='#73C2FB')
        self.none_radio = tk.Radiobutton(self.scaleing_frame, text='None', variable=self.scaling_var, value='None',bg='#73C2FB')
        self.zscore_radio.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.minmax_radio.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.none_radio.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        # Label and entry for outlier threshold
        self.outlier_testsize_frame = tk.Frame(self.input_frame, bg='#73C2FB' )
        self.outlier_testsize_frame.grid(row = 4 , column =0 , columnspan= 4 , sticky= "nsew")
        self.outlier_lbl = tk.Label(self.outlier_testsize_frame, text='Outlier Threshold:',bg='#73C2FB')
        self.outlier_lbl.grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.outlierTr_var = tk.StringVar(value=1.5) # Default outlier threshold
        self.outlierTr_entry = ttk.Entry(self.outlier_testsize_frame, textvariable=self.outlierTr_var, width=5)
        self.outlierTr_entry.grid(row=4, column=1, padx=5, pady=5, sticky='e')

        # Label and entry for test size
        self.test_size_lbl = tk.Label(self.outlier_testsize_frame, text='Test Size:',bg='#73C2FB')
        self.test_size_lbl.grid(row=4, column=2, padx=5, pady=5, sticky='w')
        self.test_size_var = tk.StringVar(value=0.3) # Default test size
        self.test_size_entry = ttk.Entry(self.outlier_testsize_frame, textvariable=self.test_size_var, width=5)
        self.test_size_entry.grid(row=4, column=3, padx=5, pady=5, sticky='e')

        # Frame for input buttons
        self.input_btns_frame = tk.LabelFrame(self.input_frame,bg='#73C2FB', text = "InputButton...")
        self.input_btns_frame.grid(row=5, column=0, columnspan=2, sticky='nsew')
        self.load_btn = ttk.Button(self.input_btns_frame, text='Load Dataset', width=15,
                                   command=lambda: self.load_dataset(self.dataset_entry.get()), style='TButton')
        self.load_btn.grid(row=0, column=0, padx=(25,10), pady=5, sticky='ew')

        self.calc_LinearRegression_btn = ttk.Button(self.input_btns_frame, text='Calculate LinearRegression', width=25,
                                        command=self.calc_LinearRegression, style='TButton')
        self.calc_LinearRegression_btn.grid(row=0, column=1, padx=10, pady=5, sticky='ew')





    def on_dataset_selected(self, event):
        """
        Handles the event when a dataset is selected from the combobox.
        Updates the useless columns entry and displays the dataset image.
        """

        # Task1:

        # dataset_name = self.dataset_entry.get()  # Get the selected dataset name
        # image_path = self.dataSet_path.get(dataset_name) + 'image.jpg'  # Construct the image path
        dataset_name = self.dataset_entry.get()
        base_path = self.dataSet_path.get(dataset_name)
        if base_path is None:
            # Handle missing path gracefully
            msg.showerror("Error", f"No path found for dataset '{dataset_name}'")
            return
        image_path = base_path + 'image.jpg'
        try:
            img = Image.open(image_path)
            img = img.resize((200, 150), Image.LANCZOS)
            self.photo_img = ImageTk.PhotoImage(img)
            self.lbl_photo.config(image=self.photo_img)
            self.lbl_photo.image = self.photo_img
        except Exception as e:
            msg.showerror("Error", f"Failed to load image:\n{e}")
        # Task2:
        # Update useless columns based on the selected dataset
        if dataset_name == 'Age_Abalone' :
            self.uselessCols_var.set('')#'Sex_I'
        elif dataset_name == 'PriceHouse' :
            self.uselessCols_var.set( 'X1 transaction date')
        elif dataset_name == 'SeminalQuality':
            self.uselessCols_var.set('Output_N')
        elif dataset_name == 'FuelEfficiencyCar':
            self.uselessCols_var.set('car_name')


        else :
            self.uselessCols_var.set('')


        #Task3:
        #Update target column based on the selected dataset
        if dataset_name == 'Age_Abalone' :
             self.target_column_var.set ('Rings')
        elif dataset_name == 'PriceHouse':
            self.target_column_var.set('Y house price of unit area')
        elif dataset_name == 'SeminalQuality':
            self.target_column_var.set('Output_O')
        elif dataset_name == 'FuelEfficiencyCar':
            self.target_column_var.set('mpg')
        elif dataset_name == 'TotalDemandOrders':
            self.target_column_var.set('Target (Total orders)')
        elif dataset_name == 'PollutionLevelCity' :
            self.target_column_var.set('pm2.5')


        else :
            self.target_column_var.set('')





    def create_image_frame(self):
        """
        Creates a frame for displaying the dataset image and a button to read the README file.
        """
        self.image_frame = tk.Frame(self.root  , bg='#73C2FB' , bd = 2)
        self.image_frame.grid(row = 0, column = 0 , padx = 10 , pady = (10,0), sticky = "snew")
        self.blank_img = ImageTk.PhotoImage(Image.new("RGB" ,(200,150) ,color = "white"))# Create a blank image
        self.lbl_photo = tk.Label(self.image_frame , image= self.blank_img) # Label to display the image
        self.lbl_photo.grid(row =0 , column = 0 , padx = 5 , pady = 5 , sticky = "snew")
        self.read_md = ttk.Button(self.image_frame, text='Read MarkDown', command=self.open_md, style='TButton') # Button to open README
        self.read_md.grid(row=1, column=0, padx=5, pady=5, sticky='s')

    def create_results_frame(self):
        """
        Creates a frame for displaying result buttons for various plots.
        """
        self.result_frame = tk.LabelFrame(self.root, relief='sunken',text="Result..." , bd=2) # Create a frame for results
        self.result_frame.grid(row=1, column=1, padx=(5,0), pady=(5,0), sticky='nsew')
        self.corrPlot_btn = ttk.Button(self.result_frame, text='Correlation Matrix', command=self.open_corrPlot, style='TButton', width= 20) # Button for correlation matrix
        self.corrPlot_btn.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.boxPlot_btn = ttk.Button(self.result_frame, text='Box Plot Chart', command=self.open_boxPlot, style='TButton', width= 20) # Button for box plot
        self.boxPlot_btn.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.result_score_err_frame = tk.Frame(self.result_frame , relief="flat")
        self.result_score_err_frame.grid(row = 1, column = 0 , columnspan= 3 , padx= 5 , pady= 5 , sticky= "nsew")
        self.train_score_lbl = tk.Label(self.result_score_err_frame , text = " Train Score: ")
        self.train_score_lbl.grid(row = 0 , column= 0 , padx = 5 , pady= 5 , sticky = "ew")
        self.test_score_lbl = tk.Label(self.result_score_err_frame , text = " Test Score: " )
        self.test_score_lbl.grid(row = 0 , column= 2 , padx = 5 , pady= 5 , sticky = "ew")
        self.mae_lbl = tk.Label(self.result_score_err_frame , text = " MAE: " )
        self.mae_lbl.grid(row = 1 , column= 0 , padx = 5 , pady= 5 , sticky = "w")
        self.mse_lbl = tk.Label(self.result_score_err_frame , text = " MSE: " )
        self.mse_lbl.grid(row = 1 , column= 1 , padx = 5 , pady= 5 , sticky = "ew")
        self.rmse_lbl = tk.Label(self.result_score_err_frame , text = " RMSE: " )
        self.rmse_lbl.grid(row = 1 , column= 2 , padx = 5 , pady= 5 , sticky = "e")

        self.plot_frame = tk.Frame(self.result_frame, relief="flat")
        self.plot_frame.grid(row=2, column=0, columnspan=3, sticky="w")

        self.residuals_plot_btn = ttk.Button(self.plot_frame, text='Residuals Plot', style='TButton', width=13,command=self.show_residuals_plot)
        self.residuals_plot_btn.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        self.coefficients_plot_btn = ttk.Button(self.plot_frame, text='Coefficients Plot', style='TButton', width=15,command=self.show_coefficients_plot)
        self.coefficients_plot_btn.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        self.predictNew_btn = ttk.Button(self.plot_frame, text='PredictNewSample', style='TButton', width=17,
                                         command=self.open_predict_new_sample_window)
        self.predictNew_btn.grid(row=0, column=3, padx=5, pady=5, sticky='w')



        # create a back button in the plot_frame
        self.back_btn = ttk.Button(self.plot_frame, text='Back to Main <---', width=30
                                   ,command=self.go_back, style='TButton')
        self.back_btn.grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky='ns')

    def go_back(self):
        """
        Closes the current window and returns to the main form.
        """
        self.root.destroy()  # Close the linearregression window
        self.parent.deiconify()  # Show the main form again





    def create_progress_frame(self):
        """
        Creates a frame for displaying progress messages.
        """

        self.progress_frame = tk.LabelFrame(self.root, relief='sunken', text = "Progress...")  # Create a frame for progress
        self.progress_frame.grid(row=1, column=0, padx=10, pady=(5,0), sticky='nsew')
        self.status_text = tk.Text(self.progress_frame, height=10, width=33, wrap=tk.WORD, font=('Arial', 8),fg="green")  # Text widget for status messages
        self.status_text.grid(row=0, column=0, padx=0, pady=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(self.progress_frame , orient= 'vertical', command= self.status_text.yview)
        scrollbar.grid(row = 0 , column = 1,sticky="sn")

        # Add a vertical scrollbar to the text widget
        scrollbar = ttk.Scrollbar(self.progress_frame, orient='vertical', command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.status_text.configure(yscrollcommand=scrollbar.set)  # Link scrollbar to text widget

    def add_status(self, message):
        """
        Adds a status message to the progress text widget.
        Parameters:
        - message: str - The message to display.
        """
        self.status_text.insert(tk.END, message + '\n')  # Insert the message at the end
        self.status_text.see(tk.END) # Scroll to the end
        self.root.update() # Update the GUI

    def create_table_frame(self):
        """
        Creates a frame for displaying the data table.
        """

        self.table_frame = tk.Frame(self.root, width=500, height=200) # Create a frame for the table
        self.table_frame.grid(row=0, column=2,rowspan= 2, columnspan=2, padx=5, pady=5, sticky='nsew')
        self.table_frame.grid_propagate(False) # Prevent the frame from resizing
        self.table_frame.grid_rowconfigure(0, weight=1) # Configure row weight
        self.table_frame.grid_columnconfigure(0, weight=1) # Configure column weight


    def open_md(self):
        """
        Opens the README file for the selected dataset in a web browser.
        """
        dataset_name = self.dataset_entry.get() # Get the selected dataset name
        local_path = self.dataSet_path.get(dataset_name) + "README.html"  # Construct the path to the README file
        absolute_path = pathlib.Path(local_path).resolve() # Get the absolute path
        file_url = absolute_path.as_uri()  # Convert to a file URL
        webbrowser.open(file_url)  # Open the URL in the default web browser

    def load_dataset(self, dataset_name):
        """
        Loads the selected dataset into a DataFrame and preprocesses it.
        Parameters:
        - dataset_name: str - The name of the dataset to load.
        """
        if not dataset_name :
            msg.showwarning("Warning", f"Please select a dataset.") # Show warning if no dataset is selected
            return
        self.distortions = [] # Reset distortions list
        self.status_text.delete(1.0, tk.END)  # Clear previous status messages
        if dataset_name:
            file_path = self.dataSet_path.get(dataset_name) + self.dataSet_file.get(dataset_name)  # Construct the file path

            self.df = pd.read_csv(file_path, sep=',')  # Load dataset with default separator

            self.add_status(f"-Dataset csv loaded successfully.")  # Update status
            target_col_raw = self.target_column_var.get()
            if ',' in target_col_raw:
                target_col = [col.strip() for col in target_col_raw.split(',')]
            else:
                target_col = target_col_raw
            # Preprocess the loaded dataset
            self.df_preProcessed, self.numeric_cols = preProcessing_regression(
                self.df,
                self.uselessCols_var.get().split(','),
                dataset_name,
                outlier_threshold=float(self.outlierTr_var.get()),
                scaler=self.scaling_var.get(),
                main_form=self,
                target_col=target_col
            )
            # self.df_preProcessed, self.numeric_cols = preProcessing_regression(self.df,
            #                         self.uselessCols_var.get().split(','),
            #                         dataset_name,
            #                         outlier_threshold=float(self.outlierTr_var.get()),
            #                         scaler=self.scaling_var.get(),
            #                         main_form=self,
            #                         target_col=self.target_column_var.get()
            #                          )
            # print(self.df_preProcessed.head().to_string()) # Print the first few rows of the preprocessed DataFrame
            self.show_table()  # Display the data in the table

    def open_corrPlot(self):
        if self.df.empty:
            msg.showwarning("Warning", "Please load a dataset first.")
            return

        plot_window = tk.Toplevel(self.root)
        plot_window.title('Correlation Matrix Chart')

        g = sns.heatmap(self.df_preProcessed[self.numeric_cols].corr(numeric_only=True), annot=True, cmap='coolwarm', square=True)
        fig = g.figure
        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def on_window_close():
            plt.close(fig)
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_window_close)

    def open_boxPlot(self):
        if self.df.empty:
            msg.showwarning("Warning", "Please load a dataset first.")
            return

        plot_window = tk.Toplevel(self.root)
        plot_window.title('Box Plot Chart')

        data = self.df_preProcessed[self.numeric_cols].copy()
        # if 'cluster_' in self.df_preProcessed.columns:
        #     data['cluster_'] = self.df_preProcessed['cluster_']
        g = sns.boxplot(data, orient='v')
        fig = g.figure
        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def on_window_close():
            plt.close(fig)
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_window_close)

    def show_residuals_plot(self):
        if not hasattr(self, 'y_test') or not hasattr(self, 'y_test_pred'):
            msg.showwarning("Warning", "Please run Linear Regression calculation first.")
            return
        residuals = self.y_test - self.y_test_pred
        rmse = np.sqrt(np.mean(residuals ** 2))  # Calculate RMSE
        # Create a new Tkinter window
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Residuals Plot")
        # Create a matplotlib figure
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        # Scatter plot of residuals
        ax.scatter(self.y_test_pred, residuals, alpha=0.6)
        # Horizontal zero line
        ax.axhline(y=0, color='r', linestyle='--', label='Zero Residual')
        # Horizontal RMSE lines
        ax.axhline(y=rmse, color='g', linestyle='--', label=f'+RMSE ({rmse:.4f})')
        ax.axhline(y=-rmse, color='g', linestyle='--', label=f'-RMSE ({rmse:.4f})')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.set_title('Residuals Plot with RMSE bounds')
        ax.grid(True)
        # Add legend
        ax.legend()

        # Embed the figure in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        def on_window_close():
            plt.close(fig)
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_window_close)

    def show_coefficients_plot(self):
        if not hasattr(self, 'model') or not hasattr(self, 'feature_names'):
            msg.showwarning("Warning", "Please run Linear Regression calculation first.")
            return
        coefs = self.model.coef_
        features = self.feature_names

        coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefs})
        coef_df['abs_coef'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values(by='abs_coef', ascending=False)
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Coefficients Plot")
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        bars = ax.bar(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')
        # ax.set_xticklabels(coef_df['Feature'], rotation=45, ha='right')
        ax.set_xticks(range(len(coef_df['Feature'])))
        ax.set_xticklabels(coef_df['Feature'], rotation=45, ha='right')
        ax.set_xlabel('Features')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Linear Regression Coefficients (sorted by absolute value)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        def on_window_close():
            plt.close(fig)
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_window_close)





    def on_close(self):
        plt.close("all")
        self.root.destroy()




    def calc_LinearRegression(self):
        """
        Trains and tests a Linear Regression model on the preprocessed dataset,
        calculates performance metrics, displays them, and saves/loads the model using joblib.
        """
        if self.df_preProcessed.empty:
            msg.showwarning("Warning", "Please load and preprocess the dataset first.")
            return
        target_col = self.target_column_var.get().strip()
        if target_col == "":
            msg.showwarning("Warning", "Please specify the target column.")
            return
        if target_col not in self.df_preProcessed.columns:
            msg.showerror("Error", f"Target column '{target_col}' not found in dataset.")
            return

        try:
            test_size = float(self.test_size_var.get())
            if not (0 < test_size < 1):
                raise ValueError
        except ValueError:
            msg.showerror("Error", "Test Size must be a float between 0 and 1.")
            return
        #if target_col == ['Output_N',  'Output_O'] :

        # Prepare features and target
        X = self.df_preProcessed.drop(columns=[target_col])
        #if target_col != ['Output_N', 'Output_O']:
        y = self.df_preProcessed[target_col]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.y_test = y_test

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        self.feature_names = X.columns
        # Predict
        # y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        self.y_test_pred = y_test_pred
        # Calculate metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        MAE = mean_absolute_error(y_test, y_test_pred)
        MSE = mean_squared_error(y_test, y_test_pred)
        RMSE = np.sqrt(MSE)


        # Update the labels in the result frame with the calculated values
        self.train_score_lbl.config(text=f"Train Score: {train_score:.4f}")
        self.test_score_lbl.config(text=f"Test Score: {test_score:.4f}")
        self.mae_lbl.config(text=f"MAE: {MAE:.4f}")
        self.mse_lbl.config(text=f"MSE: {MSE:.4f}")
        self.rmse_lbl.config(text=f"RMSE: {RMSE:.4f}")


        # Display results in the status text widget
        self.add_status("Linear Regression Results:")
        self.add_status(f"Train Score (R^2): {train_score:.4f}")
        self.add_status(f"Test Score (R^2): {test_score:.4f}")
        self.add_status(f"Mean Absolute Error (MAE): {MAE:.4f}")
        self.add_status(f"Mean Squared Error (MSE): {MSE:.4f}")
        self.add_status(f"Root Mean Squared Error (RMSE): {RMSE:.4f}")


        # Save the model using joblib
        dataset_name = self.dataset_entry.get()
        model_dir = self.dataSet_path.get(dataset_name)
        if model_dir is None:
            msg.showerror("Error", f"No path found for dataset '{dataset_name}' to save the model.")
            return


        # After evaluation and saving the model trained on train split, train final model on all data:
        final_model = LinearRegression()
        final_model.fit(X, y)  # Train on entire dataset
        # Save the final model
        final_model_path = os.path.join(model_dir, "linear_regression_final_model.joblib")
        joblib.dump(final_model, final_model_path)
        self.add_status(f"Final model trained on all data saved to: {final_model_path}")
        # Optionally load and verify final model
        loaded_final_model = joblib.load(final_model_path)
        final_score = loaded_final_model.score(X, y)
        self.add_status(f"Final model score on all data: {final_score:.4f}")
        # Store final model for prediction use
        self.model = final_model

    def open_predict_new_sample_window(self):
        if not hasattr(self, 'model') or not hasattr(self, 'feature_names'):
            msg.showwarning("Warning", "Please run Linear Regression calculation first.")
            return
        # Create new window
        self.predict_window = tk.Toplevel(self.root, background='#73C2FB')
        self.predict_window.title("Predict New Sample")
        # Dictionary to hold slider widgets for each feature
        self.feature_sliders = {}
        # For each feature, create a slider with min and max from the training data
        X = self.df_preProcessed[self.feature_names]
        row = 0
        for feature in self.feature_names:
            # Get min and max for slider range
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            mean_val = float(X[feature].mean())
            # Label for feature
            label = tk.Label(self.predict_window, text=feature, bg = '#73C2FB')
            label.grid(row=row, column=0, padx=5, pady=5, sticky='w')
            # Create a scale (slider) widget
            # Use resolution=0.01 for float precision, adjust as needed
            slider = tk.Scale(self.predict_window, from_=min_val, to=max_val, orient=tk.HORIZONTAL, resolution=0.01,
                              length=300, bg = '#73C2FB')
            slider.set(mean_val)  # Set default to mean value
            slider.grid(row=row, column=1, padx=5, pady=5, sticky='ew')
            self.feature_sliders[feature] = slider
            row += 1
            # Predict button
        predict_btn = ttk.Button(self.predict_window, text="Predict", command=self.predict_new_sample)
        predict_btn.grid(row=row, column=0, columnspan=2, pady=10)
        # Label to show prediction result
        self.prediction_result_label = tk.Label(self.predict_window, text="", font=('Arial', 12, 'bold'),  bg = '#73C2FB')
        self.prediction_result_label.grid(row=row + 1, column=0, columnspan=2, pady=10)

    def predict_new_sample(self):
        # Collect feature values from sliders
        input_features = [self.feature_sliders[feature].get() for feature in self.feature_names]
        # Create a DataFrame with one row and columns named as features
        input_df = pd.DataFrame([input_features], columns=self.feature_names)
        # Predict using the trained model
        prediction = self.model.predict(input_df)[0]
        # Display the prediction result
        self.prediction_result_label.config(text=f"Predicted {self.target_column_var.get()}: {prediction:.4f}")








    def show_table(self):
        # Clear existing table if any
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        # Create container frame with fixed size
        container = ttk.Frame(self.table_frame, width=500, height=200)
        container.pack(fill='both', expand=True)
        container.pack_propagate(False)  # Prevent container from resizing

        # Create inner frame for treeview and scrollbars
        inner_frame = ttk.Frame(container)
        inner_frame.pack(fill='both', expand=True, padx=5, pady=5)
        inner_frame.pack_propagate(False)  # Prevent inner frame from resizing

        # Create Treeview with both scrollbars
        v_scroll = ttk.Scrollbar(inner_frame, orient='vertical')
        h_scroll = ttk.Scrollbar(inner_frame, orient='horizontal')
        tree = ttk.Treeview(inner_frame, yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set, height=8)

        # Configure scrollbars
        v_scroll.pack(side='right', fill='y')
        h_scroll.pack(side='bottom', fill='x')
        tree.pack(side='left', fill='both', expand=True)
        v_scroll.config(command=tree.yview)
        h_scroll.config(command=tree.xview)

        # Define columns
        columns = list(self.df_preProcessed.columns)
        tree['columns'] = columns

        # Format columns with fixed width
        tree.column('#0', width=0, stretch=False)  # Hidden first column
        col_width = 80  # Fixed width for all columns
        for col in columns:
            tree.column(col, anchor='w', width=col_width, stretch=False)
            tree.heading(col, text=col, anchor='w')

        # Add data (first 50 rows)
        for i in range(min(50, len(self.df_preProcessed))):
            values = self.df_preProcessed.iloc[i].values.tolist()
            values = [round(x, 3) if isinstance(x, float) else x for x in values]
            tree.insert('', 'end', values=values)






# app_Regression = MainFormRegression()
# app_Regression.run()
