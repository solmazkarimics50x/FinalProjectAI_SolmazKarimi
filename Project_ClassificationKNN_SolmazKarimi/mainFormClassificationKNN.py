from tkinter import *
import tkinter as tk
from tkinter import ttk
from  tkinter import messagebox as msg
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk

import pathlib
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib
import os


from .preProcessing_classificationKNN import preProcessing_classificationKNN




class MainFormKNN:
    def __init__(self,parent):
        """
        Initializes the main application window and sets up the GUI components.
        """
        self.parent = parent  # Store reference to parent window
        # self.root = tk.Tk() # Create the main window
        self.root = tk.Toplevel(parent)
        self.root.title("Classification KNN...")# Set the window title
        self.root.resizable(0,0)# Disable window resizing
        self.root.geometry("1090x430") # Set the window size #"590x630"
        x = int(self.root.winfo_screenwidth() / 2 - 1090 / 2 )
        y = int (self.root.winfo_screenheight() / 2 - 430 / 2 )
        self.root.geometry(f'+{x}+{y}')
        self.root.protocol("WM_DELETE_WINDOW", self.on_close) # Handle window close event

        # Create a style for the GUI
        self.style = ttk.Style()
        self.style.map("TButton",
                       background = [("active" ,"#4CFE50" ),("!active", "SystemButtonFace")],
                       foreground = [("active" ,"blue"),("!active", "black")]
                       )

        self.df = pd.DataFrame() # Initialize an empty DataFrame for data storage
        # Define paths for datasets
        self.dataSet_path = {
            "Chance_Bankruptcy" : "./Project_ClassificationKNN_SolmazKarimi/classification_knn_data/Estimate the chance of bankruptcy from qualitative parameters by experts/",
            "WIFI_Signal_Strength" : "./Project_ClassificationKNN_SolmazKarimi/classification_knn_data/Estimate the location from WIFI Signal Strength/",
            "Predict_Acceptability_Car": "./Project_ClassificationKNN_SolmazKarimi/classification_knn_data/PredictTheAcceptabilityOfACar/"

        }

        self.dataSet_file = {
            "Chance_Bankruptcy":"Qualitative_Bankruptcy.csv",
            "WIFI_Signal_Strength": "wifi_localization.csv",
            "Predict_Acceptability_Car" : "car.csv"

        }

        # Create various frames for the GUI

        self.create_image_frame()
        self.create_input_frame()
        self.create_progress_frame()
        self.create_results_frame()
        self.create_table_frame()
        self.distortions = []  # Initialize a list to store KNN distortions



    def create_image_frame(self):
        self.image_frame = tk.Frame(self.root , bg='#73C2FB' , bd = 2)
        self.image_frame.grid(row = 0 , column= 0 ,padx = (10,0) , pady = (10,0), sticky= "snew" )
        self.black_img = ImageTk.PhotoImage(Image.new("RGB" ,(200,150) , color = "white"))# Create a blank image
        self.lbl_photo =  tk.Label(self.image_frame , image= self.black_img) # Label to display the image
        self.lbl_photo.grid(row = 0 , column = 0 , padx = 10 , pady = 10 , sticky = "snew")
        self.read_md = ttk.Button(self.image_frame , text='Read MarkDown', command = self.open_md, style='TButton') # Button to open README
        self.read_md.grid(row = 1 , column = 0 , padx =5 ,pady = (10,5) , sticky = "s")

    def create_input_frame(self):
        """
        Creates the input frame for selecting datasets and parameters.
        """
        self.input_frame = tk.Frame(self.root, bg='#73C2FB' )
        self.input_frame.grid(row =0 , column = 1 , padx = (10,0 ) , pady = (10,0), sticky= "snew")
        # Label and combobox for dataset selection
        self.combo_label = tk.Label(self.input_frame , text = "Select Dataset:", bg='#73C2FB')
        self.combo_label.grid(row = 0 , column = 0 , padx = 5 , pady = 5 , sticky="w")
        self.dataset_entry = tk.StringVar() # Variable to hold the selected dataset
        self.dataset_combo = ttk.Combobox(self.input_frame , textvariable=self.dataset_entry , width = 30, state = 'readonly')
        self.dataset_combo.grid(row = 0 , column = 1 , padx = 5, pady = 5 , sticky ="e")
        self.dataset_combo["value"] = list(self.dataSet_file.keys()) # Populate combobox with dataset names
        self.dataset_combo.current = '' # Set default selection
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
        self.scaleing_frame.grid(row=3, column=0, columnspan=2 ,sticky='nsew')
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

        self.n_neighbors_lbl = tk.Label(self.input_frame , text = 'N_neighbors: ',bg='#73C2FB')
        self.n_neighbors_lbl.grid(row = 5 ,column = 0 ,  padx = 5 , pady = 5 , sticky= 'w')
        self.n_neighbors_var = tk.StringVar(value = 9 )
        self.n_neighbors_entry = ttk.Entry(self.input_frame , textvariable= self.n_neighbors_var ,width = 5 )
        self.n_neighbors_entry.grid(row = 5 , column = 1 , padx = 5 , pady = 5 , sticky= 'w')

        # Frame for input buttons
        self.input_btns_frame = tk.LabelFrame(self.input_frame,bg='#73C2FB', text = "InputButton...")
        self.input_btns_frame.grid(row=6, column=0, columnspan=2, sticky='nsew')
        self.load_btn = ttk.Button(self.input_btns_frame, text='Load Dataset', width=15,
                                    command=lambda: self.load_dataset(self.dataset_entry.get()),style='TButton')
        self.load_btn.grid(row=0, column=0, padx=(25,10), pady=5, sticky='ew')

        self.calc_LinearRegression_btn = ttk.Button(self.input_btns_frame, text='Calculate KNN', width=25,
                        command=self.calc_classificationKNN, style='TButton')
        self.calc_LinearRegression_btn.grid(row=0, column=1, padx=10, pady=5, sticky='ew')


    def create_progress_frame(self):
        """
        Creates a frame for displaying progress messages.
        """
        self.progress_frame = tk.LabelFrame(self.root, relief='sunken',text="Progress...")  # Create a frame for progress
        self.progress_frame.grid(row=1, column=0, padx=(10,5), pady=(5, 0), sticky='nsew')
        self.status_text = tk.Text(self.progress_frame ,font = ('Arial',8), fg = 'green', height = 10 , width = 33 , wrap = tk.WORD)# Text widget for status messages
        self.status_text.grid(row = 0 , column =0, padx = 0 , pady = 0 , sticky= 'snew' )
        # Add a vertical scrollbar to the text widget
        scrollbar = ttk.Scrollbar(self.progress_frame,command=self.status_text.yview , orient = 'vertical')
        scrollbar.grid(row = 0 , column = 1, sticky= 'sn')
        self.status_text.configure(yscrollcommand=scrollbar.set )# Link scrollbar to text widget


    def add_status(self, message):
        """
        Adds a status message to the progress text widget.
        Parameters:
        - message: str - The message to display.
        """
        self.status_text.insert(tk.END, message + '\n')  # Insert the message at the end
        self.status_text.see(tk.END) # Scroll to the end
        self.root.update() # Update the GUI

    def create_results_frame(self):
        """
        Creates a frame for displaying result buttons for various plots.
        """
        self.result_frame = tk.LabelFrame(self.root, relief='sunken', text="Result...",
                                          bd=2)  # Create a frame for results
        self.result_frame.grid(row=1, column=1, padx=(10, 0), pady=(5, 0), sticky='nsew')
        self.corrPlot_btn = ttk.Button(self.result_frame, text='Correlation Matrix',
                                       style='TButton', command=self.open_corrPlot, width=20)  # Button for correlation matrix
        self.corrPlot_btn.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.boxPlot_btn = ttk.Button(self.result_frame, text='Box Plot Chart',
                                      style='TButton', command=self.open_boxPlot, width=20)  # Button for box plot
        self.boxPlot_btn.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.accuracy_lbl = tk.Label(self.result_frame, text="Accuracy: ")
        self.accuracy_lbl.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.elbow_btn = ttk.Button(self.result_frame, text='KNN Elbow Plot', command=self.calculate_and_plot_elbow,
                                    style='TButton', width=20)
        self.elbow_btn.grid(row=2, column=0, padx=5, pady=5, sticky='w')

        self.classification_report_btn = ttk.Button(self.result_frame, text='Classification Report', style='TButton',
                                                    width=20, command=self.show_classification_report)

        self.classification_report_btn.grid(row=2, column=1, padx=5, pady=5, sticky='e')

        self.predictNew_btn = ttk.Button(self.result_frame, text='PredictNewSample', style='TButton',
                                         command=self.open_predict_new_sample_window,width=20)

        self.predictNew_btn.grid(row=3, column=0, padx=5, pady=5, sticky='w')

        # create a back button in the plot_frame
        self.back_btn = ttk.Button(self.result_frame, text='Back to Main <---', width=20
                                   ,command=self.go_back, style='TButton')
        self.back_btn.grid(row=3, column=1, columnspan=2, padx=10, pady=5, sticky='e')

    def go_back(self):
        """
        Closes the current window and returns to the main form.
        """
        self.root.destroy()  # Close the classification window
        self.parent.deiconify()  # Show the main form again

    def create_table_frame(self):
        """
        Creates a frame for displaying the data table.
        """

        self.table_frame = tk.Frame(self.root, width=500, height=200) # Create a frame for the table
        self.table_frame.grid(row=0, column=2, columnspan=2,rowspan= 2, padx=5, pady=5, sticky='nsew')
        self.table_frame.grid_propagate(False) # Prevent the frame from resizing
        self.table_frame.grid_rowconfigure(0, weight=1) # Configure row weight
        self.table_frame.grid_columnconfigure(0, weight=1) # Configure column weight


    def on_dataset_selected(self, event):
        """
        Handles the event when a dataset is selected from the combobox.
        Updates the useless columns entry and displays the dataset image.
        """
        #Task 1:
        dataset_name = self.dataset_entry.get() # Get the selected dataset name
        base_path = self.dataSet_path.get(dataset_name)
        if base_path is None:
            # Handle missing path gracefully
            msg.showerror("Error", f'No path found for {dataset_name}')
            return
        # Construct the image path
        image_path = base_path + 'image.jpg'
        try:
            img = Image.open(image_path)
            img = img.resize((200, 150), Image.LANCZOS)
            self.photo_img = ImageTk.PhotoImage(img)
            self.lbl_photo.config(image=self.photo_img)
            self.lbl_photo.image = self.photo_img
        except Exception as e:
            msg.showerror("Error", f"Failed to load image:\n{e}")

        # Task2 :
        # Update useless columns based on the selected dataset
        if dataset_name == 'Chance_Bankruptcy':
            self.uselessCols_var.set('')
        elif dataset_name == 'WIFI_Signal_Strength':
            self.uselessCols_var.set('')

        elif dataset_name == "Predict_Acceptability_Car" :
            self.uselessCols_var.set('')

        else:
            self.uselessCols_var.set('')

        # Task3:
        # Update target column based on the selected dataset
        if dataset_name == 'Chance_Bankruptcy':
            self.target_column_var.set('Class')
        elif dataset_name == 'WIFI_Signal_Strength':
            self.target_column_var.set('Room')
        elif dataset_name == "Predict_Acceptability_Car" :
            self.target_column_var.set('class')

        else:
            self.target_column_var.set('')

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
            # Preprocess the loaded dataset
            self.df_preProcessed, self.numeric_cols = preProcessing_classificationKNN(
                self.df,
                self.uselessCols_var.get().split(','),
                dataset_name,
                outlier_threshold=float(self.outlierTr_var.get()),
                scaler=self.scaling_var.get(),
                main_form=self,
                target_col=self.target_column_var.get()
                )
            #print(self.df_preProcessed.head().to_string()) # Print the first few rows of the preprocessed DataFrame
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

    def calculate_and_plot_elbow(self):
        """
        Calculates the test error rates for KNN with k from 2 to 15,
        stores them in self.distortions, and plots the elbow curve.
        """
        if self.df_preProcessed.empty:
            msg.showwarning("Warning", "Please load and preprocess the dataset first.")
            return
        target_col = self.target_column_var.get()
        if not target_col or target_col not in self.df_preProcessed.columns:
            msg.showerror("Error", "Target column is not set or invalid.")
            return
        X = self.df_preProcessed.drop(columns=[target_col]).values
        y = self.df_preProcessed[target_col].astype('int').values

        try:
            test_size = float(self.test_size_var.get())
            if not (0 < test_size < 1):
                raise ValueError
        except ValueError:
            msg.showerror("Error", "Test size must be a float between 0 and 1.")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.distortions = []
        for k in range(2, 16):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            error_rate = 1 - accuracy_score(y_test, y_pred)
            self.distortions.append(error_rate)
        self.add_status("Elbow method calculation completed.")
        # Now plot the elbow curve in a new window
        plot_window = tk.Toplevel(self.root)
        plot_window.title('KNN Elbow Method - Error Rate vs K')
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(range(2, 16), self.distortions, marker='o')
        ax.set_title('KNN Elbow Method')
        ax.set_xlabel('Number of Neighbors (k)')
        ax.set_ylabel('Test Error Rate')
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def on_window_close():
            fig.clf()
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_window_close)


    def calc_classificationKNN(self):
        """
        Trains and tests the KNN classification model on the preprocessed data,
        saves the trained model using joblib, and updates the accuracy label.
        """
        # Check if dataset is loaded and preprocessed
        if self.df_preProcessed.empty:
            msg.showwarning("Warning", "Please load and preprocess the dataset first.")
            return
        # Get target column name
        target_col = self.target_column_var.get()
        if not target_col or target_col not in self.df_preProcessed.columns:
            msg.showerror("Error", "Target column is not set or invalid.")
            return
        # print(self.df_preProcessed[self.target_column_var.get()].unique())
        # Prepare features and target
        X = self.df_preProcessed.drop(columns=[target_col])
        # y = self.df_preProcessed[target_col]
        y = self.df_preProcessed[target_col].astype('int')  # convert target to discrete integer labels

        # Save feature names before converting to numpy array
        self.feature_names = X.columns

        # # Convert to numpy arrays if needed
        # X = X.values
        # y = y.values

        # Get test size from input
        try:
            test_size = float(self.test_size_var.get())
            if not (0 < test_size < 1):
                raise ValueError
        except ValueError:
            msg.showerror("Error", "Test size must be a float between 0 and 1.")
            return
        # Get number of neighbors
        try:
            n_neighbors = int(self.n_neighbors_var.get())
            if n_neighbors <= 0:
                raise ValueError
        except ValueError:
            msg.showerror("Error", "N_neighbors must be a positive integer.")
            return

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        # Initialize and train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)


        # Predict on test set
        y_pred = knn.predict(X_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Update accuracy label in the GUI
        self.accuracy_lbl.config(text=f"Accuracy: {accuracy:.4f}")

        # Save the model using joblib
        dataset_name = self.dataset_entry.get()
        model_dir = self.dataSet_path.get(dataset_name)
        if model_dir is None:
            msg.showerror("Error", f"No path found for dataset '{dataset_name}' to save the model.")
            return

        # After evaluation and saving the model trained on train split, train final model on all data:
        # Save the model trained on train split
        model_path_split = os.path.join(model_dir, "knn_model_train_test.joblib")
        joblib.dump(knn, model_path_split)
        self.add_status(f"KNN model trained on train/test split saved to {model_path_split}")
        # Train final model on all data
        knn_final = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_final.fit(X, y)
        # Save the final model trained on all data
        model_path_final = os.path.join(model_dir, "knn_model_final_all_data.joblib")
        joblib.dump(knn_final, model_path_final)
        self.add_status(f"Final model trained on all data saved to: {model_path_final}")
        # Optionally load and verify final model
        loaded_final_model = joblib.load( model_path_final)
        final_score = loaded_final_model.score(X, y)
        self.add_status(f"Final model score on all data: {final_score:.4f}")
        # Store final model for prediction use
        self.model = knn_final

    def show_classification_report(self):
        """
        Generates and displays the classification report for the test set.
        """
        if not hasattr(self, 'model'):
            msg.showwarning("Warning", "Please train the model first by clicking 'Calculate KNN'.")
            return
        target_col = self.target_column_var.get()
        if not target_col or target_col not in self.df_preProcessed.columns:
            msg.showerror("Error", "Target column is not set or invalid.")
            return
        # Prepare features and target
        X = self.df_preProcessed.drop(columns=[target_col])
        # y = self.df_preProcessed[target_col]
        y = self.df_preProcessed[target_col].astype('int')  # convert target to discrete integer labels
        try:
            test_size = float(self.test_size_var.get())
            if not (0 < test_size < 1):
                raise ValueError
        except ValueError:
            msg.showerror("Error", "Test size must be a float between 0 and 1.")
            return
        # Split data into train and test sets (same as used in training)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        # Predict on test set
        y_pred = self.model.predict(X_test)
        # Generate classification report
        report = classification_report(y_test, y_pred)
        # Create a new window to display the report
        report_window = tk.Toplevel(self.root)
        report_window.title("Classification Report")
        # Create a Text widget with vertical scrollbar
        text_area = tk.Text(report_window, wrap='word', font=('Courier', 10))
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(report_window, orient='vertical', command=text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill='y')
        text_area.configure(yscrollcommand=scrollbar.set)
        # Insert the report text
        text_area.insert(tk.END, report)
        text_area.config(state=tk.DISABLED)  # Make read-only
        # Optional: set a minimum size for the window
        report_window.geometry("600x400")

    def open_predict_new_sample_window(self):
        """
        Opens a new window with sliders for each numeric feature to input a new sample,
        and a button to predict the class using the final trained model.
        """
        if not hasattr(self, 'model') or not hasattr(self, 'feature_names'):
            msg.showwarning("Warning", "Please train and save the model first by clicking 'Calculate KNN'.")
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
            label = tk.Label(self.predict_window, text=feature, bg='#73C2FB')
            label.grid(row=row, column=0, padx=5, pady=5, sticky='w')
            # Create a scale (slider) widget
            # Use resolution=0.01 for float precision, adjust as needed
            slider = tk.Scale(self.predict_window, from_=min_val, to=max_val, orient=tk.HORIZONTAL, resolution=0.01,
                              length=300, bg='#73C2FB')
            slider.set(mean_val)  # Set default to mean value
            slider.grid(row=row, column=1, padx=5, pady=5, sticky='ew')
            self.feature_sliders[feature] = slider
            row += 1
        # Predict button
        predict_btn = ttk.Button(self.predict_window, text="Predict", command=self.predict_new_sample)
        predict_btn.grid(row=row, column=0, columnspan=2, pady=10)
        # Label to show prediction result
        self.prediction_result_label = tk.Label(self.predict_window, text="", font=('Arial', 12, 'bold'), bg='#73C2FB')
        self.prediction_result_label.grid(row=row + 1, column=0, columnspan=2, pady=10)

    def predict_new_sample(self):
        """
        Collects values from sliders, creates a sample, predicts the class using the final model,
        and displays the prediction result.
        """
        if not hasattr(self, 'model'):
            msg.showwarning("Warning", "Model is not loaded.")
            return
        # Collect feature values from

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



    def run(self):
        """
                Starts the main event loop of the Tkinter application.
        """
        self.root.mainloop()

    def on_close(self):
        plt.close("all")
        self.root.destroy()





# app_classification = MainFormKNN()
# app_classification.run()


