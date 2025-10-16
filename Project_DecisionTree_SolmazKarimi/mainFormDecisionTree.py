import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as msg
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image,ImageTk
import pathlib
import webbrowser
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
from sklearn.tree import plot_tree



from .preProcessing_decisionTree import preProcessing_decisionTree



class MainFormDecisionTree:
    def __init__(self, parent):
        """
        Initializes the main application window and sets up the GUI components.
        """
        self.parent = parent  # Store reference to parent window
        # self.root = tk.Tk() # Create the main window
        self.root = tk.Toplevel(parent)
        self.root.title("DecisionTree...") # Set the window title
        self.root.resizable(0,0)# Disable window resizing
        self.root.geometry("1120x430") # Set the window size#("580x430")
        # Center the window on the screen
        x = int(self.root.winfo_screenwidth() / 2  - 1120 / 2 )
        y = int(self.root.winfo_screenheight() / 2 - 430 / 2 )
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
            'Age_Abalone': './Project_DecisionTree_SolmazKarimi/decisionTree_data/Find out the age of Abalone/',
            "Chance_Bankruptcy": './Project_DecisionTree_SolmazKarimi/decisionTree_data/Estimate the chance of bankruptcy from qualitative parameters by experts/'
            # 'SeminalQuality': './regression_data/PredictTheSeminalQualityOfAnIndividual/',
            # 'FuelEfficiencyCar': './regression_data/PredictTheFuelEfficiencyOfACar/',
            # 'TotalDemandOrders': './regression_data/PredictTheTotalNumberOfDemandOFOrOrders/',
            # 'PollutionLevelCity' : './regression_data/ForecastPollutionLevelOfACity/'
        }
        # Define filenames for datasets
        self.dataSet_file = {
            # 'User_Knowledge': 'Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.csv',
            'Age_Abalone': 'preprocessed_abalone_data.csv',
            "Chance_Bankruptcy": 'Qualitative_Bankruptcy.csv'
            # 'SeminalQuality': 'fertility_Diagnosis.csv',
            # 'FuelEfficiencyCar': 'auto-mpg.csv',
            # 'TotalDemandOrders': 'Daily_Demand_Forecasting_TotalOrders.csv',
            # 'PollutionLevelCity': 'PRSA_data.csv'

        }
        # Create various frames for the GUI

        self.create_image_frame()
        self.create_input_frame()
        self.create_progress_frame()
        self.create_results_frame()
        self.create_table_frame()

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

    def open_md(self):
        """
        Opens the README file for the selected dataset in a web browser.
        """
        dataset_name = self.dataset_entry.get() # Get the selected dataset name
        local_path = self.dataSet_path.get(dataset_name) + "README.html"  # Construct the path to the README file
        absolute_path = pathlib.Path(local_path).resolve() # Get the absolute path
        file_url = absolute_path.as_uri()  # Convert to a file URL
        webbrowser.open(file_url)  # Open the URL in the default web browser

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
        self.outlier_testsize_frame.grid(row = 4 , column =0 , columnspan= 6 , sticky= "nsew")
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

        ## Label and entry for Max Depth
        self.maxDepth_lbl = tk.Label(self.outlier_testsize_frame, text='Max Depth:',bg='#73C2FB')
        self.maxDepth_lbl.grid(row=4, column=4, padx=5, pady=5, sticky='w')
        self.maxDepth_var = tk.StringVar(value=5) # Default outlier threshold
        self.maxDepth_entry = ttk.Entry(self.outlier_testsize_frame, textvariable=self.maxDepth_var, width=5)
        self.maxDepth_entry.grid(row=4, column=5, padx=5, pady=5, sticky='e')

        # Frame for input buttons
        self.input_btns_frame = tk.LabelFrame(self.input_frame,bg='#73C2FB', text = "InputButton...")
        self.input_btns_frame.grid(row=5, column=0, columnspan=6, sticky='nsew')
        self.load_btn = ttk.Button(self.input_btns_frame, text='Load Dataset', width=25,
                                    command=lambda: self.load_dataset(self.dataset_entry.get()),style='TButton')
        self.load_btn.grid(row=0, column=0, padx=(25,10), pady=5, sticky='w')

        self.calc_DecisionTree_btn = ttk.Button(self.input_btns_frame, text='Calculate DecisionTree', width=25,
                                                command=self.calc_DecisionTree, style='TButton')
        self.calc_DecisionTree_btn.grid(row=0, column=2, padx=10, pady=5, sticky='e')

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

        elif dataset_name == 'Chance_Bankruptcy':
            self.uselessCols_var.set('')

        # elif dataset_name == 'PriceHouse' :
        #     self.uselessCols_var.set( 'X1 transaction date')
        # elif dataset_name == 'SeminalQuality':
        #     self.uselessCols_var.set('Output_N')
        # elif dataset_name == 'FuelEfficiencyCar':
        #     self.uselessCols_var.set('car_name')


        else :
            self.uselessCols_var.set('')


        #Task3:
        #Update target column based on the selected dataset
        if dataset_name == 'Age_Abalone' :
             self.target_column_var.set ('Rings')
        elif dataset_name == 'Chance_Bankruptcy':
            self.target_column_var.set('Class')
        # elif dataset_name == 'PriceHouse':
        #     self.target_column_var.set('Y house price of unit area')
        # elif dataset_name == 'SeminalQuality':
        #     self.target_column_var.set('Output_O')
        # elif dataset_name == 'FuelEfficiencyCar':
        #     self.target_column_var.set('mpg')
        # elif dataset_name == 'TotalDemandOrders':
        #     self.target_column_var.set('Target (Total orders)')
        # elif dataset_name == 'PollutionLevelCity' :
        #     self.target_column_var.set('pm2.5')


        else :
            self.target_column_var.set('')

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

    def create_results_frame(self):
        """
        Creates a frame for displaying result buttons for various plots.
        """
        self.result_frame = tk.LabelFrame(self.root, relief='sunken',text="Result..." , bd=2) # Create a frame for results
        self.result_frame.grid(row=1, column=1, padx=(5,0), pady=(5,0), sticky='nsew')
        self.corrPlot_btn = ttk.Button(self.result_frame, text='Correlation Matrix', style='TButton', command=self.open_corrPlot, width= 20)# Button for correlation matrix
        self.corrPlot_btn.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.boxPlot_btn = ttk.Button(self.result_frame, text='Box Plot Chart', style='TButton', command=self.open_boxPlot, width= 20) ## Button for box plot
        self.boxPlot_btn.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.accuracy_lbl = tk.Label(self.result_frame , text = "Accuracy: ")
        self.accuracy_lbl.grid(row = 1 , column= 0 , padx = 5 , pady= 5 , sticky = "ew")


        self.plot_frame = tk.Frame(self.result_frame, relief="flat")
        self.plot_frame.grid(row=2, column=0, columnspan=3, sticky="w")

        self.tree_plot_btn = ttk.Button(self.plot_frame, text='Tree Plot', style='TButton',command=self.show_tree_plot, width=15)
        self.tree_plot_btn.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        self.ImportanceFeatures_plot_btn = ttk.Button(self.plot_frame, text='Importance Features Plot', style='TButton',command=self.show_importance_features_plot, width=25)
        self.ImportanceFeatures_plot_btn.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        self.predictNew_btn = ttk.Button(self.plot_frame, text='PredictNewSample', style='TButton', width=17,command=self.open_predict_new_sample_window)

        self.predictNew_btn.grid(row=0, column=3, padx=5, pady=5, sticky='w')



        # create a back button in the plot_frame
        self.back_btn = ttk.Button(self.plot_frame, text='Back to Main <---', width=30
                                  , command=self.go_back, style='TButton')
        self.back_btn.grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky='ns')

    def go_back(self):
        """
        Closes the current window and returns to the main form.
        """
        self.root.destroy()  # Close the decisiontree window
        self.parent.deiconify()  # Show the main form again

    def create_table_frame(self):
        """
        Creates a frame for displaying the data table.
        """

        self.table_frame = tk.Frame(self.root, width=500, height=200) # Create a frame for the table
        self.table_frame.grid(row=0, column=2,rowspan= 2, columnspan=2, padx=5, pady=5, sticky='nsew')
        self.table_frame.grid_propagate(False) # Prevent the frame from resizing
        self.table_frame.grid_rowconfigure(0, weight=1) # Configure row weight
        self.table_frame.grid_columnconfigure(0, weight=1) # Configure column weight

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
            self.df_preProcessed, self.numeric_cols = preProcessing_decisionTree(self.df,
                                    self.uselessCols_var.get().split(','),
                                    dataset_name,
                                    outlier_threshold=float(self.outlierTr_var.get()),
                                    scaler=self.scaling_var.get(),
                                    main_form=self,
                                    target_col=self.target_column_var.get()
                                     )
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

    def show_tree_plot(self):
        dataset_name = self.dataset_entry.get()
        if dataset_name not in ['Age_Abalone', 'Chance_Bankruptcy']:
            msg.showinfo("Info", "Tree plot is only available for Age_Abalone and Chance_Bankruptcy datasets.")
            return
        model_file = None
        if dataset_name == 'Age_Abalone':
            model_file = os.path.join(self.dataSet_path[dataset_name], 'final_decision_tree_regressor_model.joblib')
        elif dataset_name == 'Chance_Bankruptcy':
            model_file = os.path.join(self.dataSet_path[dataset_name], 'final_decision_tree_classifier_model.joblib')
        if not model_file or not os.path.exists(model_file):
            msg.showwarning("Warning", "Please calculate the DecisionTree first to generate and save the model.")
            return
        # Load the model
        model = joblib.load(model_file)
        # Create a new window for the plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Decision Tree Plot - {dataset_name}")
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, filled=True,
                  feature_names=self.df_preProcessed.drop(columns=[self.target_column_var.get()]).columns, ax=ax,
                  rounded=True, fontsize=8)
        plt.tight_layout()
        # Embed the plot in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def on_close():
            plt.close(fig)
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_close)

    def show_importance_features_plot(self):
        dataset_name = self.dataset_entry.get()
        if dataset_name not in ['Age_Abalone', 'Chance_Bankruptcy']:
            msg.showinfo("Info",
                         "Importance Features Plot is only available for Age_Abalone and Chance_Bankruptcy datasets.")
            return
        model_file = None
        if dataset_name == 'Age_Abalone':
            model_file = os.path.join(self.dataSet_path[dataset_name], 'final_decision_tree_regressor_model.joblib')
        elif dataset_name == 'Chance_Bankruptcy':
            model_file = os.path.join(self.dataSet_path[dataset_name], 'final_decision_tree_classifier_model.joblib')
        if not model_file or not os.path.exists(model_file):
            msg.showwarning("Warning", "Please calculate the DecisionTree first to generate and save the model.")
            return
        # Load the model
        model = joblib.load(model_file)
        # Get feature names
        feature_names = self.df_preProcessed.drop(columns=[self.target_column_var.get()]).columns
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Importance Features Plot - {dataset_name}")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Get feature importances
        importances = model.feature_importances_
        # Sort descending
        sorted_idx = importances.argsort()[::-1]
        sorted_features = feature_names[sorted_idx]
        sorted_importances = importances[sorted_idx]

        bars = ax.bar(sorted_features, sorted_importances, color='skyblue')
        ax.set_title(f"Feature Importances ({dataset_name})")
        ax.set_ylabel("Importance")
        ax.set_xticklabels(sorted_features, rotation=45, ha='right')
        # Add value labels on bars
        for bar, val in zip(bars, sorted_importances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        # Embed the plot in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def on_close():
            plt.close(fig)
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_close)


    def calc_DecisionTree(self):
        dataset_name = self.dataset_entry.get()
        if self.df_preProcessed.empty:
            msg.showwarning("Warning", "Please load and preprocess the dataset first.")
            return
        target_col = self.target_column_var.get()
        if target_col == '':
            msg.showwarning("Warning", "Target column is not specified.")
            return
        # Prepare features and target
        X = self.df_preProcessed.drop(columns=[target_col])
        y = self.df_preProcessed[target_col]
        # Split data
        try:
            test_size = float(self.test_size_var.get())
        except ValueError:
            test_size = 0.3  # default
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        max_depth = None
        try:
            max_depth = int(self.maxDepth_var.get())
        except ValueError:
            max_depth = None


        if dataset_name == 'Age_Abalone':
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            r2 = model.score(X_test, y_test)
            self.accuracy_lbl.config(text=f"R^2 Score for DecisionTreeRegressor : {r2:.4f}")
            self.add_status(f"DecisionTreeRegressor trained. R^2 Score: {r2:.4f}")
            # Train on all data (regression)
            final_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            final_model.fit(X, y)
            self.add_status("DecisionTreeRegressor trained on entire dataset.")
            # Dump model
            model_path = os.path.join(self.dataSet_path[dataset_name], 'final_decision_tree_regressor_model.joblib')
            joblib.dump(final_model, model_path)
            self.add_status(f"Final regressor model saved to {model_path}")


        elif dataset_name == 'Chance_Bankruptcy':
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            self.accuracy_lbl.config(text=f"Accuracy for DecisionTreeClassifier: {acc:.4f}")
            self.add_status(f"DecisionTreeClassifier trained. Accuracy: {acc:.4f}")
            # Train on all data (classification)
            final_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            final_model.fit(X, y)
            self.add_status("DecisionTreeClassifier trained on entire dataset.")
            # Dump model
            model_path = os.path.join(self.dataSet_path[dataset_name], 'final_decision_tree_classifier_model.joblib')
            joblib.dump(final_model, model_path)
            self.add_status(f"Final classifier model saved to {model_path}")
        else:
            msg.showinfo("Info",
                         "DecisionTree calculation is only implemented for Age_Abalone and Chance_Bankruptcy datasets.")

        # Then bind this method to the button in create_input_frame:
        self.calc_DecisionTree_btn.config(command=self.calc_DecisionTree)

    def open_predict_new_sample_window(self):
        dataset_name = self.dataset_entry.get()
        if dataset_name not in ['Age_Abalone', 'Chance_Bankruptcy']:
            msg.showinfo("Info",
                         "Prediction for new samples is only available for Age_Abalone and Chance_Bankruptcy datasets.")
            return
        model_file = None
        if dataset_name == 'Age_Abalone':
            model_file = os.path.join(self.dataSet_path[dataset_name], 'final_decision_tree_regressor_model.joblib')
        elif dataset_name == 'Chance_Bankruptcy':
            model_file = os.path.join(self.dataSet_path[dataset_name], 'final_decision_tree_classifier_model.joblib')
        if not model_file or not os.path.exists(model_file):
            msg.showwarning("Warning", "Please calculate the DecisionTree first to generate and save the model.")
            return
        # Load the model
        self.predict_model = joblib.load(model_file)
        # Create new window
        self.pred_window = tk.Toplevel(self.root)
        self.pred_window.title(f"Predict New Sample - {dataset_name}")
        self.pred_window.geometry("500x650")
        # Center the window
        x = int(self.pred_window.winfo_screenwidth() / 2 - 500 / 2)
        y = int(self.pred_window.winfo_screenheight() / 2 - 650/ 2)
        self.pred_window.geometry(f'+{x}+{y}')
        # Frame for sliders
        sliders_frame = tk.Frame(self.pred_window)
        sliders_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Get feature columns (exclude target)
        target_col = self.target_column_var.get()
        feature_cols = list(self.df_preProcessed.columns)
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        self.slider_vars = {}
        self.slider_widgets = {}
        # For each feature, create a slider with min and max from preprocessed data
        for i, feature in enumerate(feature_cols):
            min_val = float(self.df_preProcessed[feature].min())
            max_val = float(self.df_preProcessed[feature].max())
            # For sliders, use integer scale if values are integers, else float scale with resolution
            is_int = all(float(x).is_integer() for x in self.df_preProcessed[feature])
            var = tk.DoubleVar() if not is_int else tk.IntVar()
            var.set(min_val)
            label = tk.Label(sliders_frame, text=f"{feature} ({min_val:.2f} - {max_val:.2f})")
            label.grid(row=i, column=0, sticky='w', pady=5)
            slider = tk.Scale(sliders_frame, from_=min_val, to=max_val, orient='horizontal',
                              resolution=0.01 if not is_int else 1, variable=var, length=250)
            slider.grid(row=i, column=1, sticky='ew', pady=5)
            self.slider_vars[feature] = var
            self.slider_widgets[feature] = slider
        # Frame for buttons
        btn_frame = tk.Frame(self.pred_window)
        btn_frame.pack(pady=10)
        # Predict button
        predict_btn = ttk.Button(btn_frame, text="Predict", command=self.predict_new_sample)
        predict_btn.grid(row=0, column=0, padx=5)
        # Show Accuracy button
        accuracy_btn = ttk.Button(btn_frame, text="Show Accuracy", command=self.show_model_accuracy)
        accuracy_btn.grid(row=0, column=1, padx=5)
        # TreePlot button
        treeplot_btn = ttk.Button(btn_frame, text="TreePlot", command=self.show_tree_plot)
        treeplot_btn.grid(row=0, column=2, padx=5)
        # Label to show prediction result
        self.prediction_result_lbl = tk.Label(self.pred_window, text="", font=('Arial', 12), fg='blue')
        self.prediction_result_lbl.pack(pady=10)

    def predict_new_sample(self):
        # Collect input values from sliders
        input_data = {}
        for feature, var in self.slider_vars.items():
            input_data[feature] = var.get()
        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        # Predict using the loaded model
        try:
            if isinstance(self.predict_model, DecisionTreeRegressor):
                pred = self.predict_model.predict(input_df)[0]
                self.prediction_result_lbl.config(text=f"Predicted {self.target_column_var.get()}: {pred:.4f}")
            elif isinstance(self.predict_model, DecisionTreeClassifier):
                pred_class = self.predict_model.predict(input_df)[0]
                pred_proba = self.predict_model.predict_proba(input_df)[0]
                # Show predicted class and probability of that class
                class_index = list(self.predict_model.classes_).index(pred_class)
                prob = pred_proba[class_index]
                self.prediction_result_lbl.config(text=f"Predicted Class: {pred_class}\nProbability: {prob:.4f}")
            else:
                self.prediction_result_lbl.config(text="Model type not supported for prediction.")
        except Exception as e:
            self.prediction_result_lbl.config(text=f"Prediction error: {e}")

    def show_model_accuracy(self):
        dataset_name = self.dataset_entry.get()
        if dataset_name not in ['Age_Abalone', 'Chance_Bankruptcy']:
            msg.showinfo("Info", "Accuracy info is only available for Age_Abalone and Chance_Bankruptcy datasets.")
            return
        # Load dataset and split again to calculate accuracy
        target_col = self.target_column_var.get()
        X = self.df_preProcessed.drop(columns=[target_col])
        y = self.df_preProcessed[target_col]
        try:
            test_size = float(self.test_size_var.get())
        except ValueError:
            test_size = 0.3
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        max_depth = None
        try:
            max_depth = int(self.maxDepth_var.get())
        except ValueError:
            max_depth = None
        if dataset_name == 'Age_Abalone':
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            r2 = model.score(X_test, y_test)
            msg.showinfo("R^2 Score", f"R^2 Score for DecisionTreeRegressor: {r2:.4f}")
        elif dataset_name == 'Chance_Bankruptcy':
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            msg.showinfo("Accuracy", f"Accuracy for DecisionTreeClassifier: {acc:.4f}")











    def show_table(self):
        # Clear existing table if any
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        # Create container frame with fixed size
        container = ttk.Frame(self.table_frame, width=450, height=200)
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



# app_DecisionTree = MainFormDecisionTree()
# app_DecisionTree.run()