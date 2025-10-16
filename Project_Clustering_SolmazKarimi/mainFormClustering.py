import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as msg
from PIL import Image, ImageTk
import webbrowser
import pathlib


from .preProcessing import preProcessing
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MainFormClustering:
    def __init__(self, parent):
        """
        Initializes the main application window and sets up the GUI components.
        """
        self.parent = parent  # Store reference to parent window
        # self.root = tk.Tk() # Create the main window
        self.root = tk.Toplevel(parent)
        self.root.title('clustering...')# Set the window title
        self.root.geometry('1090x400')# Set the window size #'575x600'
        self.root.resizable(False, False)# Disable window resizing
        # Center the window on the screen
        x = int(self.root.winfo_screenwidth() / 2 - 1090 / 2)
        y = int(self.root.winfo_screenheight() / 2 - 400 / 2)
        self.root.geometry(f'+{x}+{y}')
        self.root.protocol("WM_DELETE_WINDOW", self.on_close) # Handle window close event
        self.style = ttk.Style()# Create a style for the GUI
        # self.style.configure('TButton', font=('Arial', 10))
        # Configure button styles
        self.style.map('TButton',
                       background=[('active', '#4CAF50'), ('!active', 'SystemButtonFace')],
                       foreground=[('active', 'black'), ('!active', 'black')]
                       )
        self.df = pd.DataFrame()# Initialize an empty DataFrame for data storage
        # Define paths for datasets
        self.dataSet_path = {
            'User_Knowledge': './Project_Clustering_SolmazKarimi/clustering_data/User Knowledge Modeling/',
            'Liver_Disorders': './Project_Clustering_SolmazKarimi/clustering_data/Liver Disorders/',
            'Dow_Jones': './Project_Clustering_SolmazKarimi/clustering_data/Dow Jones Index/',
            'Wholesale_Customers': './Project_Clustering_SolmazKarimi/clustering_data/Wholesale Customers/',
            'Travel_Reviews': './Project_Clustering_SolmazKarimi/clustering_data/Travel Reviews/',
            'Electric_Consumption': './Project_Clustering_SolmazKarimi/clustering_data/Individual household electric power consumption/'
        }
        # Define filenames for datasets
        self.dataSet_file = {
            'User_Knowledge': 'Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.csv',
            'Liver_Disorders': 'bupa.csv',
            'Dow_Jones': 'dow_jones_index.csv',
            'Wholesale_Customers': 'Wholesale customers data.csv',
            'Travel_Reviews': 'tripadvisor_review.csv',
            'Electric_Consumption': 'household_power_consumption.csv'
        }
        # Create various frames for the GUI
        self.create_input_frame()
        self.create_image_frame()
        self.create_progress_frame()
        self.create_results_frame()
        self.create_table_frame()
        self.distortions = [] # Initialize a list to store KMeans distortions

    def go_back(self):
        """
        Closes the current window and returns to the main form.
        """
        self.root.destroy()  # Close the clustering window
        self.parent.deiconify()  # Show the main form again

    def run(self):
        """
                Starts the main event loop of the Tkinter application.
        """
        self.root.mainloop()

    def create_input_frame(self):
        """
        Creates the input frame for selecting datasets and parameters.
        """
        self.input_frame = tk.Frame(self.root,bg='#73C2FB')
        self.input_frame.grid(row=0, column=1, padx=(10,0), pady=(10,0), sticky='nsew')
        # Label and combobox for dataset selection
        self.combo_label = tk.Label(self.input_frame, text='Select Dataset:',bg='#73C2FB')
        self.combo_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.dataset_entry = tk.StringVar()# Variable to hold the selected dataset
        self.dataset_combo = ttk.Combobox(self.input_frame, textvariable=self.dataset_entry, state='readonly', width=30)
        self.dataset_combo['values'] = list(self.dataSet_file.keys())# Populate combobox with dataset names
        self.dataset_combo.current = ''  # Set default selection
        self.dataset_combo.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_selected)  # Bind selection event

        # Labels and entries for K-Means parameters
        self.KMean_range_label = tk.Label(self.input_frame, text='K-Means Range:',bg='#73C2FB')
        self.KMean_range_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.KMean_range_l = tk.StringVar(value=2) # Lower range for K
        self.KMean_range_h = tk.StringVar(value=12) # Upper range for K
        self.KMean_k = tk.StringVar(value=3) # Desired number of clusters
        self.KMean_range_l_entry = ttk.Entry(self.input_frame, textvariable=self.KMean_range_l, width=3)
        self.KMean_range_l_entry.grid(row=1, column=1, padx=(5,1), pady=5, sticky='w')
        self.KMean_between_label = tk.Label(self.input_frame, text=' : ',bg='#73C2FB')
        self.KMean_between_label.grid(row=1, column=1, padx=(29,1), pady=5, sticky='w')
        self.KMean_range_h_entry = ttk.Entry(self.input_frame, textvariable=self.KMean_range_h, width=3)
        self.KMean_range_h_entry.grid(row=1, column=1, padx=(45,1), pady=5, sticky='w')
        self.KMean_k_label = tk.Label(self.input_frame, text='Desired K :',bg='#73C2FB')
        self.KMean_k_label.grid(row=1, column=1, padx=(100, 1), pady=5, sticky='w')
        self.KMean_k_entry = ttk.Entry(self.input_frame, textvariable=self.KMean_k, width=3)
        self.KMean_k_entry.grid(row=1, column=1, padx=(170, 1), pady=5, sticky='w')

        # Label and entry for useless columns
        self.uselessCols_var = tk.StringVar() # Variable to hold useless columns
        self.uselessCols_label = tk.Label(self.input_frame, text='Useless Columns:',bg='#73C2FB')
        self.uselessCols_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.uselessCols_entry = ttk.Entry(self.input_frame, textvariable=self.uselessCols_var ,width=30)
        self.uselessCols_entry.grid(row=2, column=1, padx=5, pady=5, sticky='ew')

        # Radio buttons for scaling options
        self.scaling_var = tk.StringVar(value='zscore') # Default scaling method
        self.scaleing_frame = tk.Frame(self.input_frame, relief='groove',bg='#73C2FB')
        self.scaleing_frame.grid(row=3, column=0, columnspan=2, sticky='nsew')
        self.zscore_radio = tk.Radiobutton(self.scaleing_frame, text='Zscore Scaling', variable=self.scaling_var, value='zscore',bg='#73C2FB')
        self.minmax_radio = tk.Radiobutton(self.scaleing_frame, text='MinMax Scaling', variable=self.scaling_var, value='minmax',bg='#73C2FB')
        self.none_radio = tk.Radiobutton(self.scaleing_frame, text='None', variable=self.scaling_var, value='None',bg='#73C2FB')
        self.zscore_radio.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.minmax_radio.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.none_radio.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        # Label and entry for outlier threshold
        self.outlier_lbl = tk.Label(self.input_frame, text='Outlier Threshold:',bg='#73C2FB')
        self.outlier_lbl.grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.outlierTr_var = tk.StringVar(value=1.5) # Default outlier threshold
        self.outlierTr_entry = ttk.Entry(self.input_frame, textvariable=self.outlierTr_var, width=5)
        self.outlierTr_entry.grid(row=4, column=1, padx=5, pady=5, sticky='w')

        # Frame for input buttons
        self.input_btns_frame = tk.LabelFrame(self.input_frame,bg='#73C2FB', text = "InputButtons...")
        self.input_btns_frame.grid(row=5, column=0, columnspan=2, sticky='nsew')
        self.load_btn = ttk.Button(self.input_btns_frame, text='Load Dataset', width=20,
                                   command=lambda: self.load_dataset(self.dataset_entry.get()), style='TButton')
        self.load_btn.grid(row=0, column=0, padx=(25,10), pady=5, sticky='ew')
        self.calc_Kmeans_btn = ttk.Button(self.input_btns_frame, text='Calculate KMeans', width=20,
                                          command=self.calc_Kmeans, style='TButton')
        self.calc_Kmeans_btn.grid(row=0, column=1, padx=10, pady=5, sticky='ew')


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
        # self.KMean_k.set('3')
        # Update useless columns based on the selected dataset
        if dataset_name== 'User_Knowledge':
            self.uselessCols_var.set('UNS')
        elif dataset_name == 'Liver_Disorders':
            self.uselessCols_var.set('train/test selector')
        elif dataset_name == 'Dow_Jones':
            self.uselessCols_var.set('open,high,low,'
                                     'percent_change_volume_over_last_wk,previous_weeks_volume,'
                                     'next_weeks_open,next_weeks_close,percent_change_next_weeks_price')
        elif dataset_name == 'Wholesale_Customers':
            self.uselessCols_var.set('')
        elif dataset_name == 'Travel_Reviews':
            self.uselessCols_var.set('User ID')
        elif dataset_name == 'Electric_Consumption':
            self.uselessCols_var.set('Date,Time')
        else:
            self.uselessCols_var.set('')

    def create_image_frame(self):
        """
        Creates a frame for displaying the dataset image and a button to read the README file.
        """
        self.image_frame = tk.Frame(self.root,bg='#73C2FB', bd=2) # Create a frame for the image
        self.image_frame.grid(row=0, column=0, padx=10, pady=(10,0), sticky='snew')
        self.blank_img = ImageTk.PhotoImage(Image.new('RGB', (200, 150), color='white')) # Create a blank image
        # print("Created blank_img:", self.blank_img)
        self.lbl_photo = tk.Label(self.image_frame, image=self.blank_img)  # Label to display the image
        self.lbl_photo.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.read_md = ttk.Button(self.image_frame, text='Read MarkDown', command=self.open_md, style='TButton') # Button to open README
        self.read_md.grid(row=1, column=0, padx=10,pady = (5,5), sticky='s')

    def create_results_frame(self):
        """
        Creates a frame for displaying result buttons for various plots.
        """
        self.result_frame = tk.LabelFrame(self.root, relief='sunken',text ="Result...", bd=2) # Create a frame for results
        self.result_frame.grid(row=1, column=1, padx=(5,0), pady=(5,0), sticky='nsew')
        # self.pairplot_btn = ttk.Button(self.result_frame, text='Pair Plot Chart', command=self.open_pairPlot, style='TButton', width= 20)# Button for pair plot
        # self.pairplot_btn.grid(row=0, column=0, padx=10, pady=5, sticky='ns')
        self.corrPlot_btn = ttk.Button(self.result_frame, text='Correlation Matrix', command=self.open_corrPlot, style='TButton', width= 20) # Button for correlation matrix
        self.corrPlot_btn.grid(row=0, column=0, padx=10, pady=5, sticky='ns')
        self.boxPlot_btn = ttk.Button(self.result_frame, text='Box Plot Chart', command=self.open_boxPlot, style='TButton', width= 20) # Button for box plot
        self.boxPlot_btn.grid(row=0, column=1, padx=10, pady=5, sticky='ns')
        self.kmeans_btn = ttk.Button(self.result_frame, text='Kmeans elbow Result', command=self.open_inertiaPlot, style='TButton', width= 20)  # Button for KMeans elbow plot
        self.kmeans_btn.grid(row=1, column=0, padx=10, pady=5, sticky='ns')
        self.cluster_btn = ttk.Button(self.result_frame , text = "Cluster Distribution" , command = self.open_cluster_distribution, style= 'TButton', width= 20) # Button for Cluster Distribution plot
        self.cluster_btn.grid(row = 1 , column = 1 , padx = 10 , pady = 5 , sticky= 'ns')
        # create a back button in the result_frame
        self.back_btn = ttk.Button(self.result_frame, text='Back to Main <--- ', width=20,
                                   command=self.go_back, style='TButton')
        self.back_btn.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky='ew')

    def create_progress_frame(self):
        """
        Creates a frame for displaying progress messages.
        """
        self.progress_frame = tk.LabelFrame(self.root, relief='sunken', text ="Progress ...")  # Create a frame for progress
        self.progress_frame.grid(row=1, column=0, padx=10, pady=(5,0), sticky='nsew')
        self.status_text = tk.Text(self.progress_frame, height=10, width=33, wrap=tk.WORD, font=('Arial', 8),fg="green")  # Text widget for status messages
        self.status_text.grid(row=0, column=0, padx=0, pady=0, sticky='nsew')

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
        self.table_frame.grid(row=0, column=2, columnspan=2,rowspan= 2, padx=10, pady=5, sticky='nsew')
        self.table_frame.grid_propagate(False) # Prevent the frame from resizing
        self.table_frame.grid_rowconfigure(0, weight=1) # Configure row weight
        self.table_frame.grid_columnconfigure(0, weight=1) # Configure column weight

    def open_md(self):
        """
        Opens the README file for the selected dataset in a web browser.
        """
        dataset_name = self.dataset_entry.get() # Get the selected dataset name
        local_path = self.dataSet_path.get(dataset_name) + 'README.html'  # Construct the path to the README file
        absolute_path = pathlib.Path(local_path).resolve()  # Get the absolute path
        file_url = absolute_path.as_uri() # Convert to a file URL
        webbrowser.open(file_url) # Open the URL in the default web browser

    def load_dataset(self, dataset_name):
        """
        Loads the selected dataset into a DataFrame and preprocesses it.
        Parameters:
        - dataset_name: str - The name of the dataset to load.
        """
        if not dataset_name:
            msg.showwarning("Warning", "Please select a dataset.") # Show warning if no dataset is selected
            return

        self.distortions = [] # Reset distortions list
        self.status_text.delete(1.0, tk.END)  # Clear previous status messages
        if dataset_name:
            file_path = self.dataSet_path.get(dataset_name) + self.dataSet_file.get(dataset_name)  # Construct the file path
            if dataset_name == 'Electric_Consumption':
                self.df = pd.read_csv(file_path, sep=';', na_values='?') # Load dataset with specific separator
            else:
                self.df = pd.read_csv(file_path, sep=',')  # Load dataset with default separator

            self.add_status(f"-Dataset csv loaded successfully.") # Update status
            # Preprocess the loaded dataset
            self.df_preProcessed, self.numeric_cols = preProcessing(self.df,
                                    self.uselessCols_var.get().split(','),
                                    dataset_name,
                                    outlier_threshold=float(self.outlierTr_var.get()),
                                    scaler=self.scaling_var.get(),
                                    main_form=self)
            # print(self.df_preProcessed.head().to_string()) # Print the first few rows of the preprocessed DataFrame
            self.show_table()  # Display the data in the table

    # def open_pairPlot(self):
    #     """
    #     Opens a new window to display a pair plot of the preprocessed data.
    #     """
    #     if self.df.empty:
    #         msg.showwarning("Warning", "Please load a dataset first.") # Show warning if no dataset is loaded
    #         return
    #
    #     plot_window = tk.Toplevel(self.root) # Create a new window for the plot
    #     plot_window.title("Pair Plot Chart") # Set the window title
    #
    #     data = self.df_preProcessed[self.numeric_cols].copy() # Copy the numeric columns for plotting
    #     if 'cluster_' in self.df_preProcessed.columns:
    #         data['cluster_'] = self.df_preProcessed['cluster_'] # Add cluster labels if they exist
    #     g = sns.PairGrid(data, hue='cluster_' if 'cluster_' in data.columns else None, palette='Set1') # Create a PairGrid
    #     g.map_upper(sns.scatterplot, s=10, alpha=0.3) # Scatter plot for upper triangle
    #     g.map_diag(sns.kdeplot, lw=1, fill=True, alpha=0.5) # KDE plot for diagonal
    #     g.map_lower(sns.kdeplot, fill=True, alpha=0.5)#, cmap='Blues'  # KDE plot for lower triangle
    #     fig = g.figure  # Get the figure from the PairGrid
    #     # Embed the plot in the Tkinter window
    #     canvas = FigureCanvasTkAgg(fig, master=plot_window)
    #     canvas.draw()
    #     canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    #
    #     def on_window_close():
    #         plt.close(fig)
    #         plot_window.destroy()
    #
    #     plot_window.protocol("WM_DELETE_WINDOW", on_window_close)

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

    def open_inertiaPlot(self):
        if not self.distortions:
            msg.showwarning("Warning", "Please Calculate KMeans first.")
            return

        plot_window = tk.Toplevel(self.root)
        plot_window.title('KMeans Inertia Plot')

        plt.figure(figsize=(8, 6))
        plt.plot(range(int(self.KMean_range_l.get()), int(self.KMean_range_h.get()) + 1), self.distortions, marker='o')
        plt.title('KMeans Inertia Plot')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True)

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def on_window_close():
            plt.close('all')
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_window_close)

    def open_cluster_distribution(self):
        """
        Opens a new window to display a bar chart of cluster distribution with counts on bars.
        """
        if 'cluster_' not in self.df_preProcessed.columns:
            msg.showwarning("Warning", "Please Calculate KMeans first to generate clusters.")
            return

        plot_window = tk.Toplevel(self.root)
        plot_window.title('Cluster Distribution Chart')

        # Count the number of samples in each cluster
        cluster_counts = self.df_preProcessed['cluster_'].value_counts().sort_index()

        # Create the bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(cluster_counts.index.astype(str), cluster_counts.values,
                       color=sns.color_palette("Set2", len(cluster_counts)))

        # Add count labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom')

        plt.title('Cluster Distribution')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Samples')
        plt.grid(axis='y', alpha=0.3)

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def on_window_close():
            plt.close('all')
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_window_close)

    def on_close(self):
        import matplotlib.pyplot as plt
        plt.close('all')
        self.root.destroy()



    def calc_Kmeans(self):
        if self.df_preProcessed.empty:
            msg.showwarning("Warning", "Please load a dataset first.")
            return

        from sklearn.cluster import KMeans
        if 'cluster_' in self.df_preProcessed.columns:
            self.df_preProcessed = self.df_preProcessed.drop('cluster_', axis=1)

        self.distortions = []
        start_k = int(self.KMean_range_l.get())
        end_k = int(self.KMean_range_h.get())
        total_iterations = end_k - start_k + 1
        desired_k = int(self.KMean_k.get())

        self.add_status("-Starting KMeans calculations...")
        for k in range(start_k, end_k + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(self.df_preProcessed)
            self.distortions.append(kmeans.inertia_)
            if k == desired_k:
                self.df_preProcessed['cluster_'] = kmeans.labels_

        self.add_status("-Clustering with KMeans completed.")
        self.show_table()
        for index in range(0,desired_k):
            self.add_status(f"sample in cluster {index} : {self.df_preProcessed['cluster_'].value_counts().sort_index()[index]}")

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

# clustering_app = MainFormClustering()
# clustering_app.run()