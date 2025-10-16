import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as msg
import matplotlib.pyplot as plt
from PIL import Image,ImageTk

from Project_Clustering_SolmazKarimi.mainFormClustering import MainFormClustering
from Project_LinearRegression_SolmazKarimi.mainFormLinearRegression import MainFormRegression
from Project_ClassificationKNN_SolmazKarimi.mainFormClassificationKNN import MainFormKNN
from Project_DecisionTree_SolmazKarimi.mainFormDecisionTree import MainFormDecisionTree




class MainForm:
    def __init__(self):
        self.root = tk.Tk() # Create the main window
        self.root.title("MainForm ...")# Set the window title
        self.root.geometry('700x500')# Set the window size
        self.root.resizable(False, False)  # Disable window resizing
        # Center the window on the screen
        x = int(self.root.winfo_screenwidth() / 2 - 700 / 2 )
        y = int(self.root.winfo_screenheight() / 2 - 500 / 2 )
        self.root.geometry(f'+{x}+{y}')
        # self.root.protocol("WM_DELETE_WINDOW", self.on_close) # Handle window close event
        self.root.iconbitmap("Images/ImagesMainForm/mainForm.ico")

        self.style = ttk.Style() # Create a style for the GUI
        # Configure button styles
        self.style.map("TButton",
                       background = [ ("active" , "#4CAF50"),("!active" ,"SystemButtonFace" )],
                       foreground = [("active" , "darkblue"),("!active" ,"black" )] )

        self.root.grid_rowconfigure(0, minsize=80)  # Reserve space for the top bar
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Create various frames for the GUI
        self.create_border_frame()
        self.create_image_frame()
        self.create_input_frame()

        self.clustering_clicked = False  # Track toggle state
        self.regression_clicked = False  # Track toggle state
        self.classification_clicked = False  # Track toggle state
        self.decisionTree_clicked = False  # Track toggle state






    def run(self):
        self.root.mainloop()

    # Creates a frame for displaying the border
    def create_border_frame(self):
        self.border_frame = tk.Frame(self.root , bg = "blue", height= 50)
        self.border_frame.grid(row = 0 , column = 0 ,columnspan= 2,rowspan= 1, sticky= "nsew")

        self.border_lbl = tk.Label(self.border_frame ,text="Welcome to the Machine Learning Final Project", bg="blue", fg="white", font=("Arial", 16,"bold"))
        self.border_lbl.grid(row = 0 , column = 0 , sticky= "ns" , padx = 5 , pady = 5  )

        # Designer name label
        self.designer_lbl = tk.Label(
            self.border_frame,
            text=" Designed by Solmaz Karimi ",
            bg="blue",
            fg="black",
            font=("Arial", 12, "bold" )
        )
        self.designer_lbl.grid(row=1, column=0, sticky="nsew", padx=5)
        # Supervisor name label
        self.Supervisor_lbl = tk.Label(
            self.border_frame,
            text="Professor: Mr.Vahid Ghorbani",
            bg="blue",
            fg="black",
            font=("Arial", 12 ,"bold" )
        )
        self.Supervisor_lbl.grid(row=2, column=0, sticky="nsew", padx=5)

        # make label fill the frame
        self.border_frame.grid_columnconfigure(0, weight=1)
        self.border_frame.grid_rowconfigure(0, weight=1)
        self.border_frame.grid_rowconfigure(1, weight=1)
        self.border_frame.grid_rowconfigure(2, weight=1)



    # Creates a frame for displaying the Model image
    def create_image_frame(self):
        self.image_frame = tk.Frame(self.root , relief= "groove", bd =2) # Create a frame for the image
        self.image_frame.grid(row = 1, column = 1,padx = 20, pady= (70,70) , sticky= "e")

        self.backing_img = ImageTk.PhotoImage(Image.new("RGB",(350,250),color = "white")) # Create a blank image

        self.photo_lbl = tk.Label(self.image_frame ,image= self.backing_img )
        self.photo_lbl.grid(row = 0 , column = 0 , padx = 5 , pady = 5 , sticky= "nsew")
        Image_path = "Images/ImagesMainForm/machine-learning.jpg"
        try:
            img = Image.open(Image_path)
            img = img.resize((350, 250), Image.LANCZOS)
            self.photo_img = ImageTk.PhotoImage(image=img)
            self.photo_lbl.config(image=self.photo_img)
            self.photo_lbl.image = self.photo_img  # keep reference
        except Exception as e:
            msg.showerror("Error", f' Failed to load image:\n {e} ')

    # Frame for input buttons
    def create_input_frame(self):
        self.input_frame = tk.LabelFrame(self.root, relief='sunken')
        self.input_frame.grid(row=1, column=0, padx=(20,20), pady=(100,90), sticky='sn')

        self.input_btns_frame = tk.Frame(self.input_frame, relief='sunken')
        self.input_btns_frame.grid(row=0, column=0, sticky='sn', pady= 10 , padx= 10)
        self.clustering_btn = ttk.Button(self.input_btns_frame, text='Clustering(KMeans)',command=self.on_clustering_btn_click, width=30, style='TButton')
        self.clustering_btn.grid(row=0, column=0, padx=10, pady=5, ipady=7, sticky='ew')
        self.linearRegression_btn = ttk.Button(self.input_btns_frame, text='LinearRegression',command=self.on_linearRegression_btn_click, width=30, style='TButton')
        self.linearRegression_btn.grid(row=1, column=0, padx=10, pady=5, ipady=7, sticky='ew')
        self.classification_btn = ttk.Button(self.input_btns_frame, text='Classification(KNN)',command=self.on_classification_btn_click, width=30,style='TButton')
        self.classification_btn.grid(row=2, column=0, padx=10, pady=5, ipady=7, sticky='ew')
        self.decisionTree_btn = ttk.Button(self.input_btns_frame, text='DecisionTree',command=self.on_decisionTree_btn_click, width=30,style='TButton')
        self.decisionTree_btn.grid(row=3, column=0, padx=10, pady=5, ipady=7, sticky='ew')


        # make label fill the frame
        self.input_btns_frame.grid_columnconfigure(0, weight=1)
        self.input_btns_frame.grid_rowconfigure(0, weight=1)
        self.input_btns_frame.grid_rowconfigure(1, weight=1)
        self.input_btns_frame.grid_rowconfigure(2, weight=1)
        self.input_btns_frame.grid_rowconfigure(3, weight=1)

    def on_clustering_btn_click(self):
        if not self.clustering_clicked :
            # First click: show the desired image in the image frame
            Image_path ="Images/ImagesMainForm/Clustering_image.jpg"
            try :
                img = Image.open(Image_path)
                img = img.resize((350, 250) , Image.LANCZOS)
                self.photo_img = ImageTk.PhotoImage(image =img)
                self.photo_lbl.config(image=self.photo_img)
                self.photo_lbl.image = self.photo_img  # keep reference
                self.clustering_clicked = True
            except Exception as e :
                msg.showerror("Error", f' Failed to load image:\n {e} ')


        else:
            # Second click: open the MainFormClustering window
            Image_path = "Images/ImagesMainForm/machine-learning.jpg"
            try:
                img = Image.open(Image_path)
                img = img.resize((350, 250), Image.LANCZOS)
                self.photo_img = ImageTk.PhotoImage(image=img)
                self.photo_lbl.config(image=self.photo_img)
                self.photo_lbl.image = self.photo_img  # keep reference
            except Exception as e:
                msg.showerror("Error", f' Failed to load image:\n {e} ')

            self.root.withdraw()  # hide main form
            # Pass self.root as parent to the clustering form
            clustering_app = MainFormClustering(self.root)
            clustering_app.run()
            # self.root.deiconify()  # Show main form again after clustering window closes
            self.clustering_clicked = False

    def on_linearRegression_btn_click(self):

        if not self.regression_clicked:
            # First click: show the desired image in the image frame
            Image_path = "Images/ImagesMainForm/LinearRegression_image.jpg"
            try:
                img = Image.open(Image_path)
                img = img.resize((350, 250), Image.LANCZOS)
                self.photo_img = ImageTk.PhotoImage(image=img)
                self.photo_lbl.config(image=self.photo_img)
                self.photo_lbl.image = self.photo_img  # keep reference
                self.regression_clicked = True
            except Exception as e:
                msg.showerror("Error", f' Failed to load image:\n {e} ')



        else:
            # Second click: open the MainFormRegression window
            Image_path = "Images/ImagesMainForm/machine-learning.jpg"
            try:
                img = Image.open(Image_path)
                img = img.resize((350, 250), Image.LANCZOS)
                self.photo_img = ImageTk.PhotoImage(image=img)
                self.photo_lbl.config(image=self.photo_img)
                self.photo_lbl.image = self.photo_img  # keep reference

            except Exception as e:
                msg.showerror("Error", f' Failed to load image:\n {e} ')
            self.root.withdraw()  # hide main form
            # Pass self.root as parent to the linearRegression form
            regression_app = MainFormRegression(self.root)
            regression_app.run()
            # self.root.deiconify()  # Show main form again after linearRegression window closes
            self.regression_clicked = False

    def on_classification_btn_click(self):

        if not self.classification_clicked:
            # First click: show the desired image in the image frame
            Image_path = "Images/ImagesMainForm/ClassificationKNN.jpg"
            try:
                img = Image.open(Image_path)
                img = img.resize((350, 250), Image.LANCZOS)
                self.photo_img = ImageTk.PhotoImage(image=img)
                self.photo_lbl.config(image=self.photo_img)
                self.photo_lbl.image = self.photo_img  # keep reference
                self.classification_clicked = True
            except Exception as e:
                msg.showerror("Error", f' Failed to load image:\n {e} ')



        else:
            # Second click: open the MainFormRegression window
            Image_path = "Images/ImagesMainForm/machine-learning.jpg"
            try:
                img = Image.open(Image_path)
                img = img.resize((350, 250), Image.LANCZOS)
                self.photo_img = ImageTk.PhotoImage(image=img)
                self.photo_lbl.config(image=self.photo_img)
                self.photo_lbl.image = self.photo_img  # keep reference

            except Exception as e:
                msg.showerror("Error", f' Failed to load image:\n {e} ')
            self.root.withdraw()  # hide main form
            # Pass self.root as parent to the ClassificationKNN form
            classification_app = MainFormKNN(self.root)
            classification_app.run()
            # self.root.deiconify()  # Show main form again after ClassificationKNN window closes
            self.classification_clicked = False

    def on_decisionTree_btn_click(self):

        if not self.decisionTree_clicked:
            # First click: show the desired image in the image frame
            Image_path = "Images/ImagesMainForm/Decision-Trees.jpg"
            try:
                img = Image.open(Image_path)
                img = img.resize((350, 250), Image.LANCZOS)
                self.photo_img = ImageTk.PhotoImage(image=img)
                self.photo_lbl.config(image=self.photo_img)
                self.photo_lbl.image = self.photo_img  # keep reference
                self.decisionTree_clicked = True
            except Exception as e:
                msg.showerror("Error", f' Failed to load image:\n {e} ')



        else:
            # Second click: open the mainformDecisionTree window
            Image_path = "Images/ImagesMainForm/machine-learning.jpg"
            try:
                img = Image.open(Image_path)
                img = img.resize((350, 250), Image.LANCZOS)
                self.photo_img = ImageTk.PhotoImage(image=img)
                self.photo_lbl.config(image=self.photo_img)
                self.photo_lbl.image = self.photo_img  # keep reference

            except Exception as e:
                msg.showerror("Error", f' Failed to load image:\n {e} ')
            self.root.withdraw()  # hide main form
            # Pass self.root as parent to the mainformDecisionTree form
            decisionTree_app = MainFormDecisionTree(self.root)
            decisionTree_app.run()
            # self.root.deiconify()  # Show main form again after mainformDecisionTree window closes
            self.decisionTree_clicked = False





if __name__ == "__main__":
    app = MainForm()
    app.run()