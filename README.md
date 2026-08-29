# 🧠 ML-Tutorial-GUI

**An Interactive Machine Learning Desktop Application**

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9+-orange.svg)](https://scikit-learn.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows_10-green.svg)](https://www.microsoft.com/windows/)

---

## 📖 About

This is an interactive desktop application for learning and implementing machine learning algorithms **without coding**. Built with **Tkinter** (GUI) and Python libraries including Pandas, Scikit-learn, Matplotlib, and Seaborn.

**Developed by:** Solmaz Karimi  
**Instructor:** Mr. Vahid Ghorbani  
**Institute:** Sematec Institute

---

## ✨ Features

- 🔄 **Load & Preview** CSV datasets
- 🧹 **Preprocess** data (clean, scale, encode)
- 🤖 **Train Models**:
  - Regression (Linear, Decision Tree)
  - Classification (KNN, Decision Tree)
  - Clustering (K-Means)
- 📊 **Visualize** results (correlation, box plots, tree plots, elbow plots, etc.)
- 📈 **Evaluate** models (R², MAE, MSE, Accuracy)
- 💾 **Save/Load** trained models (.pkl)
- 🎛️ **Predict** new samples using sliders

---

## 🛠️ Quick Start

```bash
# 1. Clone
git clone https://github.com/your-username/FinalProjectAI_SolmazKarimi.git
cd FinalProjectAI_SolmazKarimi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python mainform.py
```
## 📊 Modules

### 🧩1. Clustering (K-Means)
- **Datasets:** User_Knowledge, Liver_Disorders, Dow_Jones, Wholesale_Customers, Travel_Reviews, Electric_Consumption
- **Visualizations:** Correlation Matrix, Box Plot, Elbow Plot, Cluster Distribution

### 📈2. Linear Regression
- **Datasets:** Age_Abalone, PriceHouse, SeminalQuality, FuelEfficiencyCar, TotalDemandOrders, PollutionLevelCity
- **Visualizations:** Correlation Matrix, Residuals Plot, Coefficients Plot

### 🏷️3. Classification (KNN)
- **Datasets:** WIFI_Signal_Strength, Predict_Acceptability_Car, Chance_Bankruptcy
- **Visualizations:** Correlation Matrix, Elbow Plot, Classification Report

### 🌳4. Decision Tree
- **Datasets:** Age_Abalone (Regression), Chance_Bankruptcy (Classification)
- **Visualizations:** Tree Plot, Feature Importance Plot

## 🎥 Video Tutorial
[![ML-Tutorial-GUI](https://img.youtube.com/vi/ukZxjXDnPFI/0.jpg)](https://youtu.be/ukZxjXDnPFI)
  
## 📁 Project Structure
```bash
FinalProjectAI_SolmazKarimi/
│
├── Images/                                    # All UI images/screenshots
│   └── ImagesMainForm/                        # Main form images
│
├── Project_ClassificationKNN_SolmazKarimi/    # KNN Classification module
│   ├── classification_knn_data/              # Dataset for KNN
│   ├── mainFormClassificationKNN.py          # Main UI for KNN
│   └── preProcessing_classificationKNN.py    # Preprocessing for KNN
│
├── Project_Clustering_SolmazKarimi/           # Clustering module
│   ├── clustering_data/                      # Dataset for clustering
│   ├── images_clustering/                    # Clustering result images
│   ├── Asli_First_mainFormClustering.py     # (Optional/backup)
│   ├── mainFormClustering.py                 # Main UI for clustering
│   └── preProcessing.py                      # Preprocessing for clustering
│
├── Project_DecisionTree_SolmazKarimi/         # Decision Tree module
│   ├── decisionTree_data/                    # Dataset for decision tree
│   ├── mainFormDecisionTree.py               # Main UI for decision tree
│   └── preProcessing_decisionTree.py         # Preprocessing for decision tree
│
├── Project_LinearRegression_SolmazKarimi/     # Linear Regression module
│   ├── regression_data/                      # Dataset for regression
│   ├── mainFormLinearRegression.py           # Main UI for regression
│   └── preProcessing_regression.py           # Preprocessing for regression
│
├── .gitignore                                 # Git ignore file
├── mainform.py                                # Main application entry point
├── requirements.txt                           # Project dependencies
├── README.md                                  # Project documentation
├── FinalProjectAI_SolmazKarimi.zip            # Archived project (optional)
└── ProjectAI_SolmazKarimi.mp4                 # Demo video (optional)
```


## 📋 Sample Output
```bash
# Decision Tree (Age_Abalone)
R² Score: 0.4118

# KNN Classification (Chance_Bankruptcy)
Accuracy: 1.0000 (100%)

# K-Means (Travel_Reviews)
Clusters: 3 | Cluster 0: 136, Cluster 1: 106, Cluster 2: 232
```
## 🙏 Acknowledgments

-  Mr. Vahid Ghorbani – Professor
- Sematec Institute – Course organizer
- Data Science Dojo – Datasets

## 📧 Contact
- GitHub: solmazkarimics50x
- Email: en.co.s.karimi@gmail.com
  
