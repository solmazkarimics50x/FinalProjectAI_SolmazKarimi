# 🧠 ML-Tutorial-GUI

**An Interactive Machine Learning Desktop Application**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)


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
├── mainform.py          # Main application
├── requirements.txt     # Dependencies
├── README.md           # Documentation
├── Images/             # Screenshots
└── .gitignore
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
- GitHub: your-username
- Email: your-email@example.com
  
