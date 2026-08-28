# 🧠 ML-Tutorial-GUI - Interactive Machine Learning Tutorial

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20-green)](https://github.com)

An interactive desktop application for learning and implementing machine learning algorithms without the need for coding. This project is built using **Tkinter** for the graphical user interface (GUI) and powerful **Python** libraries such as Pandas, Scikit-learn, Matplotlib, and Seaborn.

---

## 📑 Table of Contents

- [Project Goal](#project-goal)
- [Key Features](#key-features)
- [Demo & Screenshots](#demo--screenshots)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [How to Use](#how-to-use)
- [Supported Datasets](#supported-datasets)
- [Project Structure](#project-structure)
- [Sample Output](#sample-output)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## 🎯 Project Goal

The main goal of this project is to provide an **educational platform** for machine learning concepts in a practical and interactive way. Users can experience the complete data science workflow - from data loading to result prediction - without writing a single line of code.

This project was developed as the final project for the **Machine Learning course in Data Science** at Sematec Institute.

---

## ✨ Key Features

### 🔄 Data Loading & Management
- Support for CSV files with data preview capability
- Display dataset information (rows, columns, data types)
- Show basic statistics of the dataset

### 🧹 Data Preprocessing
- Handle missing values (remove rows/columns)
- Remove outliers using IQR method
- Scale features (Standardization/Normalization)
- Encode categorical features (Label Encoding, One-Hot Encoding)
- View data before and after preprocessing

### 🤖 Machine Learning Algorithms

| Task | Algorithms |
|------|------------|
| **Regression** | Decision Tree Regressor, Linear Regression |
| **Classification** | Decision Tree Classifier, K-Nearest Neighbors (KNN) |
| **Clustering** | K-Means Clustering |

### 📊 Advanced Visualization
- Correlation Matrix (Heatmap)
- Box Plot for outlier detection
- Residual Plot for regression
- Tree Plot for Decision Trees
- Feature Importance Plot
- Elbow Plot for K-Means
- Cluster Distribution Chart
- Scatter plots for clustering visualization

### 📈 Model Evaluation

| Task | Metrics |
|------|---------|
| Regression | R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE) |
| Classification | Accuracy Score |
| Clustering | Inertia, Silhouette Score |

### 💾 Save & Load Models
- Save trained models in `.pkl` format using Joblib
- Load previously saved models for reuse
- No need to retrain every time

### 🎛️ Interactive Prediction
- Predict new samples using sliders
- Adjust feature values in real-time
- Instant prediction results
- User-friendly interface

---

## 🖥️ Demo & Screenshots

*(Please add your screenshots here)*

| Main Window | Data Preprocessing | Model Training |
|-------------|-------------------|----------------|
| ![Main Window](screenshots/main_window.png) | ![Preprocessing](screenshots/preprocessing.png) | ![Training](screenshots/training.png) |

| Visualization | Prediction | Results |
|---------------|------------|---------|
| ![Visualization](screenshots/visualization.png) | ![Prediction](screenshots/prediction.png) | ![Results](screenshots/results.png) |

📹 **Watch Demo Video**: [ProjectAI_SolmazKarimi.mp4](ProjectAI_SolmazKarimi.mp4)

---

## 📦 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/FinalProjectAI_SolmazKarimi.git
cd FinalProjectAI_SolmazKarimi
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3.Run the Application
