# Breast-Cancer-Detection
Breast cancer prediction using a Decision Tree Classifier. The model is optimized with RandomizedSearchCV and handles class imbalance via SMOTE. A Flask web app allows users to input clinical data for real-time predictions, displaying results with probability scores. Feature selection and scaling were applied.

Overview
This project predicts whether a patient has breast cancer using clinical data. The model uses a Decision Tree Classifier, optimized with RandomizedSearchCV and handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique). The web application, built with Flask, allows users to input clinical data and receive real-time predictions along with probability scores.

Features
Data Preprocessing:

Dropped unnecessary columns and encoded the target variable (diagnosis).

Selected top 5 features using SelectKBest for feature selection.

Scaled features using StandardScaler.

Model Training:

Used Decision Tree Classifier optimized with RandomizedSearchCV to find the best hyperparameters, improving model accuracy.

Handled imbalanced data with SMOTE.

Achieved a high classification accuracy on test data.

Web Interface:

Developed a Flask-based web application where users can input clinical values (e.g., perimeter mean, concave points mean) and get a prediction with associated probability.

Installation & Setup
Prerequisites
Ensure you have the following dependencies installed:

Python 3.x

Flask

Pandas

NumPy

Scikit-learn

Imbalanced-learn (for SMOTE)
