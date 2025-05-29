# Chronic-Kidney-Disease Prediction 

![Alt text describing the image](images/ckd.png)
# Project Summary: Comparison of Machine Learning Models for Chronic Kidney Disease Prediction
This project uses machine learning to predict the risk level of chronic kidney disease in patients, based on clinical data.

# 🎯 Target Variable (Risk Levels)
The prediction target is a multi-class classification with the following categories:
No_Disease, Low_Risk, Moderate_Risk, High_Risk, Severe_Disease

# 🧭 Project Step-by-Step Workflow :
# Data collection:
The dataset was sourced from [source: Kaggle , name:amanik000/kidney-disease-dataset].
It contains clinical features such as blood pressure, albumin, blood urea, creatinine, hemoglobin, and more.
# Exploratory Data Analysis (EDA):
- Understand the structure and summary of the data.
- Check for missing values, imbalanced classes, and data types.

# Data processing:
- Encode categorical features : Label Encoding or OrdinalEncoding depending on the case.
- Feature scaling : Used StandardScaler to normalize numerical columns

# Define Features and Target:
- X: All clinical features (after preprocessing).
- y: The risk level of kidney disease (No_Disease, Low_Risk, etc.)

# Train-Test Split:
Split data into training and testing sets

# Model selection:
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
Support Vector Machines (SVM)
K-Nearest Neighbors (KNN)
Naive Bayes

# Models evalutaion: 
Used metrics suitable for multi-class classification:

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Cross-validation for robustness
