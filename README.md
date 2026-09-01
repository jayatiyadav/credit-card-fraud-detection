# 💳 FraudShield – Credit Card Fraud Detection

An AI-powered Machine Learning and Data Analytics system for detecting potentially fraudulent credit card transactions.

FraudShield combines data preprocessing, exploratory data analysis, SMOTE-based class balancing, machine learning classification, and an interactive Streamlit dashboard to identify suspicious transactions.

---

## 🚀 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a very small percentage of total transactions.

This project aims to:

- Analyze credit card transaction data
- Identify patterns associated with fraudulent transactions
- Handle severe class imbalance using SMOTE
- Train and compare multiple Machine Learning models
- Select the best-performing model
- Provide an interactive Streamlit interface for fraud detection
- Allow users to upload transaction CSV files and download predictions

---

## ✨ Features

### 📊 Data Analytics
- Transaction distribution analysis
- Fraud vs legitimate transaction analysis
- Transaction amount analysis
- Correlation analysis
- Fraud transaction statistics
- Visual EDA charts

### 🤖 Machine Learning
- Logistic Regression
- Random Forest Classifier
- SMOTE for handling class imbalance
- Confusion Matrix
- Classification Report
- ROC-AUC evaluation
- Trained model saved using Joblib

### 🖥️ Interactive Dashboard
- Dashboard overview
- Data Analytics section
- Fraud Detection section
- CSV file upload
- Fraud prediction
- Legitimate transaction prediction
- Prediction summary
- Downloadable prediction results

---

## 📂 Dataset

The project uses the Credit Card Fraud Detection dataset containing:

- **284,807 transactions**
- **31 columns**
- **492 fraudulent transactions**
- **284,315 legitimate transactions**

The original dataset is highly imbalanced:

| Class | Transactions | Percentage |
|-------|-------------:|-----------:|
| Legitimate | 284,315 | 99.83% |
| Fraud | 492 | 0.17% |

After removing duplicate records:

- Dataset size: **283,726**
- Legitimate transactions: **283,253**
- Fraudulent transactions: **473**

> The dataset is not included in this repository because of its size.

---

## 🔍 Exploratory Data Analysis

The dataset was analyzed to understand transaction patterns and identify relationships between features and fraud.

### Important Findings

The features showing strong negative correlation with fraud included:

- V17
- V14
- V12
- V10
- V16
- V3
- V7

The fraud transaction amount statistics were:

- Mean: **122.21**
- Median: **9.25**
- Maximum: **2125.87**

The analysis showed that fraudulent transactions are extremely rare compared with legitimate transactions, making class imbalance a major challenge.

---

## ⚖️ Handling Class Imbalance

Because fraudulent transactions represent only around **0.17%** of the dataset, training directly on the original data can cause a model to become biased toward legitimate transactions.

To address this problem, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training data.

### Before SMOTE

- Legitimate: 226,602
- Fraud: 378

### After SMOTE

- Legitimate: 226,602
- Fraud: 226,602

SMOTE was applied only to the training data to avoid data leakage.

---

## 🤖 Machine Learning Models

Two classification algorithms were trained and evaluated.

### 1. Logistic Regression

ROC-AUC:

**0.963**

Fraud performance:

- Precision: 0.05
- Recall: 0.87
- F1-score: 0.10

### 2. Random Forest

ROC-AUC:

**0.966**

Fraud performance:

- Precision: 0.92
- Recall: 0.74
- F1-score: 0.82

---

## 🏆 Model Selection

Random Forest was selected as the final model because it provided a better balance between precision and recall while achieving the highest ROC-AUC score.

### Final Model Performance

| Metric | Random Forest |
|--------|--------------:|
| Accuracy | ~99.95% |
| Fraud Precision | 0.92 |
| Fraud Recall | 0.74 |
| Fraud F1-score | 0.82 |
| ROC-AUC | 0.9664 |
                

Actual Legitimate    56645      6
Actual Fraud            25     70
