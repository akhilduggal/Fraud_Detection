# Fraud Detection Project

## Overview
This project is about **detecting fraudulent transactions** in financial data. Fraudulent transactions are suspicious or unauthorized payments. The goal is to **identify fraud early**, helping banks and businesses save money and protect their customers.  

We use computer programs (models) to learn patterns from past transactions and predict whether a new transaction is **fraudulent** or **normal**.

---

## What the Project Does
1. **Explores the data**:  
   - Counts normal vs fraudulent transactions.  
   - Checks relationships between different factors like transaction amount, time, or user behavior.  

2. **Prepares the data**:  
   - Fills missing values.  
   - Converts categories into numbers.  
   - Scales values for model processing.  

3. **Trains machine learning models**:  
   - **Logistic Regression**: A simple model to predict fraud.  
   - **Random Forest**: Uses multiple decision trees to improve predictions.  
   - **Calibration**: Ensures predicted probabilities are reliable.  

4. **Evaluates the models**:  
   - Measures accuracy and ROC-AUC,which tells how well the model separates fraud from normal transactions.  
   - Produces reports showing detailed performance.  

5. **Generates plots** :  
   - **Fraud vs Normal distribution** – shows number of fraudulent transactions.  
   - **Correlation heatmap** – shows how features relate to fraud.  
   - **Calibration plots** – shows if predicted probabilities are realistic.  
   - **Top features correlated with fraud** – highlights most influential factors.  

---

## Why This Project is Useful
- Helps **detect fraud early** in financial transactions.  
- Reduces **financial losses** and **risks to customers**.  
- Shows which factors are **most important for fraud detection**.  

---
