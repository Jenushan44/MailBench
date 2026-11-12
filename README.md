# MailBench

**Comparing Machine Learning Algorithms for Email Spam Detection**

## Overview

MailBench is a machine learning project that aims to identify the best performing algorithm for classifying emails as spam or legitimate. 

Naive Bayes seems like a good choice for spam detection but that doesn't necessarily mean that it is the best. I wanted to test other algorithms as well to see hwo they compare and udnerstand where each one performs well or struggles. 

## Algorithms

I will be comparing six machine learning algorithms: 
- Naive Bayes
- Logistic Regression
- K-Nearest Neighbours (KNN)
- Support Vector Machine (SVM)
- Decision Tree 
- Artificial Neural Network (ANN)

## Evaluation

I will train and test each algorithm using the same dataset so I can compare their performance fairly. 

I will be looking at: 
- Accuracy: How many emails the model classifies correctly. 
- Precision: How many emails flagged as spam are actually spam. 
- Recall: How many of the actual spam emails the model catches. 
- F1-score: A balance between the precision and recall.
- Training time: How long each model takes to train.
- ROC-AUC: How well the model distinguishes spam form legitimate emails across classification thresholds.
- Cross-validation accuracy: Average validation accuracy across multiple training splits. 

I will also look at cases where the models make incorrect predictions since a spam filter that marks legitimate emails as spam could be a problem even if its overall accuracy is high. 

## Progress

- **Logistic Regression:** Completed
- **Naive Bayes:** Next
- **KNN, SVM, Decision Tree, ANN:** Planned

## Dataset

This project uses the Phishing Email Dataset published by Naser Abdullah Alam on Kaggle. 

The dataset contains 82486 emails collected from six sources including Enron, Ling, CEAS, Nazario, Nigerian Fraud and SpamAssassin. I will be using the combined phishing_email.csv file which contains email text and labels for classification. 

The dataset is licensed under CC BY-SA 4.0. 

### Dataset Citation

Alam, N.A. (2024). _Phishing Email Dataset_. Kaggle. https://doi.org/10.34740/kaggle/ds/5074342

## Model Results

### Logistic Regression

Logistic Regression was used as the first baseline model for MailBench. It learns relationships between numerical email features and their labels to predict whether an email is spam or legitimate. 

#### Implementation

- Removed 408 duplicate emails before training
- Used an 80/20 stratified train/test split
- Applied TF-IDF to convert email text into numerical features
- Trained Logistic Regression with max_iter=1000
- Performed 5-fold stratified cross-validation using a Pipeline to prevent preprocessing data leakage

#### Results

| Metric | Score |
|---|---:|
| Accuracy | 98.50% |
| Precision | 98.09% |
| Recall | 99.05% |
| F1 Score | 98.57% |
| ROC-AUC | 0.9986 |
| Mean CV Accuracy | 98.40% |
| CV Standard Deviation | 0.093 percentage points |

The model correctly identified 8,488 spam/phishing emails and missed 81. It also incorrectly classified 165 legitimate emails as spam. 

The cross-validation results were close to the held-out test accuracy which suggests consistent performance across the evaluated splits. 
