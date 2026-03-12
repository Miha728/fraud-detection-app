# 💳 Fraud Detection App

Demo application that simulates credit card fraud detection using a machine learning model and additional behavioral risk rules.

## Features

* Machine learning fraud probability prediction
* Risk signals (online transaction, new device, transaction frequency)
* Adjustable fraud alert threshold
* Interactive web interface

## Demo Interface

![App Screenshot](assets/app_screenshot.png)

## How it works

The application combines two signals:

**1. Machine Learning Model**

A logistic regression model trained on anonymized credit card transaction features.

**2. Behavioral Risk Rules**

Additional fraud indicators:

* transactions after 10 PM
* online payments
* new device used
* many transactions in last 24 hours
* foreign country transaction

These signals are combined into a **final fraud score**.

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas

## Run locally

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
streamlit run app/app.py
```


## Model Comparison

This project includes a comparison between multiple machine learning models trained on the **Credit Card Fraud Detection dataset (Kaggle)**.

Models evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

The comparison is generated using the script:

python src/compare_models.py --data data/creditcard.csv

The results are saved in:

model/model_comparison.json  
model/model_comparison.csv

The Streamlit application includes a **Model Comparison dashboard** where these results are visualized.

