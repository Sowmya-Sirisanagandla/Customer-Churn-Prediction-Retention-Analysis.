Customer Churn Prediction & Retention Analysis
This project develops a machine learning model to identify customers at high risk of churn and enable proactive retention strategies. The goal is to minimize revenue loss by detecting potential churners early and supporting data-driven business decisions.

Project Overview:

Customer churn is a critical problem for subscription-based businesses. Losing customers directly affects revenue and growth.

This project aims to predict whether a customer will churn using the IBM Telco Customer Churn dataset and follows a complete machine learning pipeline from data cleaning to model evaluation.
Dataset Information:

Dataset: IBM Telco Customer Churn

Total rows: 7043

Cleaned rows: 7032

Features: 19

Target: Churn (Yes/No)

Problem Statement:

-> Given customer details such as demographics, services, contract type, and billing information, predict whether the customer will churn.

-> This is a binary classification problem:

-> 0 → No Churn

-> 1 → Churn

Data Preprocessing:
1. Data Loading

Loaded dataset using pandas.

2. Data Cleaning

Dropped customerID (no predictive value)

Converted TotalCharges to numeric using:
pd.to_numeric(..., errors='coerce')

Removed missing values using:
df.dropna()

3. Data Validation

Checked missing values using df.isnull().sum()

Verified no zero values in MonthlyCharges

Ensured no rows with tenure = 0 after cleaning

Feature Engineering:
-> Target Separation

X = input features

y = target column (Churn)

-> Target Encoding

Converted:
No → 0
Yes → 1

-> Feature Encoding

Applied One-Hot Encoding using:
pd.get_dummies(X, drop_first=True)

Ensured all features are numeric

Train-Validation-Test Split:

Data was split into:

Train: 70%

Validation: 15%

Test: 15%

Used:

random_state=42 for reproducibility

stratify=y to maintain class balance

-> Models Used:

This project follows a structured modeling approach, starting with a simple baseline and progressing to a more complex model to improve predictive performance.

1. Logistic Regression (Baseline Model)

Logistic Regression was used as the initial baseline model for customer churn prediction.
It is a widely used algorithm for binary classification problems and provides high interpretability.

Helps understand how individual features influence churn probability

Fast to train and easy to implement

Serves as a benchmark to evaluate more complex models

However, Logistic Regression assumes a linear relationship between features and the target variable, which may limit its performance on complex patterns in the data.

2. Random Forest Classifier

Random Forest, an ensemble learning method, was used to improve model performance by capturing non-linear relationships and feature interactions.

Combines multiple decision trees to improve robustness

Handles both numerical and categorical features effectively

Reduces overfitting compared to a single decision tree

Achieved better performance, especially in identifying churned customers (higher recall)

->Model Selection Strategy

The final model was selected based on its ability to align with business objectives.

Recall was prioritized, as failing to identify a churn-prone customer leads to direct revenue loss

Random Forest was chosen as the final model due to its improved performance over Logistic Regression

The focus of this project is not just predictive accuracy, but enabling proactive customer retention through reliable churn detection.

-> Model performance was evaluated using accuracy, precision, recall, and ROC-AUC to ensure a balanced and business-relevant assessment. While accuracy provides an overall measure of correctness, it is not sufficient for churn prediction due to potential class imbalance. Precision measures how many predicted churners were actually correct, helping control unnecessary retention efforts. Recall, which was the primary focus, measures how effectively the model identifies actual churners, since missing a churn-prone customer leads to direct revenue loss. ROC-AUC was used to evaluate the model’s ability to distinguish between churn and non-churn across different thresholds. Additionally, the classification threshold was tuned using the precision–recall trade-off to prioritize recall and align the model with business objectives.

## Results

### Logistic Regression

- Accuracy: 75%
- Precision (Churn): 52%
- Recall (Churn): 81%
- ROC-AUC: 0.85

### Random Forest

- Accuracy: 74%
- Precision (Churn): 50%
- Recall (Churn): 85%
- ROC-AUC: 0.84

### Final Model Selection

Random Forest was selected as the final model due to its higher recall, enabling better identification of customers at risk of churn.

The model prioritizes recall over precision to minimize the risk of missing potential churners, aligning with the business goal of proactive customer retention.