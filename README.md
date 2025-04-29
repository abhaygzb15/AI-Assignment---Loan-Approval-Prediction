# AI-Assignment---Loan-Approval-Prediction

Loan Approval Prediction 🏦📊
This project implements a machine learning-based loan approval prediction system using historical loan application data. The model predicts whether a loan application will be approved based on key applicant information like income, education, employment status, credit history, and loan amount.

📌 Project Overview
Loan approval is a critical task in the banking and finance sector. Traditionally done manually, this process can be optimized using machine learning for faster and more reliable decisions. This project:

Cleans and preprocesses real-world loan data,

Explores the data using visualizations,

Builds classification models,

Evaluates them using metrics like accuracy, confusion matrix, and ROC-AUC,

Performs hyperparameter tuning for optimization.

🧠 Technologies & Tools
Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-learn (Logistic Regression, Random Forest, SVM, GridSearchCV)

Jupyter Notebook / Kaggle Notebook

📁 Dataset
The dataset is sourced from Kaggle:
🔗 Loan Prediction Dataset

It contains 614 entries with 13 columns, including:

Applicant demographics

Income details

Loan amount and term

Credit history

Loan approval status (target variable)

🔍 Key Steps
Data Cleaning

Handled missing values using mode and median.

Removed irrelevant columns like Loan_ID.

Exploratory Data Analysis (EDA)

Univariate and bivariate visualizations.

Outlier detection using IQR method.

Feature Engineering

Label Encoding for categorical features.

StandardScaler for numerical features.

Model Building

Logistic Regression ✅

Random Forest Classifier

Support Vector Classifier (SVC)

Model Evaluation

Logistic Regression achieved highest accuracy: 86%

ROC-AUC score: 0.78

Hyperparameter Tuning

Used GridSearchCV to optimize Logistic Regression model.

📈 Results
Best Model: Logistic Regression

Test Accuracy: 86%

AUC Score: 0.78

The model shows strong predictive performance and generalizes well.

🏁 Conclusion
This project demonstrates the effective application of machine learning for binary classification problems in the financial domain. The final model can help banks automate their loan approval process, improving both speed and reliability.
