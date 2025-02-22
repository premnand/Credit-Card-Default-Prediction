# Credit Card Default Prediction

This project aims to predict credit card defaulters using machine learning techniques. The dataset contains information about credit card users, including their payment history, credit limit, and demographic details. The goal is to build a model that can accurately classify users as defaulters or non-defaulters.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Acknowledgments](#acknowledgments)

---

## Project Overview
The project involves the following steps:
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature engineering.
- **Exploratory Data Analysis (EDA)**: Visualizing data distributions and correlations.
- **Model Training**: Training and evaluating multiple machine learning models, including Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- **Model Evaluation**: Using metrics like ROC-AUC, confusion matrix, and classification report to assess model performance.

---

## Dataset
The dataset used in this project is `Credit_Card_Defaulter_Prediction.csv`. It contains the following features:
- **ID**: Unique identifier for each user.
- **CREDIT_LIMIT**: Credit limit of the user.
- **SEX**: Gender of the user (Male/Female).
- **EDUCATION**: Education level of the user.
- **MARRIAGE**: Marital status of the user.
- **PAY_1 to PAY_6**: Payment history for the last 6 months.
- **BILL_AMT1 to BILL_AMT6**: Bill amounts for the last 6 months.
- **PAY_AMT1 to PAY_AMT6**: Payment amounts for the last 6 months.
- **DEFAULT**: Target variable indicating whether the user defaulted (Yes/No).

---

## Methodology
### 1. Data Preprocessing:
- Renamed columns for consistency.
- Handled missing values and duplicates.
- Encoded categorical variables using label encoding and one-hot encoding.
- Created new features like `CREDIT_UTILIZATION` and `PAST_PAYMENT_CONSISTENCY`.

### 2. Exploratory Data Analysis (EDA):
- Visualized the distribution of categorical and numerical features.
- Analyzed the correlation between features using a heatmap.

### 3. Model Training:
- Split the data into training and testing sets.
- Applied SMOTE to handle class imbalance.
- Trained and evaluated four models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost

### 4. Model Evaluation:
- Used metrics like ROC-AUC, confusion matrix, and classification report.
- Compared the performance of all models.

---

## Results
The performance of the models is summarized below:

| Model               | ROC-AUC Score |
|---------------------|---------------|
| Logistic Regression | 0.69          |
| Decision Tree       | 0.74          |
| Random Forest       | 0.62          |
| XGBoost             | 0.75          |

---

## Conclusion
- The best-performing model was **XGBoost** with an ROC-AUC score of 0.75.
- Feature engineering and handling class imbalance using SMOTE improved model performance.
- The project successfully demonstrates the application of machine learning to predict credit card defaults.

---

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/premnand/Credit-Card-Default-Prediction.git
   cd Credit-Card-Default-Prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook code1.ipynb
   ```

---

## Usage
1. Open the Jupyter Notebook `code1.ipynb`.
2. Execute the cells sequentially to preprocess the data, train the models, and evaluate their performance.
3. Modify the code to experiment with different models or techniques.

---

## Acknowledgments
- Dataset sourced from Kaggle.
- Libraries used: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, and imbalanced-learn.

---
