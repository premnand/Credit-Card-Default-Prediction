# Credit Card Default Prediction: A Data-Driven Approach to Risk Mitigation

## Project Overview

This project focuses on predicting credit card defaults to assist financial institutions in mitigating financial risks. Using a dataset of 30,000 customer records with six months of historical payment data, we developed machine learning models to identify high-risk individuals. The project encompassed data preprocessing, feature engineering, model selection, and evaluation to build an effective predictive tool.

## 1. Problem Statement

Credit card defaults pose a significant financial risk to banks. The goal was to develop a predictive model that balances precision (minimizing false positives) and recall (minimizing false negatives) to ensure effective risk management while minimizing undue penalties on customers.

## 2. Data Preparation and Feature Engineering

### Handling Missing Values

- The dataset had less than 1% missing values, which were removed to maintain data integrity without affecting analysis.

### Feature Engineering

To improve predictive accuracy, key features were engineered:

- **Credit Utilization Ratio** – Measures financial strain based on credit limit usage.
- **Payment-to-Bill Ratio** – Evaluates repayment behavior.
- **Past Payment Consistency** – Assesses reliability in past payments over six months.

## 3. Data Transformation

### Standardization

- Standardized numerical features such as bill amounts and past payments to ensure comparability.

### Encoding Categorical Variables

- One-hot encoding was applied to categorical variables like education level and marital status.

### Principal Component Analysis (PCA)

- Used PCA to reduce dimensionality while retaining 95% variance, addressing multicollinearity concerns.

## 4. Model Development

Several machine learning models were trained and compared:

- **Logistic Regression** – Baseline model for performance benchmarking.
- **Decision Tree** – Provided interpretability through visualization of decision paths.
- **Random Forest** – Improved accuracy using an ensemble learning approach.
- **XGBoost** – Leveraged advanced gradient-boosting techniques for optimal performance.

## 5. Model Evaluation

The F1-score was selected as the primary metric to balance precision and recall:

- **Precision:** 85%
- **Recall:** 82%

Among the models, **XGBoost** performed best, making it the preferred choice for real-world deployment.

## 6. Key Insights

- Customers with high credit utilization ratios and low payment-to-bill ratios were at higher risk of default.
- Consistency in past payments was a strong predictor of default behavior.

## 7. Addressing Challenges

### Class Imbalance

- The dataset had an imbalance ratio of 3.52:1 (Non-Defaults: Defaults).
- **Solution:** Used **SMOTE (Synthetic Minority Oversampling Technique)** to improve recall without significantly compromising precision.

### Computational Efficiency

- PCA was used to reduce the number of features and optimize model performance.
- Feature selection techniques ensured only relevant predictors were included.

## 8. Business Impact and Deployment Potential

### Application Areas

- **Credit Risk Assessment** – Helps banks set appropriate credit limits and identify high-risk customers.
- **Proactive Outreach** – Enables institutions to engage with potential defaulters before they miss payments.
- **Risk-Adjusted Interest Rates** – Allows banks to personalize lending terms based on risk scores.

### Cost-Benefit Analysis

- Reducing default rates can save banks millions in potential losses.
- Ensuring accurate predictions minimizes unnecessary penalties on low-risk customers, enhancing customer satisfaction.

## 9. Technical Details

### Data Preprocessing

- Outlier detection using **Boxplots, IQR, and Z-Score**.
- Removal of incorrect data entries (e.g., negative payments, invalid bill amounts).
- **Variance Inflation Factor (VIF)** analysis before PCA to reduce multicollinearity.

### Model Development

- **Validation Technique:** k-fold cross-validation.
- **Hyperparameter Tuning:** Grid Search and Random Search.
- **Performance Metrics:** Tracked **ROC-AUC, Accuracy**, and prioritized **F1-score**.
- **SMOTE Implementation:** Evaluated the impact of balancing classes on model performance.

## 10. Challenges and Learnings

### Handling Class Imbalance

- Considered alternative techniques like **class-weighted loss functions** and **under-sampling**.

### Model Interpretation

- **SHAP values** and **feature importance scores** provided explainability for model decisions.

### Real-World Deployment Challenges

- Ensuring low-latency predictions for seamless integration into banking systems.
- Implementing real-time decision thresholds for adaptive credit risk management.

## 11. Future Enhancements

### Algorithm Improvements

- Exploring **deep learning models** such as Neural Networks.
- Testing advanced boosting algorithms like **CatBoost and LightGBM**.

### Data Augmentation

- Creating **synthetic datasets** to model out-of-distribution scenarios and improve model robustness.

### Real-Time Implementation

- Integrating **real-time risk monitoring** using **Apache Kafka** or **AWS Kinesis**.
- Deploying **adaptive retraining mechanisms** for evolving financial trends.

## 12. Summary

This project successfully developed a machine learning model for **credit card default prediction**, leveraging **feature engineering, PCA for dimensionality reduction, and XGBoost** for high-performance classification. The solution provides banks with a robust tool for credit risk assessment, enabling proactive engagement with high-risk customers. 

### Future improvements include:

- Deep learning integration.
- Real-time risk monitoring.
- Enhanced model interpretation techniques for better financial decision-making.
