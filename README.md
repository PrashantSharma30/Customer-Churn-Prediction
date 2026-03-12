# Customer Churn Prediction (Ensemble ML)

This project builds a machine learning pipeline to predict **telecom customer churn** using tabular customer data. The solution combines **hyperparameter optimization with Optuna** and an **ensemble of XGBoost and LightGBM models** to achieve strong predictive performance.

---
## Technologies Used

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Optuna
- Matplotlib / Seaborn
- Kaggle Notebook

---

## Dataset

The dataset contains telecom customer information used to predict whether a customer will **churn (leave the service)**.

- **Training size:** ~500,000 customers  
- **Features:** ~20–30 variables  
- **Target:** `Churn` (1 = churn, 0 = stay)

---

## Exploratory Data Analysis

Visualizations were used to understand customer behavior and identify churn drivers.

Key insights:

- **Churn distribution:** Shows class imbalance, which makes ROC-AUC a better evaluation metric than accuracy alone.
- **Tenure vs Churn:** New customers are more likely to churn.
- **Monthly Charges vs Churn:** Higher charges correlate with higher churn probability.
- **Contract Type vs Churn:** Month-to-month contracts have significantly higher churn rates.

These insights help guide **feature importance and model design**.

---

## Model Development

Two gradient boosting models were trained:

- **XGBoost**
- **LightGBM**

Hyperparameters were optimized using **Optuna**, which performs automated hyperparameter search to improve model performance.

---

## Ensemble Learning

Instead of relying on a single model, predictions from both models were combined.
