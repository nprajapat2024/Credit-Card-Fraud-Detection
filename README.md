# Credit Card Fraud Detection

## 📌 Problem Statement
The goal of this project is to predict fraudulent credit card transactions using Machine Learning.  
The dataset contains **284,807 transactions**, out of which only **492 are frauds (0.172%)**, making it a highly imbalanced classification problem.

Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## 📊 Business Problem
Credit card fraud leads to **billions of dollars in losses** each year for banks and financial institutions.  
Machine Learning helps reduce manual reviews, costly chargebacks, and false transaction denials.  

---

## 🔍 Project Pipeline
1. **Data Understanding** – Loaded dataset, reviewed features (time, amount, PCA components V1–V28, and class).  
2. **Exploratory Data Analysis (EDA)** – Univariate/bivariate analysis, skewness checks, fraud vs. non-fraud comparison.  
3. **Data Preprocessing** – Train-test split, scaling, handling imbalance using SMOTE/undersampling.  
4. **Model Building** – Logistic Regression, Random Forest, XGBoost, and ANN. Hyperparameter tuning via GridSearchCV.  
5. **Model Evaluation** – Evaluated using AUC-ROC, Precision, Recall, F1-score, and Confusion Matrix (recall prioritized).  

---

## ⚙️ Technologies Used
- **Python**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn, Imbalanced-learn**
- **XGBoost, LightGBM**
- **Jupyter Notebook**

---

## 📈 Results
- Best Model: **XGBoost Classifier**
- **AUC-ROC:** 0.97  
- **Recall (Fraud class):** 0.91  
- **Precision:** 0.88  
- **F1-score:** 0.89  

---

## 🚀 Key Learnings
- Handling **imbalanced datasets** using SMOTE and undersampling.  
- Choosing the right **evaluation metrics** for fraud detection (Recall > Accuracy).  
- Building end-to-end ML pipeline for real-world financial applications.  

---

## 📂 Repository Structure
- `notebooks/` – Jupyter notebooks for EDA & modeling  
- `scripts/` – Python scripts for modular implementation  
- `results/` – Performance metrics & visualizations  
- `requirements.txt` – List of dependencies  

---

## 🔗 References
- [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- [Worldline & ULB ML Group Research Paper](https://arxiv.org/abs/1908.01802)  
