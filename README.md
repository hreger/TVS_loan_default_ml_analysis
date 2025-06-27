# TVS Loan Default Prediction

## 📌 Introduction
This repository provides an end-to-end machine learning pipeline for predicting loan defaults using **TVS Credit’s motorcycle loan dataset**. It benchmarks **gradient-boosted trees** (XGBoost) against several deep learning models (MLP, 1D CNN, LSTM), and evaluates them with both traditional metrics (ROC-AUC, PR-AUC) and a **custom business-weighted scoring function** that penalizes false positives more heavily — mimicking real-world lending risks.

---

## 📁 Project Structure

```
.
├── TVS_Loan_Default_Prediction.ipynb   # Jupyter notebook with full workflow
├── data/                               # Dataset directory (contains TVS.csv)
├── best_model_xgb.joblib               # Persisted XGBoost model (if selected)
├── best_nn_model.keras                 # Persisted neural network model (if selected)
├── requirements.txt                    # Project dependencies
└── README.md                           # This documentation
```

---

## ⚙️ Setup

### Clone the repository
```bash
git clone <repo-url>
cd tvs-loan-default-prediction
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Prepare data
- Download `TVS.csv` into the `data/` folder.
- Ensure it contains exactly **32 columns** including the `Loan_Default` target.

---

## 🧠 Notebook Workflow

### 🔧 Installation
Installs:
- `xgboost`, `optuna`, `imbalanced-learn`
- `scikit-learn`, `tensorflow`, `joblib`, `matplotlib`, etc.

### 📊 Data Loading & EDA
- Loads `data/TVS.csv`
- Prints dataset shape & summary stats
- Visualizes class imbalance

### 🏗️ Preprocessing Pipeline
- Numeric: `StandardScaler`
- Categorical: `OneHotEncoder`
- Combined using `ColumnTransformer`

### 🎯 Baseline Model: XGBoost
- Trained with `SMOTE` for handling imbalance
- Evaluated with ROC-AUC & PR-AUC

### 🔍 Hyperparameter Tuning
- `Optuna` runs 25 trials
- Optimizes based on **cross-validated ROC-AUC**

### 🤖 Neural Network Architectures
- **MLP**: Dense layers with BatchNorm + Dropout
- **1D CNN**: Treats features as 1D sequence
- **LSTM**: Reshapes flat input into sequences

All trained with `EarlyStopping` and evaluated on ROC-AUC.

### 💼 Business-Weighted Cost Metric
Custom function:
```
score = TP − 5 × FP
```
Reflects that **false positives (approving a defaulter)** are costlier than **false negatives**.

### 💾 Model Persistence
- Saves best model:
  - `.joblib` for XGBoost
  - `.keras` for neural networks

---

## 📈 Evaluation Metrics

| Metric         | Description                                      |
|----------------|--------------------------------------------------|
| **ROC-AUC**    | Area under ROC Curve (separates classes well)    |
| **PR-AUC**     | Area under Precision-Recall Curve (for imbalance)|
| **Business Score** | Custom metric: `TP − 5 × FP`                   |

---

## 🚀 Usage

### Run the notebook
```bash
jupyter notebook TVS_Loan_Default_Prediction.ipynb
```

### Retrain with new data
1. Replace or add your new `.csv` inside `data/`
2. Update path inside the notebook
3. Run all cells

### Load persisted model for inference

```python
import joblib, pandas as pd
model = joblib.load('best_model_xgb.joblib')
df_new = pd.read_csv('data/new_batch.csv')
preds = model.predict_proba(df_new)[:, 1]
```

---

## ✅ Results Summary

| Model     | ROC-AUC | PR-AUC | Business Score |
|-----------|---------|--------|----------------|
| XGBoost   | 0.8016  | 0.1827 | -34            |
| MLP       | 0.6236  | 0.1332 | 0              |
| 1-D CNN   | 0.6520  | 0.1384 | 0              |
| LSTM      | 0.4782  | 0.0931 | 0              |

> ⚠️ Note: Negative business score means false positives outweigh true positives.

---

## 🔮 Future Directions

- Explore **transformer-based tabular models** (e.g. FTTransformer, SAINT)
- Try **model ensembling** (blend XGBoost + MLP for better recall)
- Improve **LSTM performance** using actual temporal features
- Add **streamlit dashboard** for internal business users
- Deploy via **FastAPI** or **Flask** for API-based predictions

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
