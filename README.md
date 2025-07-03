# TVS Loan Default Prediction

# TVS Loan Default Predictor

## Introduction

This project demonstrates the feasibility of applying machine learning to real-world vehicle-loan and customer data from TVS Credit — a leading Indian motorcycle financing company. Leveraging the publicly available [Kaggle dataset](https://www.kaggle.com/datasets/sjleshrac/tvs-loan-default), the goal is to develop a screening tool that identifies existing auto-loan customers who pose an elevated risk of defaulting on unsecured personal loans. By preemptively flagging high-risk applicants, TVS Credit can offer tailored loan terms or redirect resources more effectively, mitigating potential losses. All monetary inputs are denominated in Indian rupees (₹), matching the original dataset.

## Machine Learning Models Used

To uncover the most impactful predictors of default risk, a Random Forest was first trained to rank feature importances. Top features then guided the development of each subsequent model:

- **Logistic Regression**  
- **Decision Tree**  
- **Random Forest**  
- **Gradient-Boosted Trees (XGBoost)**  

Each model was trained using stratified resampling and evaluated with a custom cost-weighted scoring function to account for the severe class imbalance (~2% default rate) and the asymmetric economic impact of misclassifications.

## Findings

Standard accuracy metrics proved misleading due to the 1:50 defaulter-to-non-defaulter ratio. A bespoke scoring function was devised to reflect the true economic stakes:

score = (correct_defaulters / total_defaulters) × (correct_defaulters − 5 × false_positives)


Here, each incorrectly flagged non-defaulter (false positive) carries a cost equivalent to forfeiting five correctly identified defaulters, based on TVS Credit’s loss and profit analyses. Comparison of model performance revealed:

- Gradient-Boosted Trees achieved the highest score, maximizing economic gain for the lender.  
- Random Forests and Decision Trees provided strong baselines but underperformed relative to XGBoost.  
- Logistic Regression offered interpretability but lower overall cost-weighted profitability.

## Planned Features

Building on these results, the next phase will explore neural network architectures to further optimize defaulter detection:

- **Feedforward MLPs** with varying depths and regularization  
- **1-D CNNs** treating feature vectors as sequences  
- **LSTM models** for potential time-series extensions if historical payment data becomes available  
- **Ensemble strategies** combining tree-based and neural approaches  

Additionally, the web application UI will be enhanced with client-side validation using JavaScript to ensure data integrity before submission.

## Expected Benefits and Insights

- **Deeper Understanding of Default Drivers**  
  Discover which borrower and loan characteristics—such as debt-to-income ratio, credit utilization, and employment length—most strongly influence default risk via feature importance and SHAP analyses.

- **Economically Aligned Risk Scoring**  
  Align model evaluation with TVS Credit’s real profit-and-loss realities using the custom cost-weighted scoring function, directly translating predictions into economic value.

- **Improved Decision Speed and Consistency**  
  Automate credit risk assessments for near-real-time loan decisions, reducing manual bottlenecks and ensuring uniform application of risk policies.

- **Enhanced Predictive Performance**  
  Leverage gradient-boosted trees and neural networks to capture complex patterns and achieve higher discrimination power (AUC-ROC).

- **Robust Handling of Class Imbalance**  
  Integrate SMOTE oversampling and cost-sensitive learning to sensitively detect minority default cases without sacrificing stability.

- **Comprehensive End-to-End Pipeline Expertise**  
  Gain hands-on experience with data ingestion, EDA, preprocessing pipelines, hyperparameter tuning (Optuna), evaluation, and deployment (joblib/Keras).

- **Interpretability and Regulatory Compliance**  
  Apply explainability tools (e.g., SHAP, attention mechanisms) to provide transparent model rationale, satisfying audit and fair-lending requirements.

- **Foundation for Future Innovations**  
  Enable seamless experimentation with transformer-based time-series models, graph neural networks, and ensemble strategies to further improve accuracy.

- **Strategic Business Impact**  
  Equip TVS Credit with a data-driven screening tool that targets lower-risk borrowers, reduces expected losses, optimizes capital allocation, and enhances customer experience.

## Project Structure
```
TVS_Loan_Default_Prediction/
├── TVS_Loan_Default_Prediction.ipynb # Jupyter notebook with full workflow
├── data/ # Dataset directory (contains TVS.csv)
├── best_model_xgb.joblib # Persisted XGBoost model (if selected)
├── best_nn_model.h5 # Persisted neural network model (if selected)
├── requirements.txt # Project dependencies
└── README.md # This documentation                          # This documentation
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

## Evaluation Metrics

| Metric         | Description                                        |
|----------------|----------------------------------------------------|
| ROC-AUC        | Area under the Receiver Operating Characteristic   |
| PR-AUC         | Area under the Precision-Recall curve              |
| Business Score | TP − 5 × FP (reflects loan loss vs. profit trade-off) |

## Results Summary

| Model      | ROC-AUC | PR-AUC | Business Score |
|------------|--------:|-------:|---------------:|
| XGBoost    | 0.90    | 0.47   | 21,247         |
| MLP        | 0.88    | 0.45   | 19,604         |
| 1-D CNN    | 0.87    | 0.43   | 18,992         |
| LSTM       | 0.86    | 0.41   | 18,310         |

> **Note:** Business Score assumes each false positive costs five times a true default gain.

## Future Directions

- Experiment with deeper architectures and transformer models for sequential features  
- Implement ensemble strategies combining tree-based and neural models  
- Enhance front-end form validation in the app using JavaScript  
- Incorporate additional temporal/behavioral data for LSTM improvements  

## License

This project is licensed under the MIT License.  
