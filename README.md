# Explainable Classification: Predictive Maintenance

Interactive web app that predicts **machine failure** from operating conditions,
compares several classification models, and explains the predictions.

- **Dataset:** AI4I 2020 Predictive Maintenance (UCI, id 601), 10,000 machines
- **Target:** `Machine failure` (about 3.4 % positive class)
- **Course:** Data Mining, II3-IA / II3-IF, ENSI, 2026-2027

## Links

- Live app: [Streamlit URL]
## Features

| Module | Content |
|---|---|
| 1. Data | Automatic UCI download or CSV upload, statistics, missing values, class balance, interactive plots |
| 2. Preparation | Feature selection, imputation, encoding, scaling, train/test split |
| 3. Modeling | Logistic Regression, Decision Tree, KNN, Random Forest, XGBoost, manual or automatic tuning |
| 4. Evaluation | ROC, precision-recall, F1 vs threshold, confusion matrix, decision threshold slider |
| 5. Explainability | SHAP, LIME, permutation importance *(in progress)* |
| 6. API | FastAPI service *(in progress)* |

## Project structure

```
projet/
├── Home.py
├── pages/
│   ├── 1_Donnees.py
│   ├── 2_Preparation.py
│   ├── 3_Modelisation.py
│   └── 4_Evaluation.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── metrics.py
│   └── plots.py
├── notebooks/predictive-maintenance.ipynb
├── models/
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
streamlit run Home.py
```

The app opens at http://localhost:8501. Follow the pages in order:
Data, Preparation, Modeling, Evaluation.

## Data handling notes

- Failure-type columns (`TWF, HDF, PWF, OSF, RNF`) are **removed** because they
  directly encode the target (data leakage). `UDI` and `Product ID` are identifiers.
- Engineered features: `temp_diff`, `power`, `wear_torque`.
- The classes are imbalanced, so models use class weights and are compared with
  PR-AUC, recall, precision and F1 rather than accuracy.
