import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

AI4I_BASE = ["Air temperature", "Process temperature", "Rotational speed",
             "Torque", "Tool wear"]

def can_add_ai4i_features(columns) -> bool:
    return all(c in columns for c in AI4I_BASE)

def add_features(X):
    X = X.copy()
    X["temp_diff"] = X["Process temperature"] - X["Air temperature"]
    X["power"] = X["Torque"] * X["Rotational speed"] * 2 * np.pi / 60
    X["wear_torque"] = X["Tool wear"] * X["Torque"]
    return X

def detect_types(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return num, cat

def build_preprocessor(num_cols, cat_cols, num_strategy="median", scaler="Standard"):
    sc = {"Standard": StandardScaler(), "MinMax": MinMaxScaler(), "Aucune": "passthrough"}[scaler]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy=num_strategy)), ("sc", sc)]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])