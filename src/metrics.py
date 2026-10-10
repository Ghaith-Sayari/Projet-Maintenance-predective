import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

def compute_metrics(y_true, proba, thr=0.5, credit=False):
    pred = (proba >= thr).astype(int)
    auc = roc_auc_score(y_true, proba)
    out = {
        "AUC-ROC": auc,
        "PR-AUC": average_precision_score(y_true, proba),
        "Précision": precision_score(y_true, pred, zero_division=0),
        "Rappel": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred, zero_division=0),
    }
    if credit:  # scoring de crédit uniquement
        out["Gini"] = 2 * auc - 1
        out["KS"] = ks_2samp(proba[y_true == 1], proba[y_true == 0]).statistic
    return out