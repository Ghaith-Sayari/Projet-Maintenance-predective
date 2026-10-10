import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------------
# Grilles pour le mode "optimisation automatique" (RandomizedSearchCV)
# ---------------------------------------------------------------
GRIDS = {
    "LogReg": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "DecisionTree": {"clf__max_depth": [3, 5, 7, 10], "clf__min_samples_leaf": [5, 10, 20, 50]},
    "KNN": {"clf__n_neighbors": [3, 5, 11, 21, 31], "clf__weights": ["uniform", "distance"]},
    "RandomForest": {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [5, 10, None],
                     "clf__min_samples_leaf": [1, 5, 10]},
    "XGBoost": {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.03, 0.1, 0.3], "clf__subsample": [0.7, 1.0]},
}

# ---------------------------------------------------------------
# NOUVEAU : widgets pour le mode "manuel"
# ("slider", min, max, défaut, pas)  ou  ("select", options, défaut)
# ---------------------------------------------------------------
PARAM_SPECS = {
    "LogReg": {
        "C": ("select", [0.01, 0.1, 1, 10, 100], 1),
    },
    "DecisionTree": {
        "max_depth": ("slider", 2, 20, 5, 1),
        "min_samples_leaf": ("slider", 1, 100, 10, 1),
    },
    "KNN": {
        "n_neighbors": ("slider", 1, 51, 5, 2),
        "weights": ("select", ["uniform", "distance"], "uniform"),
    },
    "RandomForest": {
        "n_estimators": ("slider", 50, 500, 200, 50),
        "max_depth": ("select", [3, 5, 10, 15, 20, None], 10),
        "min_samples_leaf": ("slider", 1, 30, 5, 1),
    },
    "XGBoost": {
        "n_estimators": ("slider", 50, 500, 200, 50),
        "max_depth": ("slider", 2, 10, 5, 1),
        "learning_rate": ("select", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.1),
        "subsample": ("slider", 0.5, 1.0, 1.0, 0.05),
    },
}


def make_estimator(name, pos_weight=1.0, seed=42):
    return {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        "KNN": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                                 random_state=seed, n_jobs=-1),
    }[name]


# ---------------------------------------------------------------
# NOUVEAU : construction d'un pipeline avec des paramètres choisis
# ---------------------------------------------------------------
def build_pipeline(name, preprocessor, y_train, seed=42, **params):
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pipe = Pipeline([("prep", clone(preprocessor)),
                     ("clf", make_estimator(name, pos_weight, seed))])
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    return pipe


# ---------------------------------------------------------------
# NOUVEAU : entraînement manuel (un seul fit, pas de recherche)
# ---------------------------------------------------------------
def train_manual(name, preprocessor, X_train, y_train, params, seed=42):
    pipe = build_pipeline(name, preprocessor, y_train, seed, **params)
    pipe.fit(X_train, y_train)
    return pipe


# ---------------------------------------------------------------
# Optimisation automatique (inchangée)
# ---------------------------------------------------------------
def train_model(name, preprocessor, X_train, y_train, grid,
                n_iter=10, cv_folds=5, scoring="average_precision", seed=42):
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pipe = Pipeline([("prep", clone(preprocessor)),
                     ("clf", make_estimator(name, pos_weight, seed))])
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    search = RandomizedSearchCV(
        pipe, grid, n_iter=min(n_iter, n_combos),
        cv=StratifiedKFold(cv_folds, shuffle=True, random_state=seed),
        scoring=scoring, random_state=seed, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_