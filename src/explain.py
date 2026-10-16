import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

TREE_MODELS = ("DecisionTree", "RandomForest", "XGBoost")

METHOD_LABEL = {
    "Linear": "shap.LinearExplainer",
    "Tree": "shap.TreeExplainer",
    "Explainer": "shap.Explainer (générique, algorithme permutation)",
}


# ------------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------------
def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def feature_names(prep):
    """Noms lisibles après encodage : 'num__Torque' -> 'Torque'."""
    return [n.split("__", 1)[1] if "__" in n else n for n in prep.get_feature_names_out()]


def pick_method(model_name, choice="Auto"):
    if choice.startswith("Générique"):
        return "Explainer"
    if model_name == "LogReg":
        return "Linear"
    if model_name in TREE_MODELS:
        return "Tree"
    return "Explainer"


def output_unit(model_name, method):
    """Unité des valeurs SHAP selon le modèle."""
    if method == "Linear" or (method == "Tree" and model_name == "XGBoost"):
        return "log-odds"
    return "probabilité"


# ------------------------------------------------------------------
# SHAP
# ------------------------------------------------------------------
def build_shap_bundle(est, X_train, num_cols, method, n_background=100, seed=42):
    prep, clf = est.named_steps["prep"], est.named_steps["clf"]
    names = feature_names(prep)
    bg = X_train.sample(min(n_background, len(X_train)), random_state=seed)
    Xt_bg = _dense(prep.transform(bg))

    if method == "Linear":
        explainer = shap.LinearExplainer(clf, Xt_bg)
    elif method == "Tree":
        explainer = shap.TreeExplainer(clf)
    else:
        f = lambda d: clf.predict_proba(d)[:, 1]
        explainer = shap.Explainer(
            f, shap.maskers.Independent(Xt_bg, max_samples=100),
            algorithm="permutation", feature_names=names)

    return {"est": est, "explainer": explainer, "method": method, "names": names,
            "num_cols": list(num_cols), "max_evals": 3 * (2 * len(names) + 1)}


def explain_rows(bundle, X_rows):
    """Retourne un shap.Explanation (classe positive) pour les lignes données."""
    prep = bundle["est"].named_steps["prep"]
    Xt = _dense(prep.transform(X_rows))
    m, ex = bundle["method"], bundle["explainer"]

    if m == "Tree":
        e = ex(Xt, check_additivity=False)
    elif m == "Explainer":
        e = ex(Xt, max_evals=bundle["max_evals"], silent=True)
    else:
        e = ex(Xt)

    vals, base = np.asarray(e.values), np.asarray(e.base_values)
    if vals.ndim == 3:                       # sklearn : une sortie par classe
        vals = vals[:, :, 1]
        base = base[:, 1] if base.ndim == 2 else base
    if base.ndim == 0:
        base = np.full(len(vals), float(base))

    # Valeurs affichées en unités d'origine pour les variables numériques
    disp = Xt.copy()
    n_num = len(bundle["num_cols"])
    if n_num:
        disp[:, :n_num] = X_rows[bundle["num_cols"]].to_numpy(dtype=float)

    return shap.Explanation(values=vals, base_values=base, data=disp,
                            feature_names=bundle["names"])


def aggregate_to_original(values_row, names, num_cols, cat_cols):
    """Somme les colonnes one-hot pour revenir aux variables d'origine."""
    s = pd.Series(values_row, index=names)
    out = {c: s[c] for c in num_cols}
    for c in cat_cols:
        out[c] = s[[n for n in names if n.startswith(c + "_")]].sum()
    return pd.Series(out)


# ------------------------------------------------------------------
# Figures SHAP (matplotlib)
# ------------------------------------------------------------------
def fig_bar(expl, max_display=12):
    plt.close("all")
    shap.plots.bar(expl, max_display=max_display, show=False)
    fig = plt.gcf(); fig.tight_layout()
    return fig


def fig_beeswarm(expl, max_display=12):
    plt.close("all")
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    fig = plt.gcf(); fig.tight_layout()
    return fig




def fig_waterfall(expl_one, max_display=12):
    plt.close("all")
    shap.plots.waterfall(expl_one, max_display=max_display, show=False)
    fig = plt.gcf(); fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Importances globales (hors SHAP)
# ------------------------------------------------------------------
def native_importance(est):
    prep, clf = est.named_steps["prep"], est.named_steps["clf"]
    names = feature_names(prep)
    if hasattr(clf, "coef_"):
        return pd.Series(clf.coef_.ravel(), index=names), "Coefficient (variables standardisées)"
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=names), "Importance interne de l'arbre / forêt"
    return None


def perm_importance(est, X_test, y_test, n_repeats=5, seed=42):
    r = permutation_importance(est, X_test, y_test, scoring="average_precision",
                               n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"mean": r.importances_mean, "std": r.importances_std},
                         index=X_test.columns).sort_values("mean"))


# ------------------------------------------------------------------
# LIME (travaille sur les variables d'origine : explications lisibles)
# ------------------------------------------------------------------
def make_lime_explainer(X_train, num_cols, cat_cols, seed=42):
    cols = list(num_cols) + list(cat_cols)
    levels = {c: sorted(X_train[c].astype(str).unique()) for c in cat_cols}
    data = X_train[cols].copy()
    for c in cat_cols:
        data[c] = data[c].astype(str).map({v: i for i, v in enumerate(levels[c])})
    cat_idx = list(range(len(num_cols), len(cols)))
    cat_names = {len(num_cols) + k: levels[c] for k, c in enumerate(cat_cols)}
    explainer = LimeTabularExplainer(
        data.to_numpy(dtype=float), feature_names=cols,
        class_names=["Négatif", "Positif"], categorical_features=cat_idx,
        categorical_names=cat_names, mode="classification",
        discretize_continuous=True, random_state=seed)
    return explainer, cols, levels


def lime_explain(est, lime_bundle, row, num_cols, cat_cols, num_features=10, num_samples=5000):
    explainer, cols, levels = lime_bundle
    r = row.iloc[0]
    x = [float(r[c]) for c in num_cols]
    for c in cat_cols:
        v = str(r[c])
        x.append(float(levels[c].index(v)) if v in levels[c] else 0.0)

    def predict_fn(arr):
        d = pd.DataFrame(arr, columns=cols)
        for c in cat_cols:
            d[c] = [levels[c][int(round(v))] for v in d[c]]
        return est.predict_proba(d)

    return explainer.explain_instance(np.array(x), predict_fn, labels=(1,),
                                      num_features=num_features, num_samples=num_samples)


def lime_to_series(exp, cols):
    m = dict(exp.as_map()[1])
    return pd.Series({cols[i]: w for i, w in m.items()}).reindex(cols).fillna(0.0)


def lime_r2(exp):
    try:
        sc = exp.score
        return float(sc[1]) if isinstance(sc, dict) else float(sc)
    except Exception:
        return None


def compare_shap_lime(shap_agg, lime_agg, k=5):
    df = pd.DataFrame({"SHAP": shap_agg, "LIME": lime_agg}).fillna(0.0)
    top = df["SHAP"].abs().nlargest(k).index
    agree = int((np.sign(df.loc[top, "SHAP"]) == np.sign(df.loc[top, "LIME"])).sum())
    rho = df["SHAP"].abs().corr(df["LIME"].abs(), method="spearman")
    norm = df / df.abs().max().replace(0, 1)       # échelle commune [-1, 1]
    return norm, agree, len(top), rho


# ------------------------------------------------------------------
# Phrase de synthèse
# ------------------------------------------------------------------
def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " et " + items[-1]


def synthesis_sentence(prob, thr, contribs, row, label, top=3):
    c = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(top)

    def fmt(f):
        v = row[f]
        return f"{f} ({v:.4g})" if isinstance(v, (int, float, np.number)) else f"{f} = {v}"

    ups = [fmt(f) for f, v in c.items() if v > 0]
    downs = [fmt(f) for f, v in c.items() if v < 0]
    level = "élevée" if prob >= thr else "faible"
    s = f"La probabilité de « {label} » est {level} ({prob:.0%}, seuil {thr:.2f})."
    if ups:
        s += f" Elle est principalement tirée vers le haut par {_join(ups)}."
    if downs:
        s += f" Elle est réduite par {_join(downs)}."
    return s