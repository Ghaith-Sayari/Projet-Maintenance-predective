import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, confusion_matrix)

# =====================================================================
# MODULE 1 : exploration des données
# =====================================================================
def plot_class_balance(df, target):
    counts = df[target].value_counts().rename_axis("classe").reset_index(name="n")
    counts["%"] = (counts["n"] / counts["n"].sum() * 100).round(2)
    fig = px.bar(counts, x="classe", y="n", text="%", color="classe",
                 title="Répartition de la cible")
    fig.update_traces(texttemplate="%{text}%")
    fig.update_xaxes(type="category")
    return fig


def plot_distributions(df, target, col):
    fig = px.histogram(df, x=col, color=df[target].astype(str), barmode="overlay",
                       histnorm="probability density", opacity=0.6, nbins=40,
                       marginal="box", title=f"Distribution de {col} selon la cible",
                       labels={"color": target})
    return fig


def plot_corr(df):
    corr = df.select_dtypes("number").corr()
    return px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, title="Matrice de corrélation")


def failure_rate_by(df, target, col, bins=10):
    d = df.copy()
    if pd.api.types.is_numeric_dtype(d[col]) and d[col].nunique() > 10:
        d["groupe"] = pd.qcut(d[col], q=bins, duplicates="drop").astype(str)
        order = sorted(d["groupe"].unique(),
                       key=lambda s: float(s.split(",")[0].strip("([")))
    else:
        d["groupe"] = d[col].astype(str)
        order = sorted(d["groupe"].unique())
    rate = d.groupby("groupe")[target].agg(["mean", "size"]).reset_index()
    rate["mean"] *= 100
    fig = px.bar(rate, x="groupe", y="mean", hover_data=["size"],
                 category_orders={"groupe": order},
                 labels={"mean": "Taux de classe positive (%)", "groupe": col},
                 title=f"Taux de classe positive selon {col}")
    fig.add_hline(y=d[target].mean() * 100, line_dash="dash",
                  annotation_text="moyenne globale")
    return fig


# =====================================================================
# MODULE 4 : évaluation
# =====================================================================
def roc_figure(y_true, probas: dict):
    fig = go.Figure()
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="grey"), name="hasard"))
    fig.update_layout(title="Courbes ROC", template="plotly_white",
                      xaxis_title="Taux de faux positifs",
                      yaxis_title="Taux de vrais positifs (rappel)")
    return fig


def pr_figure(y_true, probas: dict):
    fig = go.Figure()
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=f"{name} (PR-AUC={ap:.3f})"))
    fig.add_hline(y=float(np.mean(y_true)), line_dash="dash", line_color="grey",
                  annotation_text="prévalence (modèle aléatoire)")
    fig.update_layout(title="Courbes précision-rappel", template="plotly_white",
                      xaxis_title="Rappel", yaxis_title="Précision",
                      yaxis_range=[0, 1.02])
    return fig


def f1_threshold_figure(y_true, probas: dict):
    fig = go.Figure()
    for name, p in probas.items():
        prec, rec, thr = precision_recall_curve(y_true, p)
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        fig.add_trace(go.Scatter(x=thr, y=f1, mode="lines", name=name))
        i = f1.argmax()
        fig.add_trace(go.Scatter(x=[thr[i]], y=[f1[i]], mode="markers",
                                 marker=dict(size=10, symbol="star"),
                                 showlegend=False,
                                 hovertext=f"{name}: F1={f1[i]:.3f} @ seuil={thr[i]:.2f}"))
    fig.add_vline(x=0.5, line_dash="dash", line_color="grey",
                  annotation_text="seuil 0,5")
    fig.update_layout(title="F1 selon le seuil de décision (★ = meilleur seuil)",
                      template="plotly_white", xaxis_title="Seuil",
                      yaxis_title="F1", yaxis_range=[0, 1.02])
    return fig


def confusion_figure(y_true, proba, thr):
    cm = confusion_matrix(y_true, (proba >= thr).astype(int), labels=[0, 1])
    return px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                     x=["Prédit 0", "Prédit 1"], y=["Réel 0", "Réel 1"],
                     title=f"Matrice de confusion (seuil = {thr:.2f})")