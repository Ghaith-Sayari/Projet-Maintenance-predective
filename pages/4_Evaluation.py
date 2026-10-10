import streamlit as st
import pandas as pd
from src.metrics import compute_metrics
from src import plots

st.title("📈 Module 4 – Évaluation")
ss = st.session_state

# ------------------------------------------------------------------
# Garde-fous
# ------------------------------------------------------------------
if "X_test" not in ss:
    st.warning("Les données ne sont pas préparées : va dans la page Préparation.")
    st.stop()

if not ss.get("models"):
    st.warning("Aucun modèle entraîné : va dans la page Modélisation, "
               "active au moins un modèle et clique sur « Entraîner ».")
    st.stop()

# ------------------------------------------------------------------
# Probabilités de chaque modèle sur le jeu de test
# ------------------------------------------------------------------
y = ss["y_test"].values
probas = {m: est.predict_proba(ss["X_test"])[:, 1] for m, est in ss["models"].items()}

prevalence = y.mean()
st.caption(f"Jeu de test : {len(y)} lignes, {int(y.sum())} positifs "
           f"(prévalence = {prevalence:.2%})")

if prevalence > 0.5:
    st.error("⚠️ La prévalence est supérieure à 50 % : la classe positive semble "
             "inversée. Corrige la page Données, puis refais Préparation et Modélisation.")
    st.stop()

credit = st.checkbox("Cas de scoring de crédit (ajouter Gini et KS)")

# ------------------------------------------------------------------
# Courbes superposées
# ------------------------------------------------------------------
col1, col2 = st.columns(2)
col1.plotly_chart(plots.roc_figure(y, probas), use_container_width=True)
col2.plotly_chart(plots.pr_figure(y, probas), use_container_width=True)
st.plotly_chart(plots.f1_threshold_figure(y, probas), use_container_width=True)

# ------------------------------------------------------------------
# Seuil de décision
# ------------------------------------------------------------------
st.subheader("Seuil de décision")
chosen = st.selectbox("Modèle", list(probas.keys()))
thr = st.slider("Seuil", 0.0, 1.0, 0.5, 0.01)

ss["threshold"] = thr
ss["chosen_model"] = chosen

m = compute_metrics(y, probas[chosen], thr, credit)
cols = st.columns(len(m))
for c, (k, v) in zip(cols, m.items()):
    c.metric(k, f"{v:.3f}")

st.plotly_chart(plots.confusion_figure(y, probas[chosen], thr),
                use_container_width=True)

# ------------------------------------------------------------------
# Tous les modèles au seuil choisi
# ------------------------------------------------------------------
st.subheader("Tous les modèles au seuil choisi")
table = pd.DataFrame({k: compute_metrics(y, p, thr, credit) for k, p in probas.items()}).T
st.dataframe(table.round(3), use_container_width=True)