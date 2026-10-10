import streamlit as st

# Doit être la 1re commande Streamlit du fichier
st.set_page_config(
    page_title="Classification explicable – Maintenance prédictive",
    page_icon="🛠️",
    layout="wide",
)

ss = st.session_state

# ------------------------------------------------------------------
# En-tête
# ------------------------------------------------------------------
st.title("🛠️ Classification explicable – Maintenance prédictive")
st.caption("Data Mining · II3-IA / II3-IF · ENSI · Cas n°15 : AI4I 2020 (UCI)")

st.markdown("""
Cette application prédit **la panne d'une machine industrielle** à partir de ses
conditions de fonctionnement (températures, vitesse de rotation, couple, usure de l'outil),
compare plusieurs modèles de classification, puis **explique pourquoi** le modèle prédit
une panne, globalement et pour une machine donnée.
""")

# ------------------------------------------------------------------
# Contexte métier
# ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.info("**Problème métier**\n\nAnticiper une panne pour planifier la maintenance "
        "avant l'arrêt de la machine.")
c2.info("**Difficulté**\n\nSeulement ~3,4 % de pannes : l'accuracy est trompeuse. "
        "On suit le rappel, la précision, le F1 et la PR-AUC.")
c3.info("**Enjeu**\n\nUne panne ratée coûte plus cher qu'une fausse alerte. "
        "Le seuil de décision se règle selon ce compromis.")

st.divider()

# ------------------------------------------------------------------
# Parcours de l'application
# ------------------------------------------------------------------
st.subheader("Parcours de l'application")

steps = [
    ("1", "Données", "Téléchargement automatique ou import CSV, statistiques, "
                     "visualisations exploratoires.", "pages/1_Donnees.py"),
    ("2", "Préparation", "Variables, valeurs manquantes, encodage, normalisation, "
                         "séparation train/test.", "pages/2_Preparation.py"),
    ("3", "Modélisation", "5 modèles, hyperparamètres réglables, optimisation, "
                          "tableau comparatif.", "pages/3_Modelisation.py"),
    ("4", "Évaluation", "ROC, précision-rappel, matrice de confusion, "
                        "curseur de seuil.", "pages/4_Evaluation.py"),
    ("5", "Explicabilité", "SHAP, LIME, importance par permutation, "
                           "explication d'une machine.", "pages/5_Explicabilite.py"),
    ("6", "API", "Service FastAPI d'interprétation appelé par l'application.",
     "pages/6_API.py"),
]

import os
for num, title, desc, path in steps:
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if os.path.exists(path):
            st.page_link(path, label=f"{num}. {title}", icon="➡️")
        else:
            st.markdown(f"**{num}. {title}**  \n*(bientôt disponible)*")
    with col_b:
        st.write(desc)

st.divider()

# ------------------------------------------------------------------
# Avancement de la session
# ------------------------------------------------------------------
st.subheader("État de la session")

status = {
    "Données chargées": "df_raw" in ss,
    "Données préparées": "X_train" in ss,
    "Modèles entraînés": bool(ss.get("models")),
}
cols = st.columns(len(status))
for col, (label, ok) in zip(cols, status.items()):
    col.metric(label, "✅ Oui" if ok else "⏳ Non")

if status["Modèles entraînés"]:
    st.success("Modèles disponibles : " + ", ".join(ss["models"].keys()))
elif status["Données préparées"]:
    st.info("Données prêtes : passe à la page **Modélisation**.")
elif status["Données chargées"]:
    st.info("Données chargées : passe à la page **Préparation**.")
else:
    st.info("Commence par la page **Données** dans le menu de gauche.")

# ------------------------------------------------------------------
# Barre latérale
# ------------------------------------------------------------------
with st.sidebar:
    st.header("À propos")
    st.markdown("""
**Jeu de données** : AI4I 2020 Predictive Maintenance (UCI)
**Taille** : 10 000 machines
**Cible** : `Machine failure`

**Variables retirées** (fuite de données) :
`TWF, HDF, PWF, OSF, RNF`, `UDI`, `Product ID`

**Variables ajoutées** : `temp_diff`, `power`, `wear_torque`
""")
    st.divider()
    if st.button("🔄 Réinitialiser la session"):
        ss.clear()
        st.rerun()

# ------------------------------------------------------------------
# Pied de page
# ------------------------------------------------------------------
st.divider()
st.caption("Projet Challenge 1 · 2026-2027")