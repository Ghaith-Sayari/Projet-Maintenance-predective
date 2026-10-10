import streamlit as st
from sklearn.model_selection import train_test_split
from src.preprocessing import (add_features, can_add_ai4i_features,
                               detect_types, build_preprocessor)

st.title("⚙️ Module 2 – Préparation")
ss = st.session_state

# ------------------------------------------------------------------
# Garde-fou : les données doivent être chargées (page 1)
# ------------------------------------------------------------------
if "df_raw" not in ss:
    st.warning("Commence par la page Données.")
    st.stop()

df = ss["df_raw"]
cfg = ss.get("cfg", {})
target = ss["target"]
pos = ss.get("pos_class")

st.info(f"Cible : **{target}** | Classe positive : **{pos}** | "
        f"Proportion de positifs : **{df['y_bin'].mean():.2%}**")

# ------------------------------------------------------------------
# Choix des variables explicatives
# ------------------------------------------------------------------
excluded = cfg.get("exclude", [])
cands = [c for c in df.columns if c not in (target, "y_bin")]

features = st.multiselect(
    "Variables explicatives", cands,
    default=[c for c in cands if c not in excluded],
)

leak = set(features) & set(excluded)
if leak:
    st.error(f"⚠️ Fuite de données : {sorted(leak)} ne doivent pas être utilisées "
             "(types de pannes ou identifiants).")

if not features:
    st.warning("Sélectionne au moins une variable.")
    st.stop()

# ------------------------------------------------------------------
# Feature engineering (seulement si les colonnes AI4I existent)
# ------------------------------------------------------------------
use_fe = False
if can_add_ai4i_features(features):
    use_fe = st.checkbox("Ajouter temp_diff, power, wear_torque", value=True)

# ------------------------------------------------------------------
# Paramètres de préparation
# ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
num_strategy = c1.selectbox("Imputation numérique", ["median", "mean"])
scaler = c2.selectbox("Normalisation", ["Standard", "MinMax", "Aucune"])
test_size = c3.slider("Taille du jeu de test", 0.1, 0.5, 0.2, 0.05)
seed = st.number_input("Graine aléatoire", value=42, step=1)

# ------------------------------------------------------------------
# Préparation
# ------------------------------------------------------------------
if st.button("Préparer les données", type="primary"):
    X = df[features]
    if use_fe:
        X = add_features(X)
    y = df["y_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=int(seed))

    # Garde-fou : la classe positive doit être la minoritaire
    if y_train.mean() > 0.5:
        st.error(f"⚠️ {y_train.mean():.1%} de positifs dans le train : la classe "
                 "positive semble inversée. Retourne dans la page Données et corrige-la.")
        st.stop()

    num_cols, cat_cols = detect_types(X_train)
    prep = build_preprocessor(num_cols, cat_cols, num_strategy, scaler)

    ss.update(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
              preprocessor=prep, seed=int(seed),
              num_cols=num_cols, cat_cols=cat_cols)

    # Les anciens modèles ne correspondent plus à ce nouveau split
    ss["models"] = {}
    ss["model_params"] = {}

    st.success("Données préparées. Les modèles précédents ont été supprimés : "
               "ré-entraîne-les dans la page Modélisation.")

# ------------------------------------------------------------------
# Résumé de l'état courant (reste affiché après les clics)
# ------------------------------------------------------------------
if "X_train" in ss:
    st.subheader("Résumé de la préparation")
    a, b, c = st.columns(3)
    a.metric("Lignes d'entraînement", f"{len(ss['X_train'])}")
    b.metric("Lignes de test", f"{len(ss['X_test'])}")
    c.metric("Positifs (train)", f"{ss['y_train'].mean():.2%}")

    st.write("**Variables numériques :**", ss["num_cols"])
    st.write("**Variables catégorielles :**", ss["cat_cols"])
    st.dataframe(ss["X_train"].head(10), use_container_width=True)