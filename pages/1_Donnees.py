import streamlit as st, pandas as pd
from src.data_loader import DATASETS, load_uci
from src import plots

st.title("📊 Module 1 – Données")

@st.cache_data(show_spinner="Téléchargement depuis l'UCI...")
def get_uci(uci_id):
    return load_uci(uci_id)

source = st.radio("Source", ["Jeu de données du cas (UCI)", "Importer un CSV"], horizontal=True)

if source.startswith("Jeu"):
    name = st.selectbox("Jeu de données", list(DATASETS))
    cfg = DATASETS[name]
    df = get_uci(cfg["uci_id"])
else:
    f = st.file_uploader("Fichier CSV", type="csv")
    if f is None:
        st.info("Importe un CSV pour continuer."); st.stop()
    df, cfg = pd.read_csv(f), {}

# Cible + classe positive
default = cfg.get("target", df.columns[-1])
target = st.selectbox("Variable cible", df.columns, index=list(df.columns).index(default))
classes = sorted(df[target].dropna().unique().tolist())

# Classe positive par défaut : celle du cas métier, sinon la classe minoritaire
if target == cfg.get("target") and cfg.get("positive") in classes:
    default_pos = cfg["positive"]
else:
    default_pos = df[target].value_counts().idxmin()

pos = st.selectbox("Classe positive (classe d'intérêt)", classes,
                   index=classes.index(default_pos), key=f"pos_{target}")

df["y_bin"] = (df[target] == pos).astype(int)

# Garde-fou : la classe d'intérêt est presque toujours la minoritaire
if df["y_bin"].mean() > 0.5:
    st.warning(f"⚠️ La classe positive « {pos} » représente {df['y_bin'].mean():.1%} "
               "des données. Vérifie que c'est bien la classe d'intérêt (ex. la panne).")

st.session_state.update(df_raw=df, cfg=cfg, target=target, pos_class=pos)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Aperçu", "Qualité & classes", "Distributions", "Corrélations", "Taux par variable"])

with tab1:
    st.dataframe(df.head(50), use_container_width=True)
    st.dataframe(df.describe().T, use_container_width=True)
with tab2:
    miss = df.isna().sum().to_frame("manquantes")
    miss["%"] = (miss["manquantes"] / len(df) * 100).round(2)
    st.dataframe(miss)
    st.plotly_chart(plots.plot_class_balance(df, "y_bin"), use_container_width=True)
with tab3:
    num = df.drop(columns=["y_bin"]).select_dtypes("number").columns
    col = st.selectbox("Variable numérique", num)
    st.plotly_chart(plots.plot_distributions(df, "y_bin", col), use_container_width=True)
with tab4:
    st.plotly_chart(plots.plot_corr(df.drop(columns=["y_bin"])), use_container_width=True)
with tab5:
    col = st.selectbox("Variable", [c for c in df.columns if c != "y_bin"], key="rate")
    st.plotly_chart(plots.failure_rate_by(df, "y_bin", col), use_container_width=True)