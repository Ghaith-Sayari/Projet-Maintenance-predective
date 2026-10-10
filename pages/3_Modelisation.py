import streamlit as st, pandas as pd
from src.models import GRIDS, PARAM_SPECS, train_model, train_manual
from src.metrics import compute_metrics

st.title("🤖 Module 3 – Modélisation")
ss = st.session_state
if "X_train" not in ss:
    st.warning("Prépare d'abord les données (Module 2)."); st.stop()

ss.setdefault("models", {})          # modèles entraînés (conservés entre les clics)
ss.setdefault("model_params", {})    # paramètres utilisés pour chacun

fmt = lambda v: "illimitée" if v is None else str(v)

def param_widget(model, pname, spec):
    key = f"{model}_{pname}"
    if spec[0] == "slider":
        _, lo, hi, default, step = spec
        return st.slider(pname, lo, hi, default, step, key=key)
    _, options, default = spec
    return st.select_slider(pname, options=options, value=default,
                            format_func=fmt, key=key)

# ---------- Mode ----------
mode = st.radio("Mode d'entraînement",
                ["Manuel (je règle les hyperparamètres)",
                 "Optimisation automatique (RandomizedSearchCV)"], horizontal=True)
manual = mode.startswith("Manuel")

# ---------- Choix des modèles (interrupteurs) ----------
st.subheader("1. Modèles à entraîner")
cols = st.columns(len(PARAM_SPECS))
active = [m for c, m in zip(cols, PARAM_SPECS)
          if c.toggle(m, value=m in ("LogReg", "RandomForest", "XGBoost"), key=f"on_{m}")]

# ---------- Hyperparamètres ----------
st.subheader("2. Hyperparamètres")
settings = {}
for m in active:
    with st.expander(m, expanded=True):
        if manual:
            settings[m] = {p: param_widget(m, p, s) for p, s in PARAM_SPECS[m].items()}
        else:
            settings[m] = {
                p: (st.multiselect(p.replace("clf__", ""), v, default=v,
                                   format_func=fmt, key=f"grid_{m}_{p}") or v)
                for p, v in GRIDS[m].items()
            }

if not manual:
    c1, c2, c3 = st.columns(3)
    n_iter = c1.slider("Combinaisons testées", 3, 30, 10)
    cv_folds = c2.slider("Folds de validation croisée", 3, 10, 5)
    scoring = c3.selectbox("Critère d'optimisation", ["average_precision", "roc_auc", "f1"])

# ---------- Entraînement ----------
st.subheader("3. Entraînement")
b1, b2 = st.columns([1, 1])
run = b1.button("🚀 Entraîner les modèles activés", type="primary", disabled=not active)
if b2.button("🗑️ Réinitialiser tous les modèles"):
    ss["models"], ss["model_params"] = {}, {}
    st.rerun()

if run:
    bar = st.progress(0.0)
    for i, m in enumerate(active):
        with st.spinner(f"Entraînement de {m}..."):
            if manual:
                est = train_manual(m, ss["preprocessor"], ss["X_train"],
                                   ss["y_train"], settings[m], ss["seed"])
                ss["model_params"][m] = {"mode": "manuel", **settings[m]}
            else:
                grid = {f"clf__{p.replace('clf__', '')}": v for p, v in settings[m].items()}
                est, best, cv_score = train_model(m, ss["preprocessor"], ss["X_train"],
                                                  ss["y_train"], grid, n_iter, cv_folds,
                                                  scoring, ss["seed"])
                ss["model_params"][m] = {"mode": "auto", f"{scoring} (CV)": round(cv_score, 3),
                                         **{k.replace("clf__", ""): v for k, v in best.items()}}
            ss["models"][m] = est          # remplace uniquement ce modèle
        bar.progress((i + 1) / len(active))
    st.success("Entraînement terminé.")

# ---------- Tableau comparatif ----------
if ss["models"]:
    st.subheader("Tableau comparatif (jeu de test, seuil 0,5)")
    rows = []
    for m, est in ss["models"].items():
        p = est.predict_proba(ss["X_test"])[:, 1]
        rows.append({"modèle": m, **compute_metrics(ss["y_test"].values, p)})
    st.dataframe(pd.DataFrame(rows).set_index("modèle").round(3)
                 .sort_values("PR-AUC", ascending=False), use_container_width=True)

    st.subheader("Paramètres utilisés")
    st.dataframe(pd.DataFrame(ss["model_params"]).T.astype(str), use_container_width=True)