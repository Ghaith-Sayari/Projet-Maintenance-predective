import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from src import explain as xai

st.title("🔍 Module 5 – Explicabilité (XAI)")
ss = st.session_state

# ------------------------------------------------------------------
# Garde-fous
# ------------------------------------------------------------------
if "X_test" not in ss:
    st.warning("Les données ne sont pas préparées : va dans la page Préparation.")
    st.stop()
if not ss.get("models"):
    st.warning("Aucun modèle entraîné : va dans la page Modélisation.")
    st.stop()

X_train, X_test, y_test = ss["X_train"], ss["X_test"], ss["y_test"]
num_cols, cat_cols = ss["num_cols"], ss["cat_cols"]
label = f"{ss.get('target', 'cible')} = {ss.get('pos_class', 1)}"


def cached(key, owner, fn):
    """Cache dans la session ; invalidé si l'objet 'owner' (modèle, X_train) change."""
    cache = ss.setdefault("xai_cache", {})
    entry = cache.get(key)
    if entry is not None and entry["owner"] is owner:
        return entry["val"]
    val = fn()
    cache[key] = {"owner": owner, "val": val}
    return val


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------
# Choix du modèle et de la méthode
# ------------------------------------------------------------------
models = list(ss["models"])
default = models.index(ss["chosen_model"]) if ss.get("chosen_model") in models else 0

c1, c2, c3 = st.columns(3)
name = c1.selectbox("Modèle à expliquer", models, index=default)
choice = c2.radio("Méthode SHAP", ["Auto (adaptée au modèle)", "Générique (shap.Explainer)"])
est = ss["models"][name]
method = xai.pick_method(name, choice)
n_sample = c3.slider("Individus pour les graphes globaux", 50, 1000,
                     100 if method == "Explainer" else 300, 50)

unit = xai.output_unit(name, method)
st.info(f"**{name}** → `{xai.METHOD_LABEL[method]}` · valeurs SHAP exprimées en **{unit}**")

bundle = cached(("bundle", name, method), est,
                lambda: xai.build_shap_bundle(est, X_train, num_cols, method))

tab_g, tab_l = st.tabs(["🌍 Explications globales", "👤 Explication d'un individu"])

# ==================================================================
# GLOBAL
# ==================================================================
with tab_g:
    # --- Importance native ---
    st.subheader("1. Importance propre au modèle")
    nat = xai.native_importance(est)
    if nat is None:
        st.info("Ce modèle (KNN) n'a pas d'importance interne : utilise la permutation ou SHAP.")
    else:
        s, kind = nat
        s = s.reindex(s.abs().sort_values().index).tail(15)
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                               marker_color=["#d62728" if v > 0 else "#1f77b4" for v in s.values]))
        fig.update_layout(title=kind, template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # --- Permutation ---
    st.subheader("2. Importance par permutation (jeu de test)")
    with st.spinner("Calcul de l'importance par permutation..."):
        perm = cached(("perm", name), est, lambda: xai.perm_importance(est, X_test, y_test))
    fig = go.Figure(go.Bar(x=perm["mean"], y=perm.index, orientation="h",
                           error_x=dict(type="data", array=perm["std"])))
    fig.update_layout(title="Baisse de PR-AUC quand la variable est mélangée",
                      xaxis_title="Baisse de PR-AUC", template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Comparable entre modèles. Les variables corrélées se partagent l'importance.")

    # --- SHAP global ---
    st.subheader("3. SHAP global")
    Xs = X_test.sample(min(n_sample, len(X_test)), random_state=42)
    with st.spinner(f"Calcul des valeurs SHAP sur {len(Xs)} individus..."):
        expl = cached(("global", name, method, len(Xs)), est,
                      lambda: xai.explain_rows(bundle, Xs))

    st.markdown("**Importance moyenne (|SHAP|)**")
    show(xai.fig_bar(expl))

    st.markdown("**Beeswarm** : chaque point est un individu. "
                "Rouge = valeur élevée de la variable, bleu = valeur faible. "
                "À droite, la variable augmente le risque.")
    show(xai.fig_beeswarm(expl))


# ==================================================================
# LOCAL
# ==================================================================
with tab_l:
    proba_all = pd.Series(est.predict_proba(X_test)[:, 1], index=X_test.index)

    thr = st.slider("Seuil de décision", 0.0, 1.0, float(ss.get("threshold", 0.5)),
                    0.01, key="xai_thr")
    pred_all = (proba_all >= thr).astype(int)

    flt = st.radio("Individus du jeu de test à afficher",
                   ["Tous", "Positifs réels", "Négatifs réels",
                    "Prédits positifs", "Erreurs du modèle"], horizontal=True)
    mask = {
        "Tous": pd.Series(True, index=X_test.index),
        "Positifs réels": y_test == 1,
        "Négatifs réels": y_test == 0,
        "Prédits positifs": pred_all == 1,
        "Erreurs du modèle": pred_all != y_test,
    }[flt]
    pool = proba_all[mask].sort_values(ascending=False).index.tolist()

    if not pool:
        st.info("Aucun individu ne correspond à ce filtre.")
        st.stop()

    sel = st.selectbox(
        f"Individu ({len(pool)} disponibles, triés par probabilité décroissante)", pool,
        format_func=lambda i: f"#{i} | réel = {int(y_test.loc[i])} | proba = {proba_all.loc[i]:.2f}")
    row = X_test.loc[[sel]]
    p = float(proba_all.loc[sel])

    a, b, c = st.columns(3)
    a.metric("Probabilité prédite", f"{p:.1%}")
    b.metric("Décision au seuil", "POSITIF ⚠️" if p >= thr else "Négatif ✅")
    c.metric("Valeur réelle", int(y_test.loc[sel]))
    st.dataframe(row.T.astype(str).rename(columns={sel: "valeur"}), use_container_width=True)

    # --- SHAP waterfall ---
    st.subheader("SHAP : graphique en cascade (waterfall)")
    expl1 = cached(("local", name, method, sel), est, lambda: xai.explain_rows(bundle, row))
    show(xai.fig_waterfall(expl1[0]))
    st.caption(f"Part de la valeur moyenne E[f(x)] vers la prédiction f(x), en {unit}. "
               "Rouge = pousse vers la classe positive, bleu = vers la négative.")

    # --- LIME ---
    st.subheader("LIME : explication locale")
    l1, l2 = st.columns(2)
    nf = l1.slider("Nombre de variables LIME", 5, 15, 10)
    ns = l2.select_slider("Échantillons de perturbation", [1000, 2000, 5000, 10000], value=5000)

    with st.spinner("Calcul de l'explication LIME..."):
        lime_b = cached(("lime_b",), X_train,
                        lambda: xai.make_lime_explainer(X_train, num_cols, cat_cols))
        exp = cached(("lime", name, sel, nf, ns), est,
                     lambda: xai.lime_explain(est, lime_b, row, num_cols, cat_cols, nf, ns))

    items = exp.as_list(label=1)[::-1]
    fig = go.Figure(go.Bar(
        x=[w for _, w in items], y=[r for r, _ in items], orientation="h",
        marker_color=["#d62728" if w > 0 else "#1f77b4" for _, w in items]))
    fig.update_layout(title="Poids LIME (effet sur la probabilité de la classe positive)",
                      template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)
    r2 = xai.lime_r2(exp)
    if r2 is not None:
        st.caption(f"Fidélité du modèle local LIME (R²) : **{r2:.2f}**. "
                   "Un R² faible signifie que l'explication est peu fiable pour cet individu.")

    # --- Comparaison SHAP / LIME ---
    st.subheader("Comparaison SHAP / LIME")
    shap_agg = xai.aggregate_to_original(expl1.values[0], expl1.feature_names, num_cols, cat_cols)
    lime_agg = xai.lime_to_series(exp, num_cols + cat_cols)
    norm, agree, k, rho = xai.compare_shap_lime(shap_agg, lime_agg)

    show_df = norm.reindex(norm["SHAP"].abs().sort_values().index).tail(10)
    fig = go.Figure()
    fig.add_bar(y=show_df.index, x=show_df["SHAP"], orientation="h", name="SHAP")
    fig.add_bar(y=show_df.index, x=show_df["LIME"], orientation="h", name="LIME")
    fig.update_layout(barmode="group", template="plotly_white", height=450,
                      title="Contributions normalisées (échelle commune [-1, 1])")
    st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric(f"Accord de signe (top {k} SHAP)", f"{agree} / {k}")
    m2.metric("Corrélation de rang (Spearman)", "n/a" if pd.isna(rho) else f"{rho:.2f}")
    st.caption("SHAP et LIME n'ont pas la même unité : on compare les signes et les rangs, "
               "pas les valeurs brutes. Les colonnes one-hot sont regroupées par variable d'origine.")

    # --- Phrase de synthèse ---
    st.subheader("Phrase de synthèse")
    st.success(xai.synthesis_sentence(p, thr, shap_agg, row.iloc[0], label))