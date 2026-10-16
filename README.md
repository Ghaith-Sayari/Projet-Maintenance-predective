# Classification explicable — Maintenance prédictive

Application web interactive développée avec **Streamlit** pour explorer des données de maintenance prédictive, préparer les variables, entraîner plusieurs modèles de classification et analyser leurs prédictions.

> **État du projet :** prototype fonctionnel pour une utilisation interactive et pédagogique. L'application n'est pas encore une solution de production : l'API n'est pas implémentée, aucun modèle n'est sauvegardé automatiquement et aucun test automatisé n'est fourni dans le dépôt.

## Contexte

- **Jeu de données principal :** AI4I 2020 Predictive Maintenance, disponible via l'UCI (identifiant 601)
- **Taille annoncée du jeu de données :** 10 000 observations
- **Variable cible :** `Machine failure`
- **Problème traité :** prédire la probabilité de panne d'une machine industrielle à partir de ses conditions de fonctionnement
- **Déséquilibre des classes :** environ 3,4 % de pannes dans le jeu de données AI4I 2020
- **Cadre pédagogique :** Data Mining — II3-IA / II3-IF — ENSI — 2026-2027

## Démonstration

- Application Streamlit : https://projet-maintenance-predective.streamlit.app/Modelisation

La disponibilité de l'application déployée dépend de l'environnement Streamlit Cloud et des versions des dépendances installées.

## Fonctionnalités actuellement disponibles

### 1. Données — `pages/1_Donnees.py`

- Téléchargement du jeu AI4I 2020 depuis l'UCI via `ucimlrepo`.
- Import d'un fichier CSV local.
- Sélection de la variable cible et de la classe positive.
- Aperçu des données et statistiques descriptives.
- Analyse des valeurs manquantes.
- Visualisation de la répartition des classes.
- Visualisations exploratoires interactives.

### 2. Préparation — `pages/2_Preparation.py`

- Sélection des variables explicatives.
- Création de variables dérivées, notamment :
  - `temp_diff`
  - `power`
  - `wear_torque`
- Imputation des valeurs manquantes.
- Encodage des variables catégorielles.
- Normalisation des variables numériques.
- Séparation stratifiée entre les jeux d'entraînement et de test.
- Conservation du préprocesseur dans la session Streamlit.

### 3. Modélisation — `pages/3_Modelisation.py`

Les modèles suivants sont disponibles :

- Régression logistique (`LogReg`)
- Arbre de décision (`DecisionTree`)
- K plus proches voisins (`KNN`)
- Forêt aléatoire (`RandomForest`)
- XGBoost (`XGBoost`)

Deux modes sont proposés :

- **Mode manuel :** réglage des hyperparamètres dans l'interface.
- **Optimisation automatique :** recherche aléatoire avec `RandomizedSearchCV` et validation croisée stratifiée.

Les modèles entraînés sont conservés dans `st.session_state` pendant la session courante. Ils ne sont pas enregistrés automatiquement sur disque.

### 4. Évaluation — `pages/4_Evaluation.py`

- Comparaison des modèles sur le jeu de test.
- ROC-AUC.
- PR-AUC (`average precision`).
- Précision, rappel et F1-score.
- Courbe précision-rappel.
- Courbe F1 en fonction du seuil.
- Matrice de confusion.
- Ajustement interactif du seuil de décision.

### 5. Explicabilité — `pages/5_Explicabilite.py`

- Importance native des modèles lorsqu'elle est disponible.
- Importance par permutation.
- Explications globales avec SHAP.
- Explication locale d'une observation avec SHAP.
- Explication locale avec LIME.
- Comparaison des contributions SHAP et LIME.
- Génération d'une phrase de synthèse pour une prédiction individuelle.

La compatibilité et les résultats des méthodes SHAP et LIME peuvent dépendre des versions installées des bibliothèques et du modèle sélectionné.

## Fonctionnalités non disponibles ou incomplètes

### API

Le fichier `pages/6_API.py` existe, mais il est actuellement vide. Aucun service FastAPI ou endpoint d'inférence n'est implémenté dans le dépôt.

### Notebook

Le fichier `notebooks/predictive-maintenance.ipynb` existe, mais il est actuellement vide. Il ne contient pas encore d'analyse exploratoire ou de résultats reproductibles.

### Persistance des modèles

Le dossier `models/` n'est pas présent dans l'arborescence actuelle. Les modèles sont conservés uniquement dans la session Streamlit et sont perdus lorsque la session est réinitialisée ou arrêtée.

### Tests et intégration continue

Le dépôt ne contient actuellement ni suite de tests automatisés, ni configuration de linting, ni workflow CI documenté.

## Prétraitement et prévention de la fuite de données

Pour le jeu AI4I 2020, les colonnes suivantes sont prévues pour être exclues :

- Identifiants : `UDI`, `Product ID`
- Colonnes décrivant directement certains types de panne : `TWF`, `HDF`, `PWF`, `OSF`, `RNF`

Ces variables peuvent introduire une fuite de données, car elles sont liées à la cible `Machine failure`.

> Lors de l'import d'un fichier CSV personnalisé, il est recommandé de vérifier manuellement que les variables sélectionnées ne contiennent pas d'information directement dérivée de la cible.

## Structure actuelle du projet

```text
Projet-Maintenance-predective/
├── Home.py
├── pages/
│   ├── 1_Donnees.py
│   ├── 2_Preparation.py
│   ├── 3_Modelisation.py
│   ├── 4_Evaluation.py
│   ├── 5_Explicabilite.py
│   └── 6_API.py                 # actuellement vide
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── explain.py
│   ├── metrics.py
│   ├── models.py
│   ├── plots.py
│   └── preprocessing.py
├── notebooks/
│   └── predictive-maintenance.ipynb  # actuellement vide
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation et exécution locale

### Prérequis

- Python 3.9 ou version ultérieure recommandée
- Connexion Internet nécessaire pour télécharger le jeu de données UCI automatiquement

### Installation

```bash
python -m venv venv
```

Sous Windows :

```bash
venv\\Scripts\\activate
```

Sous macOS ou Linux :

```bash
source venv/bin/activate
```

Installez ensuite les dépendances :

```bash
pip install -r requirements.txt
```

### Lancement

```bash
streamlit run Home.py
```

L'application est généralement accessible à l'adresse :

```text
http://localhost:8501
```

## Parcours recommandé

1. Ouvrir la page **Données** et charger le jeu UCI ou un fichier CSV.
2. Vérifier la cible, les classes et la qualité des données.
3. Configurer les variables dans **Préparation**.
4. Entraîner un ou plusieurs modèles dans **Modélisation**.
5. Comparer les résultats dans **Évaluation**.
6. Examiner les facteurs influençant les prédictions dans **Explicabilité**.

## Limites connues

- Les performances dépendent fortement de la qualité et de la structure du fichier CSV importé.
- Les petits jeux de données ou les classes très rares peuvent empêcher une séparation stratifiée ou une validation croisée fiable.
- Le seuil optimal ne doit pas être considéré comme un seuil de production sans validation complémentaire.
- Le KNN ne dispose pas d'un paramètre `class_weight` natif comme certains autres modèles.
- Les résultats SHAP et LIME doivent être interprétés comme des explications de modèle et non comme des preuves causales.
- L'application ne fournit actuellement pas de mécanisme de suivi, de versionnement ou de sauvegarde des modèles entraînés.

## Dépendances principales

Les bibliothèques utilisées sont listées dans `requirements.txt`, notamment :

- Streamlit
- pandas
- NumPy
- SciPy
- scikit-learn
- XGBoost
- Plotly
- ucimlrepo
- joblib
- SHAP
- LIME
- Matplotlib

Les versions ne sont pas actuellement verrouillées dans `requirements.txt`. Les résultats peuvent donc varier selon l'environnement d'installation.

## Améliorations prévues

- Implémenter une API d'inférence réelle.
- Ajouter la validation complète des fichiers CSV importés.
- Ajouter des tests automatisés.
- Ajouter la sauvegarde et le chargement des modèles.
- Compléter le notebook d'analyse.
- Verrouiller les versions des dépendances.
- Renforcer la prévention automatique des fuites de données.
- Ajouter une configuration de qualité de code et d'intégration continue.
