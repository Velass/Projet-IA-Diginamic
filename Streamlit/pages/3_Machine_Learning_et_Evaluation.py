import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import (
    neural_network,
    pipeline,
    metrics,
    model_selection,
    preprocessing,
    tree,
    ensemble
)

st.set_page_config(page_title="Machine Learning et Évaluation", page_icon="🤖")

df_raw = pd.read_csv("./data/vin.csv", sep=",").iloc[:, 1:]
st.title("Machine Learning et Évaluation")

st.markdown(
    """
Bienvenue dans la section Machine Learning.  
📌 **Choisissez une approche :**
"""
)

sous_page = st.radio(
    "🔹 Sélectionnez une méthode :", ["Arbre de Décision", "Réseau de Neurones", "Foret Aleatoire"]
)

if sous_page == "Arbre de Décision":
    st.markdown("## 🌳 Arbre de Décision")
    st.write("Nous allons maintenant entraîner un **arbre de décision**.")

    df_raw["target"] = df_raw["target"].map(
        {"Vin sucré": 0, "Vin éuilibré": 1, "Vin amer": 2}
    )

    target = ["target"]
    features = [col for col in df_raw.columns if col not in target]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        df_raw[features], df_raw[target], test_size=0.2, random_state=42
    )

    st.title("Machine Learning et Evaluation")
    st.markdown("## données pret.")
    st.write(
        "nous pouvons donc voir que les données sont prêtes pour le machine learning et ci-dessous les different pourcentage pour notre target."
    )
    st.write(y_train["target"].value_counts(normalize=True))

    st.markdown("## Entraînement d'un arbre de décision")
    pipe = pipeline.Pipeline(
        [
            # ("feature_selection", feature_selection),
            # ('std_scaler', preprocessing.StandardScaler()),
            ("decision_tree", tree.DecisionTreeClassifier())
        ]
    )
    pipe.fit(X_train, y_train)
    st.write(
        "Nous allons maintenant entraîner un arbre de décision sur nos données.\n"
        "Nous avons créé et entraîné un pipeline qui contient un arbre de décision.\n"
        "Nous pouvons maintenant l'évaluer."
    )
    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)

    st.markdown(
        f"""
    **📊 Accuracy sur le train set :** `{train_acc:.4f}`  
    **📊 Accuracy sur le test set :** `{test_acc:.4f}`
    """
    )

    st.markdown("### 🌳 Visualisation de l'Arbre de Décision")

    fig, ax = plt.subplots(figsize=(12, 6))
    tree.plot_tree(
        pipe.named_steps["decision_tree"],  # Utilisation du bon index dans le pipeline
        feature_names=X_train.columns,
        filled=True,
        rounded=True,
        class_names=["Vin sucré", "Vin équilibré", "Vin amer"],
        fontsize=8,
        ax=ax,
    )

    st.pyplot(fig)

    st.markdown("### 📊 Matrice de Confusion")
    # with st.expander("Matrice de confusion", expanded=False):
    cm_display = metrics.ConfusionMatrixDisplay.from_predictions(
        y_test, pipe.predict(X_test)
    )
    fig, ax = plt.subplots(figsize=(3, 2))
    cm_display.plot(ax=ax, cmap="Blues", colorbar=True)
    st.pyplot(fig, use_container_width=False)

    st.markdown("### 📄 Rapport de Classification")
    report = metrics.classification_report(y_test, pipe.predict(X_test))
    st.code(report, language="text")


if sous_page == "Réseau de Neurones":
    st.markdown("## 🧠 Réseau de Neurones")
    st.write("Nous allons maintenant entraîner un **réseau de neurones**.")

    df_raw["target"] = df_raw["target"].map(
        {"Vin sucré": 0, "Vin éuilibré": 1, "Vin amer": 2}
    )

    target = ["target"]
    features = [col for col in df_raw.columns if col not in target]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        df_raw[features], df_raw[target], test_size=0.2, random_state=42
    )

    st.title("Machine Learning et Evaluation")
    st.markdown("## données pret.")
    st.write(
        "nous pouvons donc voir que les données sont prêtes pour le machine learning et ci-dessous les different pourcentage pour notre target."
    )
    st.write(y_train["target"].value_counts(normalize=True))

    st.markdown("## Entraînement d'un Réseau de Neurones")
    pipe = pipeline.Pipeline(
        [
            # ("feature_selection", feature_selection),
            ("std_scaler", preprocessing.StandardScaler()),
            ("neural_network", neural_network.MLPClassifier()),
        ]
    )
    pipe.fit(X_train, y_train)
    st.write(
        "Nous allons maintenant entraîner un réseau de neurones sur nos données.\n"
        "Nous avons créé et entraîné un pipeline qui contient un réseau de neurones.\n"
        "Nous pouvons maintenant l'évaluer."
    )
    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)

    st.markdown(
        f"""
    **📊 Accuracy sur le train set :** `{train_acc:}`  
    **📊 Accuracy sur le test set :** `{test_acc}`
    """
    )

    st.markdown("### Visualisation Réseau de Neurones")

    st.markdown("### Nombre de Neurones")
# Vérifier si le modèle contient un réseau de neurones
    if "neural_network" in pipe.named_steps:
        hidden_layers = pipe.named_steps["neural_network"].hidden_layer_sizes

        # Affichage du nombre de neurones par couche
        st.write("### 🧠 Nombre de neurones par couche du Réseau de Neurones :")
        st.write(f"📌 Couches cachées : {hidden_layers}")
        st.write(
            f"📌 Nombre total de couches (y compris entrée et sortie) : {len(hidden_layers) + 2}"
        )

        st.markdown("### 📊 Matrice de Confusion")
        # with st.expander("Matrice de confusion", expanded=False):
        cm_display = metrics.ConfusionMatrixDisplay.from_predictions(
            y_test, pipe.predict(X_test)
        )
        fig, ax = plt.subplots(figsize=(3, 2))
        cm_display.plot(ax=ax, cmap="Blues", colorbar=True)
        st.pyplot(fig, use_container_width=False)

        st.markdown("### 📄 Rapport de Classification")
        report = metrics.classification_report(y_test, pipe.predict(X_test))
        st.code(report, language="text")
    
if sous_page == "Foret Aleatoire":
    st.markdown("## 🌳🌳 Foret Aleatoire")
    st.write("Nous allons maintenant entraîner une **Foret Aleatoire**.")
    
    df_raw["target"] = df_raw["target"].map(
        {"Vin sucré": 0, "Vin éuilibré": 1, "Vin amer": 2}
    )
    target = ["target"]
    features = [col for col in df_raw.columns if col not in target]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        df_raw[features], df_raw[target], test_size=0.2, random_state=42
    )

    st.title("Machine Learning et Evaluation")
    st.markdown("## les données sont pretes.")
    st.write(
        "nous pouvons donc voir que les données sont prêtes pour le machine learning et ci-dessous les different pourcentage pour notre target."
    )
    st.write(y_train["target"].value_counts(normalize=True))

    st.markdown("## Entraînement de la Foret Aleatoire")
    pipe = pipeline.Pipeline(
        [
            ('std_scaler', preprocessing.StandardScaler()),
            ("random_forest", ensemble.RandomForestClassifier())
        ]
    )
    pipe.fit(X_train, y_train)
    st.write(
        "Nous allons maintenant entraîner un arbre de décision sur nos données.\n"
        "Nous avons créé et entraîné un pipeline qui contient un arbre de décision.\n"
        "Nous pouvons maintenant l'évaluer."
    )
    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)

    st.markdown(
        f"""
    **📊 Accuracy sur le train set :** `{train_acc}`  
    **📊 Accuracy sur le test set :** `{test_acc}`
    """
    )

if "random_forest" in pipe.named_steps:
    rf_model = pipe.named_steps["random_forest"]
    
    st.markdown("### 🌳🌳 Visualisation de plusieurs Arbres de la Forêt Aléatoire")

    # Création de la figure
    fig, axes = plt.subplots(2, 2, figsize=(26, 18))  # Grille de 2x2
    axes = axes.ravel()  # Aplatir la grille en 1D pour itérer facilement
    n_trees = min(4, len(rf_model.estimators_)) 
    for i in range(n_trees):
        tree.plot_tree(
            rf_model.estimators_[i],
            feature_names=X_train.columns,
            filled=True,
            rounded=True,
            class_names=["Vin sucré", "Vin équilibré", "Vin amer"],
            fontsize=6,
            ax=axes[i]  # Utiliser un sous-plot
        )
        axes[i].set_title(f"Arbre {i+1}")

    st.pyplot(fig)
    
    st.markdown("### 📊 Matrice de Confusion")
    cm_display = metrics.ConfusionMatrixDisplay.from_predictions(
        y_test, pipe.predict(X_test)
    )
    fig, ax = plt.subplots(figsize=(3, 2))
    cm_display.plot(ax=ax, cmap="Blues", colorbar=True)
    st.pyplot(fig, use_container_width=False)

    st.markdown("### 📄 Rapport de Classification")
    report = metrics.classification_report(y_test, pipe.predict(X_test))
    st.code(report, language="text")