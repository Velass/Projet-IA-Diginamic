import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import (
    pipeline,
    metrics,
    linear_model,
    model_selection,
    tree
)


# preparation des jeux de données

df_raw = (
    pd.read_csv("./data/vin.csv", sep=',').iloc[:, 1:]
   
) 

df_raw["target"] = (
    df_raw["target"]
    .map({'Vin sucré': 0, 'Vin éuilibré':1, 'Vin amer':2})
)

target = ["target"]
features = [col for col in df_raw.columns if col not in target]

X_train, X_test, y_train, y_test = (
    model_selection.train_test_split(
        df_raw[features],
        df_raw[target],
        test_size=0.2,
        random_state=42
    )
)


st.title("Machine Learning et Evaluation")
st.sidebar.header("Machine Learning et Evaluation")
st.markdown("## données pret.")
st.write("nous pouvons donc voir que les données sont prêtes pour le machine learning et ci-dessous les different pourcentage pour notre target.")
st.write(y_train["target"].value_counts(normalize=True))

st.markdown("## Entraînement d'un arbre de décision")
pipe = pipeline.Pipeline([
    # ("feature_selection", feature_selection),
    # ('std_scaler', preprocessing.StandardScaler()),
    ('decision_tree', tree.DecisionTreeClassifier())]
)
pipe.fit(X_train, y_train)
st.write("Nous allons maintenant entraîner un arbre de décision sur nos données.\n"
         "Nous avons créé et entraîné un pipeline qui contient un arbre de décision.\n"
         "Nous pouvons maintenant l'évaluer.")
train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)

st.markdown(f"""
**📊 Accuracy sur le train set :** `{train_acc:.4f}`  
**📊 Accuracy sur le test set :** `{test_acc:.4f}`
""")

st.markdown("### 🌳 Visualisation de l'Arbre de Décision")

fig, ax = plt.subplots(figsize=(12, 6))
tree.plot_tree(
    pipe.named_steps["decision_tree"],  # Utilisation du bon index dans le pipeline
    feature_names=X_train.columns,
    filled=True,
    rounded=True,
    class_names=["Vin sucré", "Vin équilibré", "Vin amer"],
    fontsize=8,
    ax=ax
)

st.pyplot(fig)

st.markdown("### 📊 Matrice de Confusion")
cm_display = metrics.ConfusionMatrixDisplay.from_predictions(
    y_test, pipe.predict(X_test)
)
fig, ax = plt.subplots(figsize=(6, 4))
cm_display.plot(ax=ax, cmap="Blues", colorbar=True)
st.pyplot(fig)

st.markdown("### 📄 Rapport de Classification")
report = metrics.classification_report(y_test, pipe.predict(X_test))
st.text(report)
