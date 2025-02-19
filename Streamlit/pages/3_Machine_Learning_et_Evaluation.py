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
st.write("Nous allons maintenant entraîner un arbre de décision sur nos données.")
pipe = pipeline.Pipeline([
    # ("feature_selection", feature_selection),
    # ('std_scaler', preprocessing.StandardScaler()),
    ('decision_tree', tree.DecisionTreeClassifier())]
)
pipe.fit(X_train, y_train)
st.write("Nous allons maintenant entraîner un arbre de décision sur nos données.\n"
         "Nous avons créé et entraîné un pipeline qui contient un arbre de décision.\n"
         "Nous pouvons maintenant l'évaluer.")
st.write("Accuracy on train set =", pipe.score(X_train,y_train), "Accuracy on test set =", pipe.score(X_test,y_test))

fig = plt.figure(figsize=(30,20))
tree.plot_tree(
    pipe[-1],
    feature_names=X_train.columns,
    filled=True,
    rounded=True,
    class_names=["Vin sucré","Vin éuilibré", "Vin amer"],
    fontsize=9
)
plt.savefig("tree_raw.png",bbox_inches="tight")
st.pyplot(fig)