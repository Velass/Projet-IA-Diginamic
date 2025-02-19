import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import linear_model, metrics, model_selection, pipeline, tree

st.set_page_config(
    page_title="Préparation data",
    page_icon="🔧",
    # page_icon=":material/edit:",
    layout="wide",
)

pd.set_option("display.max_colwidth", None)

# Lecture du CSV et transformation en DataFrame
df_raw = pd.read_csv("./data/vin.csv", sep=",").iloc[:, 1:]


st.title("🔧 Préparation des données")
st.sidebar.header("Préparation des données")

# Préparation de la target
st.markdown(
    """
        #### Préparation de la target
        - Correction du mot `éuilibré` -> `équilibré`
        - Renommage de la colonne `target` en `target_text`
        - Mappage de la colonne `target_text` en valeurs numériques dans une nouvelle colonne `target_number`
            - Vin sucré -> 0
            - Vin équilibré -> 1
            - Vin amer -> 2
        - Affichage partiel d'un sample aléatoire de 5 valeurs
    """
)
df_raw.replace(to_replace="Vin éuilibré", value="Vin équilibré", inplace=True)
df_modified=df_raw.rename(columns={"target": "target_text"})
df_modified["target_num"] = df_modified["target_text"].map(
    {"Vin sucré": 0, "Vin équilibré": 1, "Vin amer": 2}
)
df_raw["target"] = df_raw["target"].map(
    {"Vin sucré": 0, "Vin équilibré": 1, "Vin amer": 2}
)
st.dataframe(df_modified[["target_text", "target_num"]].sample(10), use_container_width=False)

st.markdown(
    """
        #### Division du jeu de données
    """
)
# target=["target_text", "target_num"]
target = ["target"]
features = [col for col in df_raw.columns if col not in target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df_raw[features], df_raw[target], test_size=0.2, random_state=13
)

# df_proportions = pd.DataFrame(y_train[target].value_counts(normalize=True),columns=["target_text", "target_num", "proportion"])
# df_proportions["proportion"].apply(lambda x: f"{round(x*100,2)} %")

# print((y_train[target].value_counts(normalize=True)))

st.write(y_train[target].value_counts(normalize=True))

# st.write(df_proportions)
# st.write(proportions)


st.markdown(
    """
        #### Feature Engineering
    """
)



has_outliers = False
if has_outliers:
    st.markdown(
        """
        ##### Outliers
        """
    )

has_features_selection = False
if has_features_selection:
    st.markdown(
        """
        ##### Sélection de features
        """
    )

    st.markdown(
        """
        ##### Normalisation des features
        """
    )


st.markdown(
    """
    ### Entraînement d'un arbre de décision
    """
)
pipe = pipeline.Pipeline(
    [
        # ("feature_selection", feature_selection),
        # ('std_scaler', preprocessing.StandardScaler()),
        ("decision_tree", tree.DecisionTreeClassifier())
    ]
)

pipe.fit(X_train, y_train)

st.write("Affichage de l'arbre de décision")
fig = plt.figure(figsize=(30, 20))
tree.plot_tree(
    pipe[-1],
    feature_names=X_train.columns,
    filled=True,
    rounded=True,
    class_names=["Vin sucré", "Vin éuilibré", "Vin amer"],
    fontsize=9,
)
plt.savefig("tree_raw.png", bbox_inches="tight")
st.pyplot(fig)

st.markdown(
    """
    #### Evaluation du modèle
    """
)

st.write("Accuracy on train set =", pipe.score(X_train, y_train))
st.write("Accuracy on test set =", pipe.score(X_test, y_test))

# En pourcentage
st.write("Accuracy on train set =", f"{(pipe.score(X_train, y_train) * 100):.2f} %")
st.write("Accuracy on test set =", f"{(pipe.score(X_test, y_test) * 100):.2f} %")
# st.write("This is the first exploration page.")


# st.dataframe(df_raw.head(10), use_container_width=True)
