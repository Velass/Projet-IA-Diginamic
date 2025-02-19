import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import linear_model, metrics, model_selection, pipeline, tree

pd.set_option("display.max_colwidth", None)

# Lecture du CSV et transformation en DataFrame
df_raw = pd.read_csv("./data/vin.csv", sep=",").iloc[:, 1:]


st.title("🔧 Préparation des données")
st.sidebar.header("Préparation des données")

# Préparation de la target
st.markdown("""
            #### Préparation de la target
            - Correction du mot `éuilibré` -> `équilibré`
            - Renommage de la colonne `target` en `target_text`
            - Mappage de la colonne `target_text` en valeurs numériques dans une nouvelle colonne `target_number`
                - Vin sucré -> 0
                - Vin équilibré -> 1
                - Vin amer -> 2
            - Affichage partiel d'un sample aléatoire de 5 valeurs
            # """)
df_raw.replace(to_replace="Vin éuilibré",value="Vin équilibré",inplace=True)
df_raw.rename(columns={"target": "target_text"},inplace=True)
df_raw["target_num"] = df_raw["target_text"].map(
    {"Vin sucré": 0, "Vin équilibré": 1, "Vin amer": 2}
)
st.dataframe(df_raw[["target_text", "target_num"]].sample(5), use_container_width=False)

st.markdown("""
            #### Division du jeu de données
            """)
target=["target_text", "target_num"]
features=[col for col in df_raw.columns if col not in target]

X_train, X_test, y_train, y_test = (
    model_selection.train_test_split(
        df_raw[features],
        df_raw[target],
        test_size=0.2,
        random_state=13
    )
)

df_proportions = pd.DataFrame(y_train[target].value_counts(normalize=True),columns=["target_text", "target_num", "proportion"])
df_proportions["proportion"].apply(lambda x: f"{round(x*100,2)} %")

print((y_train[target].value_counts(normalize=True)))

st.write(y_train[target].value_counts(normalize=True))

st.write(df_proportions)
# st.write(proportions)

# st.write("This is the first exploration page.")


# st.dataframe(df_raw.head(10), use_container_width=True)
