import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import model_selection
from src.dataframe import df_cleaned, df_raw_initial

st.set_page_config(
    page_title="Préparation data",
    page_icon="🔧",
    # page_icon=":material/edit:",
    layout="wide",
)

pd.set_option("display.max_colwidth", None)

st.title("🔧 Préparation des données")
st.sidebar.header("Préparation des données")

# Récapitulatif
st.markdown(
    """
        #### Aperçu du jeu de données
    """
)
st.markdown(
    f"Le jeu de données comporte **{df_raw_initial.shape[0]}** lignes et **{df_raw_initial.shape[1]}** colonnes."
)

# Préparation de la target
st.markdown(
    """
        #### Préparation de la target
        - Renommage de la colonne `target` en `target_text`
        - Mappage de la colonne `target_text` en valeurs numériques dans une nouvelle colonne `target_number`
            - Vin sucré -> 0
            - Vin équilibré -> 1
            - Vin amer -> 2
        - Affichage du dataframe
    """
)
df_raw = df_raw_initial.replace(
    to_replace="Vin éuilibré", value="Vin équilibré")
df_modified = df_cleaned
df_modified["target_num"] = df_modified["target_text"].map(
    {"Vin sucré": 0, "Vin équilibré": 1, "Vin amer": 2}
)
df_raw["target"] = df_raw["target"].map(
    {"Vin sucré": 0, "Vin équilibré": 1, "Vin amer": 2}
)

def affichage_sample(number):
    st.markdown(f"##### Affichage partiel d'un sample aléatoire de {number} valeurs")
    df_sample = df_modified.sample(number)
    st.dataframe(df_sample, use_container_width=False)
    st.markdown("##### Extrait des 2 dernières colonnes")
    st.dataframe(
        df_sample[["target_text", "target_num"]],
        use_container_width=False,
    )

st.markdown("#### Affichage d'un extrait des données")
cleaned_columns = df_raw.columns
# columns = []
nb_cols=len(cleaned_columns)
# # st.write(f"nb cols: {nb_cols}")

# TODO: à compléter: sélection des colonnes
# nb_rows = nb_cols//2 if nb_cols%2==0 else (nb_cols+1)//2
# st.write(f"nb rows: {nb_rows}")
# checkboxes=[]
# for i in range(0, nb_rows):
#     cols = st.columns(2)
#     for j in range(0, 2):
#         col_index = i*2+j
#         col_name = cleaned_columns[col_index]
#         cols[j].checkbox(col_name, value=True, key=col_name, on_change=None)
#         checkboxes.append(cols[j])

max_val = min(df_raw_initial.shape[0], 200)
with st.form("num_sample_form"):
    col1, col2 = st.columns([1, 1], vertical_alignment="bottom")

    with col1:
        num_sample = st.number_input(
            "Combien de lignes souhaitez-vous afficher (1 à 200)?",
            min_value=1,
            max_value=max_val,
            step=1,
            value=10,
            placeholder="Entrez un nombre",
            key="num_sample",
        )
        num_sample = int(num_sample)

    with col2:
        button_num_sample = st.form_submit_button("Valider", type="primary")

    if button_num_sample:
        affichage_sample(num_sample)


##############################
# DIVISION DU JEU DE DONNÉES #
##############################
st.markdown(
    """
        #### Division du jeu de données
    """
)
target = ["target"]
features = [col for col in df_raw.columns if col not in target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df_raw[features], df_raw[target], test_size=0.2, random_state=13
)

df_y_train = pd.DataFrame(y_train)
df_y_test = pd.DataFrame(y_test)

st.markdown("""
            Le jeu d'entraînement contient 80% des données.
            Le jeu de test contient 20% des données.
            """)

divisions = st.columns(2)
md_text_0 = f"###### Données d'entraînement (80%, soit {df_y_train.shape[0]} lignes)"
divisions[0].markdown(md_text_0)
divisions[0].write(y_train[target].value_counts(normalize=True))

md_text_1 = f"###### Données de test (80%, soit {df_y_test.shape[0]} lignes)"
divisions[1].markdown(md_text_1)
divisions[1].write(y_test[target].value_counts(normalize=True))


# TODO: A continuer s'il y a des outliers et
# TODO: si on souhaite effectuer une sélection de features
has_outliers = False
if has_outliers:
    st.markdown(
        """
        #### Feature Engineering
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
