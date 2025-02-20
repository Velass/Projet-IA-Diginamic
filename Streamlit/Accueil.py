import streamlit as st

st.set_page_config(
    page_title="Analyse et Prédiction de la catégorie du Vin 🍷",
    page_icon="🍷",
    layout="wide"
)


st.title("🍷 Analyse et Prédiction de la catégorie du Vin")
st.image("./Streamlit/images/wine.jpg")
st.markdown(
    """
    Bienvenue dans cette application de **Machine Learning** dédiée à l'analyse et la classification des vins !  
    Ce projet vise à explorer les caractéristiques des vins, à effectuer des analyses de données et à construire des modèles de classification pour prédire leur catégorie.

    ---
    ### 🔍 Fonctionnalités principales :
    
    📊 **Exploration de données**  
    - Analyse descriptive des caractéristiques du vin  
    - Visualisations : distributions, pairplots, matrice de corrélation  

    🛠 **Préparation des données**  
    - Normalisation des variables  
    - Gestion des valeurs manquantes et des outliers  

    🤖 **Machine Learning**  
    - Séparation des données (train/test)  
    - Entraînement de modèles : Arbre de Décision, Réseau de Neurones, Forêt Aléatoire  

    📈 **Évaluation des modèles**  
    - Précision et autres métriques  
    - Matrice de confusion  

    ---
    **👈 Sélectionnez une section dans le menu latéral pour explorer les différentes étapes du projet !**
    """
)

# Message dans la barre latérale
st.sidebar.success("📌 Sélectionnez une section pour commencer.")
