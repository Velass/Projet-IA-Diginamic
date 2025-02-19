import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


 
df_raw = (
    pd.read_csv("./data/vin.csv", sep=',').iloc[:, 1:]
   
)  
  
st.title("Exploration des données - dataset Vin")
st.sidebar.header("Exploration des données")
st.markdown("**1. Affichage des données du dataset Vin**")
st.dataframe(df_raw, use_container_width=True, height=500)
st.markdown("**2. Affichage d'un échantillon aléatoire**")
st.dataframe(df_raw.sample(5))
st.markdown("**3. Affichage de valeurs manquantes**")
df_valeurs_nulles = df_raw.isnull().sum().reset_index().T
st.dataframe(df_valeurs_nulles)
st.markdown("**4. Affichage des types de données**")
df_types = pd.DataFrame(df_raw.dtypes, columns=["Type de données"])  
st.dataframe(df_types.T)  
st.markdown("**5.Statistiques descriptives des données numériques**")
df_describe=df_raw.describe()
df_styled = df_describe.style.background_gradient(cmap="Blues")
st.markdown(df_styled.to_html(), unsafe_allow_html=True)
# st.markdown("**5. Affichage des index**")
# st.dataframe(df_raw.index)
st.markdown("**6.Affichage des dernières lignes**")
df_tail= pd.DataFrame(df_raw)
df_t=df_tail.tail()
st.dataframe(df_t)
st.markdown("### 7.Statistiques descriptives des données")
st.dataframe(df_raw.describe().T.round(3).style.background_gradient())

st.markdown("8.Histogrammes des colonnes numériques")
# num_cols = [col for col in df_raw.columns if col !="target"]
# for col in num_cols:
#     st.markdown(f"### Histogramme de `{col}`")
#     fig, ax = plt.subplots()
#     df_raw[[col]].plot.hist(bins=50)
#     st.pyplot(fig)

num_cols = [col for col in df_raw.select_dtypes(include=['int64', 'float64']).columns if col != "target"]

if not num_cols:
    st.warning("Aucune colonne numérique à afficher.")
else:
    for col in num_cols:
        st.markdown(f" Histogramme de `{col}`")
        with st.expander(f"📊`{col}`", expanded=False):
            fig, ax = plt.subplots(figsize=(4, 3))  
            ax.hist(df_raw[col], bins=50, color="skyblue", edgecolor="black")
            ax.set_xlabel(col)
            ax.set_ylabel("Fréquence")
            ax.set_title(f"Distribution de {col}")
            st.pyplot(fig)  

st.markdown("**9.Distribution des valeurs de la variable cible")
df_hist= df_raw["target"].value_counts().reset_index()
df_hist.columns = ["target", "count"]
st.bar_chart(df_hist.set_index("target"))

st.markdown("**10.Analyse des relations entre les variables numériques indépendants et la cible")
pairplot_cols = num_cols[:10]
pairplot_cols.append("target")
sns.pairplot(df_raw[pairplot_cols], hue="target")
plt.show()
st.pyplot(plt)