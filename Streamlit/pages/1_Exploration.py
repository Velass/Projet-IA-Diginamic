import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
df_raw = (
    pd.read_csv("./data/vin.csv", sep=',').iloc[:, 1:]
   
)  
st.image("./Streamlit/images/Ren And Stimpy Cartoon GIF.gif")  
st.title("Exploration des données - dataset Vin")
st.sidebar.header("Exploration des données ➡️ ")
with st.expander("Explorer toutes les données", expanded=False):
    st.dataframe(df_raw, use_container_width=True, height=500)
with st.expander("Voir un échantillon aléatoire", expanded=False):
    st.dataframe(df_raw.sample(5), use_container_width=True)
with st.expander("Voir les valeurs manquantes", expanded=False):
    df_valeurs_nulles = df_raw.isnull().sum().reset_index().T
    st.dataframe(df_valeurs_nulles)
with st.expander("Voir les types de données", expanded=False):
    df_types = pd.DataFrame(df_raw.dtypes, columns=["Type de données"])  
    st.dataframe(df_types.T)  
with st.expander("Afficher les statistiques descriptives des données numériques", expanded=False):
    df_describe = df_raw.describe().round(2)  
    df_styled = df_describe.style.background_gradient(cmap="Greens")  
    df_styled = df_styled.format(lambda x: '{:.3g}'.format(x)) 
    df_styled = df_styled.set_table_styles(
        [{'selector': 'th', 
        'props': [('font-weight', 'normal'), ('font-size', '12px')]} 
        ]
    )
    st.markdown(df_styled.to_html(), unsafe_allow_html=True)
with st.expander("Histogrammes des distributions des composants", expanded=False):
        num_cols = [col for col in df_raw.select_dtypes(include=['int64', 'float64']).columns if col != "target"]
        if not num_cols:
            st.warning("Aucune colonne numérique à afficher.")
        else:
                    num_features = len(num_cols)
                    cols = 3
                    rows = (num_features // cols) + (num_features % cols)
                    fig, axes = plt.subplots(rows, cols, figsize=(7,2.5 * rows))  
                    axes = axes.flatten() if num_features > 1 else [axes]
                    for i, col in enumerate(num_cols):
                        axes[i].hist(df_raw[col], bins=50, color="green", edgecolor="black")
                        axes[i].set_xlabel(col, fontsize=6)
                        axes[i].set_ylabel("Fréquence", fontsize=6)
                        axes[i].tick_params(axis='x', labelsize=6)
                        
                    for j in range(i + 1, len(axes)):
                        fig.delaxes(axes[j])
                    st.pyplot(fig)
with st.expander("Distribution des valeurs de la variable cible", expanded=False):
        df_hist = df_raw["target"].value_counts(normalize=True).reset_index()  
        df_hist.columns = ["target", "percentage"]
        df_hist["percentage"] = (df_hist["percentage"] * 100).round(0) .astype(int) 
        df_hist["count"] = df_raw["target"].value_counts().values
        fig, ax = plt.subplots(figsize=(4, 3))  
        bar_width = 0.25  
        bars = ax.bar(df_hist['target'], df_hist['count'], width=bar_width, color="green")
        for bar, percent in zip(bars, df_hist["percentage"]):
            ax.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.5, 
                    f"{percent}%", 
                    ha='center', fontsize=6, color="black")
        ax.set_xlabel('Target', fontsize=6) 
        ax.set_ylabel('Count', fontsize=6)  
        ax.tick_params(axis='x', labelsize=5)
        ax.tick_params(axis='y', labelsize=5) 
        plt.xticks(rotation=45, ha='right')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:  
            st.pyplot(fig)
with st.expander("Analyse des relations entre les variables numériques indépendants et la cible", expanded=False):
    pairplot_cols = num_cols[:10]
    pairplot_cols.append("target")
    sns.pairplot(df_raw[pairplot_cols], hue="target")
    plt.show()
    st.pyplot(plt)