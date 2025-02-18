import streamlit as st
import pandas as pd


df_raw = (
    pd.read_csv("./data/vin.csv", sep=',').iloc[:, 1:]
   
)  

st.title("Exploration Demo")
st.sidebar.header("Exploration Demo")
st.write("This is the first exploration page.")

st.dataframe(df_raw.head(10), use_container_width=True)
