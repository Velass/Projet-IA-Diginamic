import pandas as pd
import streamlit as st

# Lecture du CSV et transformation en DataFrame
df_raw_initial = pd.read_csv("./data/vin.csv", sep=",").iloc[:, 1:]

df_cleaned = df_raw_initial.rename(columns={"target": "target_text"})
df_cleaned.replace(to_replace="Vin éuilibré", value="Vin équilibré", inplace=True)

CLEANED_COLUMNS = df_raw_initial.columns  # récupérer toutes les colonnes du dataframe

__all__ = ["df_raw_initial", "df_cleaned", "CLEANED_COLUMNS"]
