import streamlit as st

# Plotting_Demo_page = st.Page("1_📈_Plotting_Demo.py", title="Plotting_Demo")
# Mapping_Demo_page = st.Page("2_🌍_Mapping_Demo.py", title="Mapping_Demo")
# DataFrame_Demo_page = st.Page("3_📊_DataFrame_Demo.py", title="DataFrame_Demo")
Exploration_page = st.Page("1_Exploration.py", title="Exploration_Demo")

# pg = st.navigation([Plotting_Demo_page, Mapping_Demo_page, DataFrame_Demo_page])
pg = st.navigation([Exploration_page])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()