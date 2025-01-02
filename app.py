import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Wer talkt mit wem?",
    page_icon="🧊",
    layout="wide",)

@st.cache_data
def load_load_data():
    pass

st.title('Wer talkt mit wem?')


tab1, tab2, tab3 = st.tabs(["Netzwerk", "Ranking", "Personen"])

    with tab1:
        st.header("Netzwerk")
    with tab2:
        st.header("Ranking")
    with tab3:
        st.header("Personen")
        text_search = st.text_input("Suche nach Personen...", value="")
        if text_search:
            st.write(df_search)

