import streamlit as st
from tabs.intro.introducao_tab import IntroTab
from tabs.intro.metodologia_tab import MetodologiaTab
from tabs.intro.objetivo_tab import ObjetivoTab 
from util.layout import output_layout

st.set_page_config(page_title="Introdução | Tech Challenge 4 | FIAP", layout='wide')
output_layout()

with st.container():
    st.header(':orange[FIAP PÓS TECH – DATA ANALYTICS, 2024]')

    tab0, tab1, tab2 = st.tabs(tabs=['Introdução', 'Objetivo', 'Metodologia'])

    IntroTab(tab0)
    ObjetivoTab(tab1)
    MetodologiaTab(tab2)
    