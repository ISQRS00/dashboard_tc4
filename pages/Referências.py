import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf
from statsforecast import StatsForecast
from plotly.subplots import make_subplots
# Configurações do Streamlit

st.set_page_config(page_title="Deploy | Tech Challenge 4 | FIAP", layout='wide')

st.markdown("""
        ### Referências

        1. **Guerra e petróleo: veja reações mais drásticas da commodity a grandes conflitos**  
           CNN Brasil. Disponível em: [Link](https://www.cnnbrasil.com.br/economia/macroeconomia/guerra-e-petroleo-veja-reacoes-mais-drasticas-da-commodity-a-grandes-conflitos/#:~:text=Guerra%20R%C3%BAssia%2DUcr%C3%A2nia,em%20US%24%20118%2C11).  
           Acesso em: 30 nov. 2024.

        2. **Entenda por que o preço do petróleo disparou com a guerra entre Ucrânia e Rússia**  
           CNN Brasil. Disponível em: [Link](https://www.cnnbrasil.com.br/economia/mercado/entenda-por-que-o-preco-do-petroleo-disparou-com-a-guerra-entre-ucrania-e-russia/).  
           Acesso em: 30 nov. 2024.

        3. **COVID-19 e os impactos sobre o mercado de petróleo**  
           IBP - Instituto Brasileiro de Petróleo e Gás. Disponível em: [Link](https://www.ibp.org.br/observatorio-do-setor/analises/covid-19-e-os-impactos-sobre-o-mercado-de-petroleo/#:~:text=A%20dissemina%C3%A7%C3%A3o%20do%20COVID%2D19,efeitos%20da%20pandemia%20na%20economia).  
           Acesso em: 30 nov. 2024.

        4. **Contexto Mundial e Preço do Petróleo: Uma Visão de Longo Prazo**  
           Empresa de Pesquisa Energética (EPE). Disponível em: [Link](https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-250/topico-302/Contexto%20Mundial%20e%20Pre%C3%A7o%20do%20Petr%C3%B3leo%20Uma%20Vis%C3%A3o%20de%20Longo%20Prazo[1].pdf).  
           Acesso em: 30 nov. 2024.

        5. **Fatores que Influenciam a Formação do Preço do Petróleo**  
           Julia Fernandes Ramos. Disponível em: [Link](https://www.econ.puc-rio.br/uploads/adm/trabalhos/files/Julia_Fernandes_Ramos.pdf).  
           Acesso em: 30 nov. 2024.

        6. **Total energy supply (TES) by source, World, 1990-2022**  
           International Energy Agency (IEA). Disponível em: [Link](https://www.iea.org/data-and-statistics/data-tools/energy-statistics-data-browser?country=WORLD&fuel=Energy%20supply&indicator=TESbySource).  
           Acesso em: 30 nov. 2024.
    """)

def exibir_aba_referencias():
    aba = st.sidebar.radio("Navegação", ["Introdução", "Modelos", "Referências"])
    
    if aba == "Introdução":
        st.title("Introdução")
        st.write("Bem-vindo à análise de preços do petróleo.")
    elif aba == "Modelos":
        st.title("Modelos")
        st.write("Aqui estão os modelos testados.")
    elif aba == "Referências":
        st.title("Referências")
       
