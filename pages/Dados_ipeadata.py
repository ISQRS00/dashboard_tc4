import streamlit as st
import pandas as pd
import time
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

@st.cache_data
def converte_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def mensagem_sucesso():
    sucesso = st.success('Arquivo baixado com sucesso!', icon="✅")
    time.sleep(5)
    sucesso.empty()

st.header("Dados utilizados como base")

# URL do arquivo CSV no GitHub
url = 'https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/barril.csv'

# Lendo o arquivo CSV com separador ";"
dados = pd.read_csv(url, sep=';')

# Renomeando a coluna para "Preço"
dados.rename(columns={dados.columns[1]: 'Preço'}, inplace=True)

# Removendo colunas desnecessárias como "Unnamed: 2"
dados = dados.loc[:, ~dados.columns.str.contains('^Unnamed')]

# Substituindo vírgulas por pontos e convertendo para float
dados['Preço'] = dados['Preço'].str.replace(',', '.').astype(float)

# Convertendo a coluna 'Data' para formato de data (sem hora)
dados['Data'] = pd.to_datetime(dados['Data'], format='%d/%m/%Y').dt.date

with st.expander('Colunas'):
    colunas = st.multiselect('Selecione as colunas', list(dados.columns), list(dados.columns))

st.sidebar.title('Filtros')
with st.sidebar.expander('Preço'):
    preco = st.slider('Selecione o intervalo de preço', float(dados['Preço'].min()), float(dados['Preço'].max()), (float(dados['Preço'].min()), float(dados['Preço'].max())))
with st.sidebar.expander('Data'):
    data_compra = st.date_input('Selecione o intervalo de datas', (dados['Data'].min(), dados['Data'].max()))

query = '''
@preco[0] <= Preço <= @preco[1] and \
@data_compra[0] <= Data <= @data_compra[1]
'''

dados_filtrados = dados.query(query)
dados_filtrados = dados_filtrados[colunas]

st.dataframe(dados_filtrados)

st.markdown(f'A tabela possui :blue[{dados_filtrados.shape[0]}] linhas e :blue[{dados_filtrados.shape[1]}] colunas')

st.markdown('Escreva um nome para o arquivo')
coluna1, coluna2 = st.columns(2)
with coluna1:
    nome_arquivo = st.text_input('', label_visibility='collapsed', value='dados')
    nome_arquivo += '.csv'
with coluna2:
    st.download_button('Fazer o download da tabela em csv', data=converte_csv(dados_filtrados), file_name=nome_arquivo, mime='text/csv', on_click=mensagem_sucesso)
