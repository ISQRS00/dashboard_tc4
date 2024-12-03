import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Função para calcular WMAPE
def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# Função para treinar o modelo ETS com cache
@st.cache_data(show_spinner=True)
def train_ets_model(train_data, season_length=252):
    model_ets = sm.tsa.ExponentialSmoothing(
        train_data['realizado'], 
        seasonal='mul', 
        seasonal_periods=season_length
    ).fit()
    return model_ets

# Configurações do Streamlit
st.set_page_config(page_title="Deploy | Tech Challenge 4 | FIAP", layout='wide')

# Carregar e preparar os dados
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/barril.csv', sep=';')
    df.drop(columns=['Unnamed: 2'], inplace=True)
    df.rename(columns={'Data': 'data', 'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'realizado'}, inplace=True)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df['realizado'] = df['realizado'].str.replace(',', '.').astype(float)
    df['realizado'] = df['realizado'].ffill()  # Preencher valores ausentes
    return df

df_barril_petroleo = load_data()

# Adicionar seletor de intervalo de datas com limite
st.subheader('Selecione o intervalo de datas')
start_date = st.date_input('Data inicial', df_barril_petroleo['data'].min())
end_date = st.date_input('Data final', min(df_barril_petroleo['data'].max(), pd.to_datetime('2024-12-31')))

# Filtrar os dados com base no intervalo selecionado
filtered_data = df_barril_petroleo[(df_barril_petroleo['data'] >= pd.to_datetime(start_date)) & (df_barril_petroleo['data'] <= pd.to_datetime(end_date))]

# Definir
