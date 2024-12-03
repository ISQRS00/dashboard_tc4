import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf
from statsforecast import StatsForecast
from plotly.subplots import make_subplots

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

# Adicionar seletor de intervalo de datas
st.subheader('Selecione o intervalo de datas')
start_date = st.date_input('Data inicial', df_barril_petroleo['data'].min())
end_date = st.date_input('Data final', df_barril_petroleo['data'].max())

# Filtrar os dados com base no intervalo selecionado
filtered_data = df_barril_petroleo[(df_barril_petroleo['data'] >= pd.to_datetime(start_date)) & (df_barril_petroleo['data'] <= pd.to_datetime(end_date))]

# Definir a data de corte fixa (exemplo: 1º de outubro de 2024)
data_corte = pd.to_datetime('2024-10-01')

# Dividir em treino e validação com base na data de corte
train = filtered_data.loc[filtered_data['data'] < data_corte]
valid = filtered_data.loc[filtered_data['data'] >= data_corte]

# Função para previsão com o modelo ETS
@st.cache_data
def forecast_ets(train, valid):
    with st.spinner('Treinando o modelo e gerando previsões...'):
        season_length = 252  # Sazonalidade anual
        model_ets = train_ets_model(train)
        forecast_ets = model_ets.forecast(len(valid))
        forecast_dates = pd.date_range(start=train['data'].iloc[-1] + pd.Timedelta(days=1), periods=len(valid), freq='D')
        ets_df = pd.DataFrame({'data': forecast_dates, 'previsão': forecast_ets})
        ets_df = ets_df.merge(valid, on=['data'], how='inner')

        # Calcular métricas de desempenho
        wmape_ets = wmape(ets_df['realizado'].values, ets_df['previsão'].values)
        MAE_ets = mean_absolute_error(ets_df['realizado'].values, ets_df['previsão'].values)
        MSE_ets = mean_squared_error(ets_df['realizado'].values, ets_df['previsão'].values)
        R2_ets = r2_score(ets_df['realizado'].values, ets_df['previsão'].values)

    return ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets

# Exibir as métricas de desempenho
ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets = forecast_ets(train, valid)

st.subheader('Métricas de Desempenho do Modelo ETS')
st.write(f'WMAPE: {wmape_ets:.2%}')
st.write(f'MAE: {MAE_ets:.3f}')
st.write(f'MSE: {MSE_ets:.4f}')
st.write(f'R²: {R2_ets:.2f}')

# Criar gráfico ETS
fig_ets = go.Figure()
fig_ets.add_trace(go.Scatter(x=valid['data'], y=valid['realizado'], mode='lines', name='Realizado'))
fig_ets.add_trace(go.Scatter(x=ets_df['data'], y=ets_df['previsão'], mode='lines', name='Forecast'))
fig_ets.update_layout(
    title="Previsão do Modelo ETS",
    xaxis_title="Dia-Mês",
    yaxis_title="Valor do Petróleo (US$)",
    xaxis=dict(
        tickformat="%d-%m-%Y", 
        type="date", 
        dtick="D1", 
        nticks=30, 
        tickangle=45  # Inclinar as datas em 45 graus
    ),
    yaxis=dict(
        title="Valor (US$)", 
        tickformat=".3f"
    ),
    autosize=True
)

# Exibição do gráfico
st.plotly_chart(fig_ets)

# Adicionar descrição do gráfico
st.write("""
**Descrição do Gráfico:** Este gráfico compara os valores reais do preço do petróleo (Realizado) com as previsões geradas pelo modelo ETS (Forecast). 
O modelo ETS utiliza a sazonalidade anual para gerar previsões e busca capturar tendências e padrões nos dados históricos.
""")

# Explicação sobre o download do arquivo
st.write("""
### Baixar os Resultados da Previsão
Após gerar as previsões com o modelo ETS, você pode baixar um arquivo `.csv` contendo as **previsões** feitas pelo modelo para os próximos dias. O arquivo inclui a data e o valor previsto do preço do petróleo, que pode ser útil para análises futuras ou para comparar com os dados reais posteriormente.
Clique no botão abaixo para baixar as previsões em formato CSV.
""")

# Opção de Download dos Resultados
st.subheader('Baixar Resultados')
ets_df['previsão'] = ets_df['previsão'].round(2)
csv = ets_df.to_csv(index=False)
st.download_button(
    label="Baixar Previsões ETS",
    data=csv,
    file_name="previsoes_ets.csv",
    mime="text/csv"
)
