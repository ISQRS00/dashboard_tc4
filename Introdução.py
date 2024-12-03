import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configurações do Streamlit
st.set_page_config(page_title="Deploy | Tech Challenge 4 | FIAP", layout='wide')

# Função para calcular WMAPE
@st.cache_data
def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# Carregar e preparar os dados (com cache)
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/barril.csv', sep=';')
    df.drop(columns=['Unnamed: 2'], inplace=True)
    df.rename(columns={'Data': 'ds', 'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'realizado'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y', dayfirst=True)
    df['realizado'] = df['realizado'].str.replace(',', '.').astype(float)
    df['realizado'] = df['realizado'].fillna(method='ffill') 
    # Preencher valores ausentes
    return df

# Função para treinar o modelo ETS
@st.cache_resource
def train_ets_model(train_data, season_length=252):
    model_ets = sm.tsa.ExponentialSmoothing(train_data['realizado'], seasonal='mul', seasonal_periods=season_length).fit()
    return model_ets

# Função para previsão com o modelo ETS
@st.cache_data
def forecast_ets(train, valid, _model_ets):
    forecast_ets = _model_ets.forecast(len(valid))
    forecast_dates = valid['ds']
    ets_df = pd.DataFrame({'ds': forecast_dates, 'previsão': forecast_ets})
    ets_df = ets_df.merge(valid, on=['ds'], how='inner')

    wmape_ets = wmape(ets_df['realizado'].values, ets_df['previsão'].values)
    MAE_ets = mean_absolute_error(ets_df['realizado'].values, ets_df['previsão'].values)
    MSE_ets = mean_squared_error(ets_df['realizado'].values, ets_df['previsão'].values)
    R2_ets = r2_score(ets_df['realizado'].values, ets_df['previsão'].values)

    return ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets

# Carregar dados
df_barril_petroleo = load_data()

# Seleção de datas para treino e validação
default_train_end = pd.to_datetime('2024-10-01')
train_end_date = st.date_input("Data final para treino (inclusive)", default_train_end)

# Converter train_end_date para datetime64[ns]
train_end_date = pd.to_datetime(train_end_date)

# Filtrar os conjuntos de treino e validação
train = df_barril_petroleo.loc[df_barril_petroleo['ds'] < train_end_date]
valid = df_barril_petroleo.loc[df_barril_petroleo['ds'] >= train_end_date]



# Treinar o modelo ETS (**Bloco inserido aqui**)
model_ets = train_ets_model(train)

# ... (resto do código - previsão, métricas, gráfico, tabela, download)
# Prever com o modelo ETS
ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets = forecast_ets(train, valid, model_ets)

# Exibir métricas
st.subheader('Métricas de Desempenho do Modelo ETS')
st.write(f'WMAPE: {wmape_ets:.2%}')
st.write(f'MAE: {MAE_ets:.3f}')
st.write(f'MSE: {MSE_ets:.4f}')
st.write(f'R²: {R2_ets:.2f}')

# Criar gráfico ETS
fig_ets = go.Figure()
fig_ets.add_trace(go.Scatter(x=valid['ds'], y=valid['realizado'], mode='lines', name='Realizado'))
fig_ets.add_trace(go.Scatter(x=ets_df['ds'], y=ets_df['previsão'], mode='lines', name='Forecast'))
fig_ets.update_layout(
    title="Previsão do Modelo ETS",
    xaxis_title="Data",
    yaxis_title="Valor do Petróleo (US$)",
    xaxis=dict(tickformat="%d-%m-%Y", tickangle=45),
    yaxis=dict(title="Valor (US$)", tickformat=".3f"),
    autosize=True
)
st.plotly_chart(fig_ets, use_container_width=True)

# Opção de Download dos Resultados
st.subheader('Baixar Resultados')
csv = ets_df.to_csv(index=False)
st.download_button(
    label="Baixar Previsões ETS",
    data=csv,
    file_name="previsoes_ets.csv",
    mime="text/csv"
)
