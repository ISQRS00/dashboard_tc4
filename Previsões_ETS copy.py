import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from model_lstm.model_lstm import predict, predict_dates, load_model_and_scaler, load_and_process_data, evaluate_lstm_model, create_sequences

# Função para calcular WMAPE
def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# Função para treinar o modelo ETS (treinamento otimizado)
@st.cache_data
def train_ets_model(train_data):
    season_length = 252  # Sazonalidade anual
    model_ets = sm.tsa.ExponentialSmoothing(train_data['realizado'], seasonal='mul', seasonal_periods=season_length).fit(optimized=True)
    return model_ets

# Configurações do Streamlit
st.set_page_config(page_title="Deploy | Tech Challenge 4 | FIAP", layout='wide')

# Carregar e preparar os dados (com cache)
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/barril.csv', sep=';')
    df.drop(columns=['Unnamed: 2'], inplace=True)
    df.rename(columns={'Data': 'data', 'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'realizado'}, inplace=True)
    df['data'] = pd.to_datetime(df['data'])
    df['realizado'] = df['realizado'].str.replace(',', '.').astype(float)
    df['realizado'] = df['realizado'].ffill()  # Preencher valores ausentes
    return df

# Carregar dados uma vez com cache
df_barril_petroleo = load_data()

# Explicação sobre o corte de dados
st.write("""
### Como Funciona o Corte dos Dados?

O modelo de previsão ETS é treinado com base em dados históricos do preço do petróleo. Para avaliar a precisão do modelo, os dados são divididos em duas partes: 
1. **Dados de Treinamento**: São usados para treinar o modelo e entender os padrões históricos.
2. **Dados de Validação**: São usados para testar a performance do modelo e avaliar suas previsões.

O **número de dias de corte** define a quantidade de dados mais recentes que serão usados para o teste do modelo. Ou seja, o modelo será treinado com os dados até um ponto específico e testado com os dados após esse ponto.
""")

# Input para o número de dias para corte
dias_corte = st.number_input('Selecione o número de dias para o corte entre 7 e 90:', min_value=7, max_value=90, value=7)

# Calcular a data de corte com base no número de dias
cut_date = df_barril_petroleo['data'].max() - timedelta(days=dias_corte)

# Dividir em treino e validação
train = df_barril_petroleo.loc[df_barril_petroleo['data'] < cut_date]
valid = df_barril_petroleo.loc[df_barril_petroleo['data'] >= cut_date]

# Treinando o modelo ETS com dados de treino
model_ets = train_ets_model(train)

# Função para previsão com o modelo ETS
def forecast_ets(train, valid):
    season_length = 252  # Sazonalidade anual
    model_ets = sm.tsa.ExponentialSmoothing(train['realizado'], seasonal='mul', seasonal_periods=season_length).fit()
    forecast_ets = model_ets.forecast(len(valid))
    forecast_dates = pd.date_range(start=train['data'].iloc[-1] + pd.Timedelta(days=1), periods=len(valid), freq='D')
    ets_df = pd.DataFrame({'data': forecast_dates, 'previsão': forecast_ets})
    ets_df = ets_df.merge(valid, on=['data'], how='inner')

    wmape_ets = wmape(ets_df['realizado'].values, ets_df['previsão'].values)
    MAE_ets = mean_absolute_error(ets_df['realizado'].values, ets_df['previsão'].values)
    MSE_ets = mean_squared_error(ets_df['realizado'].values, ets_df['previsão'].values)
    R2_ets = r2_score(ets_df['realizado'].values, ets_df['previsão'].values)
    
    return ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets

# Exibição das métricas de desempenho
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
    xaxis_title="Data",
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

# Baixar os resultados da previsão ETS
st.write("""
### Baixar os Resultados da Previsão

Clique no botão abaixo para baixar as previsões em formato CSV.
""")
ets_df['previsão'] = ets_df['previsão'].round(2)
csv_ets = ets_df.to_csv(index=False)

st.download_button(
    label="Baixar Previsões ETS",
    data=csv_ets,
    file_name="previsoes_ets.csv",
    mime="text/csv"
)

# Iniciar carregamento de dados e LSTM
st.write("### Previsão com o Modelo LSTM")

# Carregar e preparar dados para o modelo LSTM
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'model_lstm', 'model_lstm.h5')
scaler_path = os.path.join(base_dir, 'model_lstm', 'scaler.joblib')
excel_path = os.path.join(base_dir, 'dataset', 'Petroleo.xlsx')

# Carregar modelo e scaler
try:
    model_lstm, scaler = load_model_and_scaler(model_path, scaler_path)
except FileNotFoundError:
    st.error("Modelo ou scaler não encontrado. Verifique os caminhos configurados.")
    st.stop()

# Carregar dados e preprocessar
data_corte = pd.to_datetime('2020-05-03')
df_lstm, data_scaled, _ = load_and_process_data(excel_path, data_corte)

# Configurar previsão LSTM
DATA_INICIAL = date(2024, 5, 20)
LIMITE_DIAS = 15

min_date = DATA_INICIAL + timedelta(days=1)
max_date = DATA_INICIAL + timedelta(days=LIMITE_DIAS)
end_date = st.date_input(
    "**Escolha a data de previsão (LSTM):**", 
    min_value=min_date, 
    max_value=max_date,
    value=min_date,
)

# Calcular o número de dias para a previsão com base na data selecionada
days = (end_date - DATA_INICIAL).days

sequence_length = 10

if st.button('Prever LSTM'):
    with st.spinner('Realizando a previsão com LSTM...'):
        try:
            forecast = predict(days, data_scaled, sequence_length)
            forecast_dates = predict_dates(days, df_lstm)

            train_size = int(len(data_scaled) * 0.8)
            X_test, y_test = create_sequences(data_scaled[train_size:], sequence_length)
            r2_lstm, mse_lstm, mae_lstm, mape_lstm, rmse_l
