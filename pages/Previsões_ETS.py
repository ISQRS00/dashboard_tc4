import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Função para calcular WMAPE
def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# Função para treinar o modelo ETS
def train_ets_model(train_data, season_length=252):
    model_ets = sm.tsa.ExponentialSmoothing(
        train_data['realizado'], 
        seasonal='mul', 
        seasonal_periods=season_length
    ).fit()
    return model_ets

# Carregar e preparar os dados
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/barril.csv', sep=';')
    df.drop(columns=['Unnamed: 2'], inplace=True)
    df.rename(columns={'Data': 'data', 'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'realizado'}, inplace=True)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df['realizado'] = df['realizado'].str.replace(',', '.').astype(float)
    df['realizado'] = df['realizado'].ffill()  # Preencher valores ausentes
    return df

# Dividir dados em treino e validação
def split_data(df, data_corte):
    train = df[df['data'] < data_corte]
    valid = df[df['data'] >= data_corte]
    return train, valid

# Executar previsão com ETS
def forecast_ets(train, valid):
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

# Criar gráfico das previsões
def plot_forecast(ets_df, valid):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid['data'], y=valid['realizado'], mode='lines', name='Realizado'))
    fig.add_trace(go
