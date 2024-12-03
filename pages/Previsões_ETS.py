import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configurações do Streamlit
st.set_page_config(page_title="Deploy | Tech Challenge 4 | FIAP", layout='wide')

# Função para calcular WMAPE
@st.cache_data
def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# Função para carregar previsões precalculadas
@st.cache_data
def load_forecasts():
    return pd.read_csv('previsoes_ets_precalculadas.csv')

# Função para precalcular previsões e salvar em CSV
def precalculate_forecasts(df, season_length=252, dias_corte=90):
    # Dividir em treino e validação
    cut_date = df['data'].max() - pd.Timedelta(days=dias_corte)
    train = df.loc[df['data'] < cut_date]
    valid = df.loc[df['data'] >= cut_date]

    # Treinar o modelo ETS
    model_ets = sm.tsa.ExponentialSmoothing(
        train['realizado'], seasonal='mul', seasonal_periods=season_length
    ).fit()

    # Prever com o modelo ETS
    forecast_ets = model_ets.forecast(len(valid))
    forecast_dates = pd.date_range(start=train['data'].iloc[-1] + pd.Timedelta(days=1), periods=len(valid), freq='D')

    # Criar DataFrame de Previsões
    ets_df = pd.DataFrame({'data': forecast_dates, 'previsão': forecast_ets})
    ets_df = ets_df.merge(valid, on=['data'], how='inner')

    # Salvar previsões em um arquivo CSV
    ets_df.to_csv('previsoes_ets_precalculadas.csv', index=False)
    print("Previsões precalculadas salvas em 'previsoes_ets_precalculadas.csv'.")

# Carregar os dados do barril de petróleo
@st.cache_data
def load_data():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/barril.csv', sep=';'
    )
    df.rename(columns={
        'Data': 'data', 
        'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'realizado'
    }, inplace=True)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', dayfirst=True)
    df['realizado'] = df['realizado'].str.replace(',', '.').astype(float)
    df['realizado'] = df['realizado'].ffill()  # Preencher valores ausentes
    return df

# 1. Carregar os dados
df_barril_petroleo = load_data()

# 2. Precalcular previsões se necessário
if st.sidebar.button("Precalcular Previsões"):
    st.sidebar.write("Precalculando previsões... Aguarde.")
    precalculate_forecasts(df_barril_petroleo)
    st.sidebar.success("Previsões recalculadas e salvas!")

# 3. Carregar previsões precalculadas
try:
    forecasts = load_forecasts()

    # Exibir métricas no Streamlit
    st.subheader('Métricas de Desempenho do Modelo ETS')
    wmape_value = wmape(forecasts['realizado'], forecasts['previsão'])
    mae_value = mean_absolute_error(forecasts['realizado'], forecasts['previsão'])
    mse_value = mean_squared_error(forecasts['realizado'], forecasts['previsão'])
    r2_value = r2_score(forecasts['realizado'], forecasts['previsão'])

    st.write(f"WMAPE: {wmape_value:.2%}")
    st.write(f"MAE: {mae_value:.3f}")
    st.write(f"MSE: {mse_value:.4f}")
    st.write(f"R²: {r2_value:.2f}")

    # Criar gráfico de previsões
    st.subheader("Previsão do Modelo ETS")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecasts['data'], y=forecasts['realizado'], mode='lines', name='Realizado'))
    fig.add_trace(go.Scatter(x=forecasts['data'], y=forecasts['previsão'], mode='lines', name='Previsão'))
    fig.update_layout(
        title="Previsões Precalculadas",
        xaxis_title="Data",
        yaxis_title="Valor do Petróleo (US$)",
        xaxis=dict(tickformat="%d-%m-%Y", tickangle=45),
        yaxis=dict(title="Valor (US$)", tickformat=".3f"),
        autosize=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Botão para baixar previsões
    st.subheader('Baixar Resultados')
    csv = forecasts.to_csv(index=False)
    st.download_button(
        label="Baixar Previsões ETS",
        data=csv,
        file_name="previsoes_ets.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.error("O arquivo 'previsoes_ets_precalculadas.csv' não foi encontrado. Clique no botão de precálculo para gerá-lo.")
