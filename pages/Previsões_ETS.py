import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# Configurações do Streamlit
st.set_page_config(page_title="Deploy | Tech Challenge 4 | FIAP", layout='wide')

# Função para calcular WMAPE
@st.cache_data
def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# Carregar dados do petróleo
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/barril.csv', sep=';')
    df.drop(columns=['Unnamed: 2'], inplace=True)
    df.rename(columns={'Data': 'data', 'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'realizado'}, inplace=True)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', dayfirst=True)
    df['realizado'] = df['realizado'].str.replace(',', '.').astype(float)
    df['realizado'] = df['realizado'].ffill()  # Preencher valores ausentes
    return df

# Carregar previsões previamente calculadas
@st.cache_data
def load_forecasts():
    return pd.read_csv('previsoes_ets_precalculadas.csv')

# Carregar dados
df_barril_petroleo = load_data()
forecasts = load_forecasts()

# Explicação sobre o corte de dados
st.write("""
### Como Funciona o Corte dos Dados?

O modelo de previsão ETS é treinado com base em dados históricos do preço do petróleo. 
Escolha o número de dias para o corte e veja como o modelo se comporta para diferentes períodos.
""")

# Input para o número de dias para corte
dias_corte = st.number_input('Selecione o número de dias para o corte entre 7 e 90:', min_value=7, max_value=90, value=7)

# Calcular a data de corte com base no número de dias
cut_date = df_barril_petroleo['data'].max() - timedelta(days=dias_corte)

# Filtrar previsões pré-calculadas para o período selecionado
filtered_forecast = forecasts[forecasts['dias_corte'] == dias_corte]

if not filtered_forecast.empty:
    # Exibir métricas e gráfico
    wmape_ets = filtered_forecast['wmape'].iloc[0]
    MAE_ets = filtered_forecast['mae'].iloc[0]
    MSE_ets = filtered_forecast['mse'].iloc[0]
    R2_ets = filtered_forecast['r2'].iloc[0]
    
    st.subheader('Métricas de Desempenho do Modelo ETS')
    st.write(f'WMAPE: {wmape_ets:.2%}')
    st.write(f'MAE: {MAE_ets:.3f}')
    st.write(f'MSE: {MSE_ets:.4f}')
    st.write(f'R²: {R2_ets:.2f}')

    # Criar gráfico ETS
    fig_ets = go.Figure()
    fig_ets.add_trace(go.Scatter(x=filtered_forecast['data'], y=filtered_forecast['realizado'], mode='lines', name='Realizado'))
    fig_ets.add_trace(go.Scatter(x=filtered_forecast['data'], y=filtered_forecast['previsao'], mode='lines', name='Forecast'))
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
    csv = filtered_forecast.to_csv(index=False)
    st.download_button(
        label="Baixar Previsões ETS",
        data=csv,
        file_name="previsoes_ets.csv",
        mime="text/csv"
    )
else:
    st.warning("Não há previsões para o período selecionado. Por favor, pré-calcule as previsões.")
