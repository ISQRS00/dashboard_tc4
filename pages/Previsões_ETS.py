import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Configurações do Streamlit
st.set_page_config(page_title="Deploy | Tech Challenge 4 | FIAP", layout='wide')

# Função para calcular WMAPE
def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# Carregar e preparar os dados (com cache)
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/barril.csv', sep=';')
    df.drop(columns=['Unnamed: 2'], inplace=True)
    df.rename(columns={'Data': 'data', 'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'realizado'}, inplace=True)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', dayfirst=True)
    df['realizado'] = df['realizado'].str.replace(',', '.').astype(float)
    df['realizado'] = df['realizado'].ffill()  # Preencher valores ausentes
    return df

# Função para treinar o modelo ETS e salvar em um arquivo joblib
@st.cache_data
def train_ets_model(train_data, season_length=252, model_filename='ets_model.joblib'):
    if os.path.exists(model_filename):
        model_ets = joblib.load(model_filename)  # Carrega o modelo salvo
    else:
        model_ets = sm.tsa.ExponentialSmoothing(train_data['realizado'], seasonal='mul', seasonal_periods=season_length).fit()
        joblib.dump(model_ets, model_filename)  # Salva o modelo treinado em um arquivo
    return model_ets

# Função para previsão com o modelo ETS
def forecast_ets(train, valid, _model_ets):
    forecast_ets = _model_ets.forecast(len(valid))
    forecast_dates = pd.date_range(start=train['data'].iloc[-1] + pd.Timedelta(days=1), periods=len(valid), freq='D')
    ets_df = pd.DataFrame({'data': forecast_dates, 'previsão': forecast_ets})
    ets_df = ets_df.merge(valid, on=['data'], how='inner')

    wmape_ets = wmape(ets_df['realizado'].values, ets_df['previsão'].values)
    MAE_ets = mean_absolute_error(ets_df['realizado'].values, ets_df['previsão'].values)
    MSE_ets = mean_squared_error(ets_df['realizado'].values, ets_df['previsão'].values)
    R2_ets = r2_score(ets_df['realizado'].values, ets_df['previsão'].values)

    return ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets

# Carregar dados
df_barril_petroleo = load_data()

# Explicação sobre o corte de dados
st.write("""
### Como Funciona a Divisão dos Dados?

O modelo de previsão ETS é treinado com dados históricos do preço do petróleo e funciona com dois conjuntos principais:
- **Treinamento:** Utilizado para identificar padrões históricos, como tendências e sazonalidades.
- **Validação:** Usado para testar o modelo, comparando as previsões com os valores reais.

### Por que Usar um Intervalo de Treino Menor?

Para garantir que o Streamlit funcione de forma eficiente, optamos por treinar o modelo com um intervalo menor de dados históricos. Isso reduz o tempo de processamento sem comprometer a precisão do modelo.

⚠️ **Nota:** Dados analíticos completos estão disponíveis no Power BI, enquanto aqui focamos em uma amostra representativa para demonstrar a funcionalidade.

### Como Funciona a Separação?

Escolha um número de dias para separar os dados de treino e validação. O modelo será treinado com os dados antes dessa data e testado com os dados após o corte. Experimente diferentes períodos para avaliar o desempenho!
""")




# Input para o número de dias para corte
dias_corte = st.number_input('Selecione o número de dias para o corte entre 7 e 90:', min_value=7, max_value=90, value=7)

# Calcular a data de corte com base no número de dias
cut_date = df_barril_petroleo['data'].iloc[-dias_corte]

# Dividir em treino e validação
train = f_barril_petroleo.loc[(df_barril_petroleo['data'] >= "2019-01-01") & (df_bdarril_petroleo['data'] < cut_date)]
valid = df_barril_petroleo.loc[df_barril_petroleo['data'] >= cut_date]

# Exibir o tamanho dos conjuntos
st.write(f"A data inicial da validação: {cut_date.strftime('%d-%m-%Y')}")
st.write(f"Tamanho do conjunto de treino: {len(train)} dias")
st.write(f"Tamanho do conjunto de validação: {len(valid)} dias")

# Criar uma barra de progresso
progress = st.progress(0)

# Treinar o modelo ETS
if st.button("Gerar Previsão"):
    # Atualizar barra de progresso para indicar que o treinamento está em andamento
    progress.progress(100)
    
    # Treinar o modelo
    model_ets = train_ets_model(train)  # Isso carregará ou treinará e salvará o modelo
    
    # Prever com o modelo ETS
    ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets = forecast_ets(train, valid, model_ets)

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
        xaxis=dict(tickformat="%d-%m-%Y", tickangle=45),
        yaxis=dict(title="Valor (US$)", tickformat=".3f"),
        autosize=True
    )
    st.plotly_chart(fig_ets, use_container_width=True)
