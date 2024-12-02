import streamlit as st
import requests
from PIL import Image
from io import BytesIO
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

# Função para carregar imagens
def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Dicionário com informações dos insights
insights = [
    {
        "title": "**Insight 1:**",
        "description": """
            No ponto marcado no gráfico, tivemos um pico que teve como catalisador um evento geopolítico: a Guerra na Ucrânia. 
            Como consequência dessa invasão, a Rússia enfrentou severas sanções econômicas, que impactaram diretamente o cenário global de produção de petróleo. 
            Sendo a Rússia um dos maiores produtores mundiais, essas sanções geraram uma redução na oferta de petróleo no mercado internacional.
            Essa instabilidade política resultou em um aumento significativo no preço do barril, refletindo as incertezas e a tensão no panorama energético global.
        """,
        "image_url": "https://github.com/ISQRS00/dashboard_tc4/raw/main/Insight_1.PNG",
    },
    {
        "title": "**Insight 2:**",
        "description": """
            No ponto marcado no gráfico, registramos uma baixa histórica no valor do barril de petróleo, causada pela pandemia de COVID-19. 
            Os lockdowns implementados globalmente reduziram drasticamente a demanda por petróleo, resultando em uma queda acentuada nos preços. 
            Entretanto, logo em seguida, ocorreu uma recuperação significativa. 
            Isso foi impulsionado por cortes na produção, liderados pela OPEP (Organização dos Países Exportadores de Petróleo), aliados ao aumento na demanda com a reabertura gradual das economias.
        """,
        "image_url": "https://github.com/ISQRS00/dashboard_tc4/raw/main/Insight_2.PNG",
    },
    {
        "title": "**Insight 3:**",
        "description": """
            No ponto marcado no gráfico, observamos o maior pico de valor do preço do barril de petróleo. 
            Esse movimento foi impulsionado pelo rápido crescimento de economias emergentes, como China e Índia, que elevaram o consumo de petróleo, 
            além de tensões no Oriente Médio, como a Guerra do Iraque, que afetaram a produção. 
            A especulação financeira nos mercados futuros também contribuiu para a elevação dos preços. 
            Contudo, após atingir o pico, o preço despencou no mesmo ano devido à crise financeira global de 2008, que reduziu drasticamente a demanda mundial por petróleo.
        """,
        "image_url": "https://github.com/ISQRS00/dashboard_tc4/raw/main/Insight_3.PNG",
    },
    {
        "title": "**Insight 4:**",
        "description": """
            A matriz energética é o conjunto de fontes de energia utilizadas para diversas finalidades, como movimentar veículos, preparar alimentos no fogão e gerar eletricidade. 
            Já a matriz elétrica é composta exclusivamente pelas fontes destinadas à geração de energia elétrica.
            A matriz energética mundial ainda possui uma forte dependência do petróleo. 
            Ano após ano, a demanda por essa fonte de energia aumenta. 
            No entanto, em 2020, houve uma queda significativa na demanda, embora os gráficos mostrem uma recuperação de 2021 para 2022.
        """,
        "image_url": "https://github.com/ISQRS00/dashboard_tc4/raw/main/Insight_4.PNG",
    },
]

# Criando a interface do Streamlit
with st.container():
    # Título da aplicação
    st.header("Análises")
    
    # Criando as abas principais
    main_tabs = st.tabs(["Dados Históricos", "Insights", "Motivo da escolha do modelo","Links"])

    # Aba de Insights
    with main_tabs[1]:
        # Exibindo os insights
        for i, insight in enumerate(insights):
            st.subheader(insight["title"])
            st.markdown(insight["description"])
            
            # Exibindo a imagem relacionada ao insight
            img = load_image(insight["image_url"])
            st.image(img, use_container_width=True)

    # Aba de Dados Históricos
    with main_tabs[0]:
        st.subheader("Gráfico de Preço do Barril de Petróleo")
        st.markdown(f"""
            O gráfico mostra a evolução do preço do barril de petróleo ao longo do tempo, destacando os principais picos e quedas de preço. 
            Ele apresenta o preço médio de ($50.3063), com o maior valor registrado de ($143.95) e o menor de ($9.1). 
            Além disso, a análise revela uma tendência de aumento considerável até 2014, seguidos por uma queda acentuada, e uma recuperação gradual nos anos seguintes. 
            O gráfico também aponta para períodos de variação de preço, com picos significativos e quedas acentuadas durante a crise financeira de 2008 e a pandemia de COVID-19.
        """)
    
        # Exibindo a imagem relacionada ao gráfico de preços
        img = load_image("https://github.com/ISQRS00/dashboard_tc4/raw/main/PRECO_LONGO_TEMPO.PNG")
        st.image(img, use_container_width=True)

    # Aba de "Motivo da escolha do modelo"
    with main_tabs[2]:
        st.markdown("""
        ### Análise Comparativa dos Modelos de Previsão de Preços de Petróleo
                    
        """)
        st.markdown("""
        O modelo **ETS** foi o mais eficaz entre os avaliados, superando os demais em todas as métricas de avaliação: WMAPE, MAE, MSE e R². Ele demonstrou maior precisão e capacidade de capturar a dinâmica da série temporal. Portanto, o **ETS** é a escolha recomendada para essa aplicação.
        """)   
        
        # Exibindo as imagens com links diretos
        img_arima = "https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/ARIMA.PNG"
        img_ets = "https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/ETS.PNG"
        img_prophet = "https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/Prophet.PNG"
        st.image(img_arima, use_container_width=True)
        st.image(img_ets, use_container_width=True)
        st.image(img_prophet, use_container_width=True)

        st.markdown("""
        <style>
            table {
                font-size: 18px;
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                padding: 12px;
                text-align: center;
                border: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)

        # Tabela de Comparação de Modelos
        st.markdown("""
        ### Comparação de Modelos de Previsão

        | Métrica       | AutoARIMA   | ETS          | Prophet     | Melhor Modelo  |
        |---------------|-------------|--------------|-------------|----------------|
        | **WMAPE**     | 4.71%       | **3.51%**    | 19.23%      | **ETS**        |
        | **MAE**       | 3.554       | **2.651**    | 14.509      | **ETS**        |
        | **MSE**       | 18.9183     | **12.7823**  | 216.1081    | **ETS**        |
        | **R²**        | -1.88       | **-0.94**    | -31.88      | **ETS**        |
        """, unsafe_allow_html=True)

    # Aba de "Motivo da escolha do modelo"
    with main_tabs[3]:
        st.markdown("""
        ### Links do Nootbook e do PowerBI

        Você pode acessar o notebook que contém o código completo das comparações dos modelos testados no link abaixo:
        [Notebook](https://colab.research.google.com/github/ISQRS00/dashboard_tc4/blob/main/TECH_CHALLENGE_FASE_4.ipynb)
                    
        Você pode baixar o dashboard interativo para análise no Power BI diretamente pelo link abaixo:
        [Arquivo do Power BI](https://github.com/ISQRS00/dashboard_tc4/blob/main/TECH_CHALLENGE_FASE_4.ipynb)
                    
        """)

