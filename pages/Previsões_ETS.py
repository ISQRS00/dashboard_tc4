st.write("""
### Como Funciona a Divisão dos Dados?

O modelo de previsão ETS é treinado com dados históricos do preço do petróleo e funciona com dois conjuntos principais:
- **Treinamento:** Utilizado para identificar padrões históricos, como tendências e sazonalidades.
- **Validação:** Usado para testar o modelo, comparando as previsões com os valores reais.

### Por que Usar um Intervalo de Treino Menor?

Para garantir que o Streamlit funcione de forma eficiente, optamos por treinar o modelo com um intervalo menor de dados históricos. Isso reduz o tempo de processamento sem comprometer a precisão do modelo.

⚠️ **Nota:** Dados analíticos completos estão disponíveis no Power BI, enquanto aqui focamos em uma amostra representativa para demonstrar a funcionalidade.

### Como Funciona a Separação?

Escolha um número de dias para separar os dados de treino e validação. O modelo será treinado com os dados antes dessa data e testado com os dados após o corte. Experimente diferente
