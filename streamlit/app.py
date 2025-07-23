import streamlit as st
import requests
import logging


st.set_page_config(page_title="Formulário de Previsão", layout="centered")
st.title("📋 Formulário de Previsão")

# Campos categóricos
gender = st.selectbox("Gênero", options=["Feminino", "Masculino"])
family_history = st.selectbox("Histórico Familiar de Obesidade?", options=["Sim", "Não"])
FAVC = st.selectbox("Consome alimentos com alto teor calórico frequentemente (FAVC)?", options=["Sim", "Não"])
FCVC = st.selectbox("Frequência de consumo de vegetais?", options=["Sempre", "Às vezes", "Não"])
NCP = st.selectbox("Número de refeições por dia (NCP)?", options=["1 Refeição", "2 Refeições", "3 Refeições", "4 ou mais refeições"])
CAEC = st.selectbox("Consome alimentos entre as refeições (CAEC)?", options=["Não", "A vezes", "Frequentemente", "Sempre"])
SMOKE = st.selectbox("Fuma?", options=["Não", "Sim"])
CH2O = st.selectbox("Consumo diário de água?", options=["Até 1 Litro", "Entre 1 a 2 Litros", "2 ou mais Litros"])
SCC = st.selectbox("Monitora calorias ingeridas (SCC)?", options=["Não", "Sim"])
FAF = st.selectbox("Frequência de atividade física (FAF)?", options=["0 a 1 dia", "2 a 3 dias", "4 a 5 dias", "Mais de 5 dias"])
CALC = st.selectbox("Frequência de consumo de álcool?", options=["Não", "A vezes", "Frequentemente", "Sempre"])
MTRANS = st.selectbox("Meio de transporte predominante?", options=["Transporte Público", "Caminhada", "Automóvel", "Motocicleta", "Bicicleta"])

# Campos numéricos
Age = st.number_input("Idade (anos)", min_value=0, max_value=120, value=25)
Height = st.number_input("Altura (em metros)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
Weight = st.number_input("Peso (em kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)

# Botão para enviar
if st.button("🔮 Prever"):
    input_data = {
        "Gender": gender,
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "family_history": family_history,
        "FAVC": FAVC,
        "FCVC": FCVC,
        "NCP": NCP,
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "CH2O": CH2O,
        "SCC": SCC,
        "FAF": FAF,
        "CALC": CALC,
        "MTRANS": MTRANS
    }

    try:
        response = requests.post("http://api:5000/predict", json=input_data)
        
        logging.info(input_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Resultado da Previsão: **{result['data']['prediction']}**")
        else:
            st.error(f"Erro na predição: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Não foi possível conectar à API. Verifique se o servidor está online.")