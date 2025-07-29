import streamlit as st
import pickle
import joblib
import pandas as pd
import logging
import os
from pydantic import BaseModel
from functions import MinMax, OneHotEncodeNames, OrdinalEncodeNames, BinarioTransformer, DropFeatures, Oversample

# ---------------- FUNÇÃO DE PREVISÃO ---------------- #

def predict_obesity(data):
    # Garante que os caminhos sejam relativos ao script
    base_path = os.path.dirname(__file__)
    pipeline_path = os.path.join(base_path, "pipeline.pkl")
    modelo_path = os.path.join(base_path, "modelo.pkl")

    pipeline = joblib.load(pipeline_path)
    modelo = joblib.load(modelo_path)

    class InputData(BaseModel):
        gender: str
        age: int
        height: float
        weight: float
        family_history: str
        favc: str
        fcvc: str
        ncp: str
        caec: str
        smoke: str
        ch20: str
        scc: str
        faf: str
        calc: str
        mtrans: str

    input_data = InputData(**data)

    dados = {
        'Gender': [input_data.gender],
        'Age': [input_data.age],
        'Height': [input_data.height],
        'Weight': [input_data.weight],
        'family_history': [input_data.family_history],
        'FAVC': [input_data.favc],
        'FCVC': [input_data.fcvc],
        'NCP': [input_data.ncp],
        'CAEC': [input_data.caec],
        'SMOKE': [input_data.smoke],
        'CH2O': [input_data.ch20],
        'SCC': [input_data.scc],
        'FAF': [input_data.faf],
        'CALC': [input_data.calc],
        'MTRANS': [input_data.mtrans],
        'Obesity': ['Peso_Normal']  # dummy target
    }

    df_amostra = pd.DataFrame(dados)

    # Transformação com pipeline
    x_novo_dado = pipeline.transform(df_amostra)

    if 'Obesity' in x_novo_dado.columns:
        x_transformado = x_novo_dado.drop(columns=['Obesity']).head()
    else:
        x_transformado = x_novo_dado.head()

    y_predito = modelo.predict(x_transformado)
    y_predito = y_predito[0]

    if y_predito == 0:
        prediction = 'Risco de ficar abaixo do peso'
    elif y_predito in [1,2,3]:
        prediction = 'Sem riscos'
    elif y_predito in [4,5,6]:
        prediction = 'Risco de desenvolver obesidade se continuar com esses hábitos'
    else:
        return "Erro na previsão"

# ---------------- INTERFACE STREAMLIT ---------------- #
#url = "https://app.powerbi.com/view?r=eyJrIjoiZTZjMmQ3NWEtY2IwZC00M2QyLWI0OGItMTczMTk0NTc2ZGNjIiwidCI6Ijg5NmI3ZjkyLTgyZDItNDc3Ny1hYTQwLThiNjEyZWY2MWJmNCJ9"
#git_hub = "https://github.com/MResendeSilva/data-analytics"
st.set_page_config(page_title="Formulário de Previsão", layout="centered")
st.title("📋 Formulário de Previsão")
#st.write("📈 Visão analítica [link](%s)" % url)
#st.write("Repositório [link](%s)" % git_hub)

# Campos categóricos
gender = st.selectbox("Gênero", options=["Feminino", "Masculino"])
family_history = st.selectbox("Histórico Familiar de Obesidade?", options=["Sim", "Não"])
favc = st.selectbox("Consome alimentos com alto teor calórico frequentemente (FAVC)?", options=["Sim", "Não"])
fcvc = st.selectbox("Frequência de consumo de vegetais?", options=["Sempre", "Às vezes", "Não"])
ncp = st.selectbox("Número de refeições por dia (NCP)?", options=["1 Refeição", "2 Refeições", "3 Refeições", "4 ou mais refeições"])
caec = st.selectbox("Consome alimentos entre as refeições (CAEC)?", options=["Não", "A vezes", "Frequentemente", "Sempre"])
smoke = st.selectbox("Fuma?", options=["Não", "Sim"])
ch20 = st.selectbox("Consumo diário de água?", options=["Até 1 Litro", "Entre 1 a 2 Litros", "2 ou mais Litros"])
scc = st.selectbox("Monitora calorias ingeridas (SCC)?", options=["Não", "Sim"])
faf = st.selectbox("Frequência de atividade física (FAF)?", options=["0 a 1 dia", "2 a 3 dias", "4 a 5 dias", "Mais de 5 dias"])
calc = st.selectbox("Frequência de consumo de álcool?", options=["Não", "A vezes", "Frequentemente", "Sempre"])
mtrans = st.selectbox("Meio de transporte predominante?", options=["Transporte Público", "Caminhada", "Automóvel", "Motocicleta", "Bicicleta"])

# Campos numéricos
age = st.number_input("Idade (anos)", min_value=0, max_value=120, value=25)
height = st.number_input("Altura (em metros)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
weight = st.number_input("Peso (em kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)

# Botão para enviar
if st.button("🔮 Prever"):
    input_dict = {
        "gender": gender,
        "age": age,
        "height": height,
        "weight": weight,
        "family_history": family_history,
        "favc": favc,
        "fcvc": fcvc,
        "ncp": ncp,
        "caec": caec,
        "smoke": smoke,
        "ch20": ch20,
        "scc": scc,
        "faf": faf,
        "calc": calc,
        "mtrans": mtrans
    }

    response = predict_obesity(input_dict)
    st.success(f"Resultado da Previsão: **{response}**")
