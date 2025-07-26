import streamlit as st
import pickle
import joblib
import pandas as pd
import logging
from functions import MinMax, OneHotEncodeNames, OrdinalEncodeNames, BinarioTransformer, DropFeatures, Oversample



st.set_page_config(page_title="Formulário de Previsão", layout="centered")
st.title("📋 Formulário de Previsão")

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
    input_data = {
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

    
    st.success(f"Resultado da Previsão: **{result['data']['prediction']}**")


def predict(data):

    pipeline = joblib.load('pipeline.pkl')
    modelo = joblib.load('modelo.pkl')
    
    #Classe para validar as entradas
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

    obesity_classes = {
        0: 'Abaixo_do_Peso',
        1: 'Peso_Normal',
        2: 'Sobrepeso_Level_I',
        3: 'Sobrepeso_Level_II',
        4: 'Obesidade_Tipo_I',
        5: 'Obesidade_Tipo_II',
        6: 'Obesidade_Tipo_III'
    }


    input_data = InputData(data)

    dados = {
        'Gender': [input_data.gender],
        'Age': [input_data.age],
        'Height': [input_data.height],
        'Weight' : [input_data.weight],
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
        'Obesity': ['Peso_Normal']
    }


    df_amostra = pd.DataFrame(dados)
    x_novo_dado = pipeline.transform(df_amostra)
    x_transformado = x_novo_dado.drop(columns=['Obesity']).head()
    y_predito = modelo.predict(x_transformado)


    if y_predito == 0:
        return 'Risco de ficar abaixo do peso'
    elif y_predito in [1,2,3]:
        return 'Sem riscos'
    elif y_predito in [4,5,6]:
        return 'Risco de desenvolver obesidade se continuar com esses hábitos'
    else:
        return "error"


