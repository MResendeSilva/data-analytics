from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, ValidationError
import pickle
import joblib
import pandas as pd
import logging
from functions import MinMax, OneHotEncodeNames, OrdinalEncodeNames, BinarioTransformer, DropFeatures, Oversample

app = Flask(__name__)


# Carregar o pipeline do modelo treinado
# pipeline_path = '/model_data/pipeline.pkl'
# with open(pipeline_path, 'rb') as f:
#     pipeline = pickle.load(f)

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = InputData(**request.get_json())


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

        status_code = 200
        status_message = Response(status=status_code).status

        if y_predito == 0:
            prediction = 'Risco de ficar abaixo do peso'
        elif y_predito in [1,2,3]:
            prediction = 'Sem riscos'
        elif y_predito in [4,5,6]:
            prediction = 'Risco de desenvolver obesidade se continuar com esses hábitos'

        return jsonify({
            "status": status_message,
            "data": {"prediction": prediction}
        }), status_code
    except ValidationError as e:
        status_code = 400
        return jsonify({"error": e.errors()}), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
