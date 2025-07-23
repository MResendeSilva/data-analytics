from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, ValidationError
import pickle
from utilidades import MinMax, OneHotEncodeNames, OrdinalEncodeNames, BinarioTransformer, DropFeatures, Oversample
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o pipeline do modelo treinado
pipeline_path = '/model_data/pipeline.pkl'
with open(pipeline_path, 'rb') as f:
    pipeline = pickle.load(f)

# Classe para validar as entradas
class InputData(BaseModel):
    Gender: str,
    Age: int,
    Height: float,
    Weight: float,
    family_history: str,
    FAVC: str,
    FCVC: str,
    NCP: str,
    CAEC: str,
    SMOKE: str,
    CH2O: str,
    SCC: str,
    FAF: str,
    CALC: str,
    MTRANS: str

dados = {
    'Gender': ['Masculino'],
    'Age': [26],
    'Height': [1.85],
    'family_history': ['Sim'],
    'FAVC': ['Sim'],
    'FCVC': ['Frequentemente'],
    'NCP': ['3 Refeições'],
    'CAEC': ['Sempre'],
    'SMOKE': ['Não'],
    'CH2O': ['2 ou mais Litros'],
    'SCC': ['Não'],
    'FAF': ['4 a 5 dias'],
    'CALC': ['A vezes'],
    'MTRANS': ['Transporte Público'],
    'Obesity': ['Peso_Normal'] # Não alterar, parâmetro será deletado após passar pela pipeline
}


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
        features = [[
            input_data.Gender,
            input_data.Age,
            input_data.Height,
            input_data.Weight,
            input_data.family_history,
            input_data.FAVC,
            input_data.FCVC,
            input_data.NCP,
            input_data.CAEC,
            input_data.SMOKE,
            input_data.CH2O,
            input_data.SCC,
            input_data.FAF,
            input_data.CALC,
            input_data.MTRANS
        ]]
        prediction = pipeline.predict(features)
        status_code = 200
        status_message = Response(status=status_code).status
        return jsonify({
            "status": status_message,
            "data": {"prediction": prediction[0]}
        }), status_code

    except ValidationError as e:
        status_code = 400
        return jsonify({"error": e.errors()}), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
