# Importando as bibliotecas:
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd



class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_scale=['Age', 'Height']):
        self.features_to_scale = features_to_scale

    def fit(self, df, y=None):
        # Armazena apenas as colunas que existem no DataFrame
        self.cols_present = [col for col in self.features_to_scale if col in df.columns]
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[self.cols_present])
        return self

    def transform(self, df):
        df_copy = df.copy()
        if self.cols_present:
            df_copy[self.cols_present] = self.scaler.transform(df_copy[self.cols_present])
        return df_copy
    

class OneHotEncodeNames(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode=['MTRANS']):
        self.features_to_encode = features_to_encode
        self.encoder = None
        self.cols_present = []

    def fit(self, df, y=None):
        self.cols_present = [col for col in self.features_to_encode if col in df.columns]
        if self.cols_present:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(df[self.cols_present])
        return self

    def transform(self, df):
        df_copy = df.copy()
        if not self.cols_present:
            return df_copy

        def one_hot_encode(df_subset, cols):
            one_hot = self.encoder
            one_hot_array = one_hot.transform(df_subset[cols])
            feature_names = one_hot.get_feature_names_out(cols)
            df_one_hot = pd.DataFrame(one_hot_array, columns=feature_names, index=df_subset.index)
            return df_one_hot

        def concat_with_rest(df_original, df_one_hot, cols_encoded):
            rest_features = [col for col in df_original.columns if col not in cols_encoded]
            df_concat = pd.concat([df_original[rest_features], df_one_hot], axis=1)
            return df_concat

        df_one_hot = one_hot_encode(df_copy, self.cols_present)
        df_encoded = concat_with_rest(df_copy, df_one_hot, self.cols_present)
        return df_encoded
    

class OrdinalEncodeNames(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode=None):
        if features_to_encode is None:
            self.features_to_encode = ['FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'CALC', 'Obesity']
        else:
            self.features_to_encode = features_to_encode

        self.ordinal_encoder = None

        self.category_orders = {
            'FCVC': ['Não', 'Às vezes', 'Sempre'],
            'NCP': ['1 Refeição', '2 Refeições', '3 Refeições', '4 ou mais refeições'],
            'CAEC': ['Não', 'As vezes', 'Frequentemente', 'Sempre'],
            'CH2O': ['Até 1 Litro', 'Entre 1 a 2 Litros', '2 ou mais Litros'],
            'FAF': ['0 a 1 dia', '2 a 3 dias', '4 a 5 dias', 'Mais de 5 dias'],
            'CALC': ['Não', 'As vezes', 'Frequentemente', 'Sempre'],
            'Obesity': [
                'Abaixo_do_Peso',
                'Peso_Normal',
                'Sobrepeso_Level_I',
                'Sobrepeso_Level_II',
                'Obesidade_Tipo_I',
                'Obesidade_Tipo_II',
                'Obesidade_Tipo_III'
            ]
        }

    def fit(self, df, y=None):
        self.cols_present = [col for col in self.features_to_encode if col in df.columns]
        categories = [self.category_orders[col] for col in self.cols_present]
        self.ordinal_encoder = OrdinalEncoder(
            categories=categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.ordinal_encoder.fit(df[self.cols_present])
        return self

    def transform(self, df):
        df_copy = df.copy()
        self.cols_present = [col for col in self.features_to_encode if col in df.columns]
        if self.cols_present:
            df_copy[self.cols_present] = self.ordinal_encoder.transform(df_copy[self.cols_present])
        return df_copy


class BinarioTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode=['family_history', 'FAVC', 'SMOKE', 'SCC'], feature_exception=['Gender']):
        self.features_to_encode = features_to_encode
        self.feature_exception = feature_exception

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = df.copy()  # Evita modificar o original

        for col in self.features_to_encode:
            if col in df.columns:
                df[col] = df[col].map({'Sim': 1, 'Não': 0})

        for col in self.feature_exception:
            if col in df.columns:
                df[col] = df[col].map({'Masculino': 0, 'Feminino': 1})

        return df
    

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop=['Weight', 'MTRANS']):
        self.features_to_drop = features_to_drop

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = df.copy()
        # Mantém apenas as colunas que estão no DataFrame
        cols_to_drop = [col for col in self.features_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        return df


class Oversample(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Obesity'):
        self.target_column = target_column

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        # Verifica se a coluna de target existe
        if self.target_column not in df.columns:
            print(f"[AVISO] Coluna '{self.target_column}' não encontrada no DataFrame. Nenhum oversampling será aplicado.")
            return df

        oversample = SMOTE(sampling_strategy='minority', random_state=SEED)
        x = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        x_bal, y_bal = oversample.fit_resample(x, y)

        df_bal = pd.concat([pd.DataFrame(x_bal, columns=x.columns),
                            pd.DataFrame(y_bal, columns=[self.target_column])], axis=1)

        return df_bal



def predict_obesity(data):

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