from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

expected_columns = [
    'idade',
    'casa_propria',
    'outras_rendas',
    'etnia_Negro',
    'etnia_Branco',
    'etnia_Pardo',
    'etnia_Amarelo',
    'etnia_Indígena',
    'genero_Feminino',
    'genero_Masculino',
    'estado_civil_Casado',
    'estado_civil_Divorciado',
    'estado_civil_Viúvo',
    'estado_civil_Solteiro',
    'escolaridade_Fundamental',
    'escolaridade_Médio',
    'escolaridade_Superior',
    'escolaridade_Pós-graduação'
]


def adjust_input_columns(input_data):
    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    input_data = input_data[model_columns]
    return input_data

@app.route('/cluster', methods=['POST'])
def cluster_client():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        input_data = pd.get_dummies(input_data, columns=['etnia', 'genero', 'estado_civil', 'escolaridade'], drop_first=True)

        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        cluster = cluster_model.predict(input_data)[0]

        return jsonify({'cluster': int(cluster)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    cluster_model = joblib.load('model_kmeans.joblib')
    with open('model_columns.pkl', 'rb') as f:
        model_columns = joblib.load(f)
    app.run(host='0.0.0.0', port=5000)