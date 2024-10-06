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


@app.route('/predict', methods=['POST'])
def predict_inadimplencia():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        input_data = pd.get_dummies(input_data, columns=['etnia', 'genero', 'estado_civil', 'escolaridade'], drop_first=True)

        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        prediction = model.predict(input_data)[0]

        return jsonify({'inadimplente': bool(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    model = joblib.load('model_rf.joblib')
    app.run(host='0.0.0.0', port=5000)
