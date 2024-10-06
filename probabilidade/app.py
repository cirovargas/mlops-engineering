import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

def adjust_input_columns(input_data):

    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    input_data = input_data[model_columns]
    return input_data


@app.route('/predict_probability', methods=['POST'])
def predict_probability():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        input_data = pd.get_dummies(input_data, columns=['etnia', 'genero', 'estado_civil', 'escolaridade'])
        input_data = adjust_input_columns(input_data)
        probability = model.predict_proba(input_data)[0][1]
        probability_percent = round(probability * 100, 2)
        return jsonify({'probability': probability_percent})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    model = joblib.load('model_probability.joblib')
    with open('model_columns_probability.pkl', 'rb') as f:
        model_columns = joblib.load(f)
    app.run(host='0.0.0.0', port=5000)
