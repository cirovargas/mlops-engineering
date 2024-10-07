import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

cluster_labels = {
    0: 'Cluster 1 ',
    1: 'Cluster 2 ',
    2: 'Cluster 3 '
}

def adjust_input_columns(input_data):
    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    input_data = input_data[model_columns]
    return input_data


@app.route('/cluster', methods=['POST'])
def cluster_risk():
    try:
        data = request.get_json()

        input_data = pd.DataFrame([data])

        input_data['risco_inadimplencia'] = get_risco_inadimplencia(data)
        input_data['faixa_etaria'] = get_faixa_etaria(data['idade'])
        input_data['nivel_renda'] = get_nivel_renda(data['renda'])
        input_data = pd.get_dummies(input_data, columns=['risco_inadimplencia', 'faixa_etaria', 'nivel_renda'])
        input_data = adjust_input_columns(input_data)
        cluster_id = model.predict(input_data)[0]
        cluster_label = cluster_labels.get(cluster_id, 'Cluster Desconhecido')
        return jsonify({'cluster_id': int(cluster_id), 'cluster_label': cluster_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_risco_inadimplencia(data):
    if data['inadimplente'] == 0 and data['renda'] > 9000:
        return 'Baixo Risco'
    elif data['inadimplente'] == 1 and data['renda'] < 3000:
        return 'Alto Risco'
    else:
        return 'Médio Risco'


def get_faixa_etaria(idade):
    if 18 <= idade <= 25:
        return 'Jovens Adultos'
    elif 26 <= idade <= 40:
        return 'Adultos'
    elif 41 <= idade <= 60:
        return 'Meia-idade'
    else:
        return 'Sêniores'


def get_nivel_renda(renda):
    if renda < 3000:
        return 'Baixa Renda'
    elif 3000 <= renda <= 9000:
        return 'Renda Média'
    else:
        return 'Alta Renda'


if __name__ == '__main__':
    model = joblib.load('model_kmeans_risk_cluster.joblib')
    with open('model_columns_cluster.pkl', 'rb') as f:
        model_columns = joblib.load(f)
    app.run(host='0.0.0.0', port=5000)
