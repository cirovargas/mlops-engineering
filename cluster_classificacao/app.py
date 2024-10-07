import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carregar o modelo treinado e as colunas esperadas


# Definir os labels dos clusters
cluster_labels = {
    0: 'Cluster 1 - Baixo Risco',
    1: 'Cluster 2 - Médio Risco',
    2: 'Cluster 3 - Alto Risco'
}


# Função para ajustar as colunas do dataframe de entrada
def adjust_input_columns(input_data):
    # Garantir que todas as colunas esperadas estejam presentes
    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0  # Adicionar coluna faltante com valor 0

    # Garantir a mesma ordem de colunas
    input_data = input_data[model_columns]
    return input_data


# Rota para clusterizar o cliente e retornar o label do cluster
@app.route('/cluster', methods=['POST'])
def cluster_risk():
    try:
        # Receber os dados JSON
        data = request.get_json()

        # Converter os dados para dataframe
        input_data = pd.DataFrame([data])

        print('1')

        # Criar dummies para variáveis categóricas
        input_data['risco_inadimplencia'] = get_risco_inadimplencia(data)
        input_data['faixa_etaria'] = get_faixa_etaria(data['idade'])
        input_data['nivel_renda'] = get_nivel_renda(data['renda'])
        print('2')
        # input_data = input_data.drop(['etnia', 'genero', 'casa_propria', 'outras_rendas', 'estado_civil', 'escolaridade'], axis=1)
        print('3')
        input_data = pd.get_dummies(input_data, columns=['risco_inadimplencia', 'faixa_etaria', 'nivel_renda'])
        print('4')
        # Ajustar as colunas do input para garantir consistência com o treinamento
        input_data = adjust_input_columns(input_data)
        print('5')
        # Fazer a predição do cluster
        cluster_id = model.predict(input_data)[0]
        print('6')
        # Obter o label correspondente
        cluster_label = cluster_labels.get(cluster_id, 'Cluster Desconhecido')
        print('7')
        print('cluster id')
        print(cluster_id)
        print('cluster label')
        print(cluster_label)
        # Retornar o ID do cluster e o label
        return jsonify({'cluster_id': int(cluster_id), 'cluster_label': cluster_label})

    except Exception as e:
        print('8', flush=True)
        print(str(e), flush=True)
        return jsonify({'error': str(e)}), 400


# Funções para risco, faixa etária e nível de renda
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
