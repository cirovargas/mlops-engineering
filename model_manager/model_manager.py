import json
from flask import Flask, request, jsonify
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CLUSTER_API_URL = 'http://mlops-cluster-api:5000/cluster'
INADIMPLENTE_API_URL = 'http://mlops-inadimplente-api:5000/predict'

DATABASE_URL = os.environ.get('DATABASE_URL', 'dbname=model_manager_db user=postgres password=postgres host=mlops-postgres port=5432')

def connect_db():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def init_db():
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forms (
                id SERIAL PRIMARY KEY,
                data JSONB,
                cluster INTEGER,
                prediction BOOLEAN
            );
        ''')
        conn.commit()
    conn.close()

@app.route('/cluster', methods=['POST'])
def cluster_client():
    try:
        data = request.get_json()

        cluster_response = requests.post(CLUSTER_API_URL, json=data)
        cluster_data = cluster_response.json()

        conn = connect_db()
        with conn.cursor() as cursor:
            cursor.execute(
                'INSERT INTO forms (data, cluster) VALUES (%s, %s) RETURNING id;',
                (json.dumps(data), cluster_data['cluster'])
            )
            conn.commit()
        conn.close()

        return jsonify(cluster_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict_inadimplencia():
    try:
        data = request.get_json()

        classifier_response = requests.post(INADIMPLENTE_API_URL, json=data)
        classifier_data = classifier_response.json()

        conn = connect_db()
        with conn.cursor() as cursor:
            cursor.execute(
                'INSERT INTO forms (data, prediction) VALUES (%s, %s) RETURNING id;',
                (json.dumps(data), classifier_data['inadimplente'])
            )
            conn.commit()
        conn.close()

        return jsonify(classifier_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
