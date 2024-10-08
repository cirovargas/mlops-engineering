import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib


def train_cluster_model():
    df = pd.read_csv('dataset.csv')

    df['risco_inadimplencia'] = df.apply(get_risco_inadimplencia, axis=1)
    df['faixa_etaria'] = df['idade'].apply(get_faixa_etaria)
    df['nivel_renda'] = df['renda'].apply(get_nivel_renda)

    df = df.drop(['etnia', 'genero', 'casa_propria', 'outras_rendas', 'estado_civil', 'escolaridade'], axis=1)

    df = pd.get_dummies(df, columns=['risco_inadimplencia', 'faixa_etaria', 'nivel_renda'])

    X = df.drop(['inadimplente', 'idade', 'renda'], axis=1)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    joblib.dump(kmeans, 'model_kmeans_risk_cluster.joblib')

    columns = X.columns.tolist()
    with open('model_columns_cluster.pkl', 'wb') as f:
        joblib.dump(columns, f)


# Definir categorias de risco de inadimplência
def get_risco_inadimplencia(row):
    if row['inadimplente'] == 0 and row['renda'] > 9000:
        return 'Baixo Risco'
    elif row['inadimplente'] == 1 and row['renda'] < 3000:
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
    train_cluster_model()
