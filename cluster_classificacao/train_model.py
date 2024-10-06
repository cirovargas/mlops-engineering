import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_cluster_model():
    df = pd.read_csv('dataset.csv')
    df = pd.get_dummies(df, columns=['etnia', 'genero', 'estado_civil', 'escolaridade'])
    X = df.drop('inadimplente', axis=1)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    X['cluster'] = df['cluster']
    y = df['inadimplente']

    columns = X.columns.tolist()
    with open('model_columns.pkl', 'wb') as f:
        joblib.dump(columns, f)

    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X, y)

    joblib.dump(kmeans, 'model_kmeans.joblib')
    joblib.dump(classifier, 'model_classifier.joblib')

if __name__ == '__main__':
    train_cluster_model()