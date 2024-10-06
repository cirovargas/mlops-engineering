import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_probability_model():
    df = pd.read_csv('dataset.csv')
    df = pd.get_dummies(df, columns=['etnia', 'genero', 'estado_civil', 'escolaridade'])
    X = df.drop('inadimplente', axis=1)
    y = df['inadimplente']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'model_probability.joblib')
    columns = X.columns.tolist()
    with open('model_columns_probability.pkl', 'wb') as f:
        joblib.dump(columns, f)

if __name__ == '__main__':
    train_probability_model()