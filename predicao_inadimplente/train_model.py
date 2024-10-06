import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    df = pd.read_csv('dataset.csv')

    df = pd.get_dummies(df, columns=['etnia', 'genero', 'estado_civil', 'escolaridade'])

    X = df.drop('inadimplente', axis=1)
    y = df['inadimplente']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    joblib.dump(model, 'model_rf.joblib')

if __name__ == '__main__':
    train_model()
