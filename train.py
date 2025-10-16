import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

train_data= pd.read_csv('data/train.csv')
X = train_data.drop('target', axis=1)
y = train_data['target']

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
model.fit(X, y)

train_preds = model.predict(X)
train_acc = accuracy_score(y, train_preds)

model_path = os.path.join('models', 'model.pkl')
joblib.dump(model, model_path)