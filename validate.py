import os
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

test_data = pd.read_csv('data/test.csv')

X_test = test_data.drop('target', axis=1)
y_test = test_data['target']
model = joblib.load('models/model.pkl')

y_pred = model.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
    "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
}

metrics_path = os.path.join('models', 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

    
#######################################################

#######################################################


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join('models', 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()

# Save confusion matrix as CSV for DVC plots
cm_df = pd.DataFrame(cm)
cm_csv_path = os.path.join('models', 'confusion_matrix.csv')
cm_df.to_csv(cm_csv_path, index=False, header=False)
