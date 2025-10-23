import os
import sys
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from dotenv import load_dotenv

# Fix console encoding for Windows to handle Unicode characters in dagshub auth prompt
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Try to set console code page using ctypes if available
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except (ImportError, AttributeError):
        pass  # Ignore if ctypes not available or fails

load_dotenv()
import dagshub
dagshub.init(repo_owner='bubbup', repo_name='MLOps', mlflow=True)
# Set MLflow tracking URI - IMPORTANT!
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Set MLflow experiment
mlflow.set_experiment("heart Disease Prediction")
test_data = pd.read_csv('data/test.csv')

X_test = test_data.drop('target', axis=1)
y_test = test_data['target']
model = joblib.load('models_tuned/model.pkl')

y_pred = model.predict(X_test)

# metrics_path = os.path.join('models', 'metrics.json')
# with open(metrics_path, 'w') as f:
#     json.dump(metrics, f, indent=4)


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
with mlflow.start_run(run_name="validation"):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_artifact(cm_path)
# Save confusion matrix as CSV for DVC plots
# cm_df = pd.DataFrame(cm)
# cm_csv_path = os.path.join('models', 'confusion_matrix.csv')
# cm_df.to_csv(cm_csv_path, index=False, header=False)