import os
import sys
import pandas as pd
import joblib
import mlflow
from sklearn.svm import SVC
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
# Load training data
train_data = pd.read_csv('data/train.csv')
X = train_data.drop('target', axis=1)
y = train_data['target']

# Define and train SVM model
with mlflow.start_run(run_name="baseline_svm"):
    model = SVC(
        C=0.5,
        kernel='rbf',
        probability=True,
        random_state=42
    )
    model.fit(X, y)
    preds = model.predict(X)

    acc = (preds == y).mean()
    mlflow.log_param("C", 0.5)
    mlflow.log_param("kernel", "rbf")
    mlflow.log_metric("train_accuracy", acc)
    model_path = "model.joblib"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, "model")
    os.remove(model_path)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', 'model.pkl'))