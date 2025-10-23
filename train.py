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
mlflow.set_experiment("Heart Disease Prediction")
# Load training data
train_data = pd.read_csv('data/train.csv')
X = train_data.drop('target', axis=1)
y = train_data['target']

# Define and train SVM model
model = SVC(
    C=0.5,               # Regularization parameter
    kernel='rbf',        # Kernel type
    probability=True,    # Enable probabilities for later validation
    random_state=42
)
model.fit(X, y)

# Save trained model
os.makedirs('models', exist_ok=True)
model_path = os.path.join('models', 'model.pkl')
joblib.dump(model, model_path)
