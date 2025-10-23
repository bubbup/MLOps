import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("heart disease prediction")  
# Load training data
train_data = pd.read_csv('data/train.csv')
X = train_data.drop('target', axis=1)
y = train_data['target']

# Split for quick evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
C_values = [0.1, 1, 10]
kernels = ['linear', 'rbf']

# Create models folder
os.makedirs('models_tuned', exist_ok=True)

# Parent MLflow run
with mlflow.start_run(run_name="svm_tuning") as parent_run:
    print(f"Parent Run ID: {parent_run.info.run_id}")

    best_acc = 0
    best_model = None
    best_params = {}

    for C in C_values:
        for kernel in kernels:
            with mlflow.start_run(run_name=f"C={C}_kernel={kernel}", nested=True):
                model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                acc = accuracy_score(y_val, preds)

                # Log parameters, metrics, and model
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)
                mlflow.log_metric("val_accuracy", acc)
                mlflow.sklearn.log_model(model, "model")


                # Track the best model
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = {"C": C, "kernel": kernel}

    # Save the best model to models_tuned/model.pkl
    if best_model is not None:
        model_path = os.path.join('models_tuned', 'model.pkl')
        joblib.dump(best_model, model_path)
        print(f"Best model saved with params {best_params} and accuracy {best_acc:.4f}")

