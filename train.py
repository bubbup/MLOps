import os
import pandas as pd
import joblib
from sklearn.svm import SVC

# Load training data
train_data = pd.read_csv('data/train.csv')
X = train_data.drop('target', axis=1)
y = train_data['target']

# Define and train SVM model
model = SVC(
    C=1.0,               # Regularization parameter
    kernel='rbf',        # Kernel type
    probability=True,    # Enable probabilities for later validation
    random_state=42
)
model.fit(X, y)

# Save trained model
os.makedirs('models', exist_ok=True)
model_path = os.path.join('models', 'model.pkl')
joblib.dump(model, model_path)
