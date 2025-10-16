import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('data\heart.csv')
data = data.rename(
    columns = {'cp':'chest_pain_type', 
               'trestbps':'resting_blood_pressure', 
               'chol': 'cholesterol',
               'fbs': 'fasting_blood_sugar',
               'restecg' : 'resting_electrocardiogram', 
               'thalach': 'max_heart_rate_achieved', 
               'exang': 'exercise_induced_angina',
               'oldpeak': 'st_depression', 
               'slope': 'st_slope', 
               'ca':'num_major_vessels', 
               'thal': 'thalassemia'}, 
    errors="raise")
data['sex'] = data['sex'].replace({0: 'female', 1: 'male'})



def label_encode_cat_features(data, cat_features):
    '''
    Given a dataframe and its categorical features, this function returns label-encoded dataframe
    '''
    
    label_encoder = LabelEncoder()
    data_encoded = data.copy()
    
    for col in cat_features:
        data_encoded[col] = label_encoder.fit_transform(data[col])
    
    data = data_encoded
    
    return data



cat_features = ['sex', 'chest_pain_type', 'fasting_blood_sugar',
                'resting_electrocardiogram', 'exercise_induced_angina',
                'st_slope', 'thalassemia']

data = label_encode_cat_features(data, cat_features)
seed = 42
test_size = 0.25

features = data.columns.drop('target')

X = data[features]
y = data['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = test_size, random_state=seed)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_val, y_val], axis=1)
train_path = os.path.join('data', 'train.csv')
test_path = os.path.join('data', 'test.csv')

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)