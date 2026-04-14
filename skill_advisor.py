# skill_advisor.py
import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

class SkillAdvisor:
    def __init__(self, model_path="skill_model.pkl", encoder_path="skill_encoder.pkl"):
        self.model = None
        self.encoder = None
        self.model_path = model_path
        self.encoder_path = encoder_path
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            self.load()

    def train(self, data_file):
        # data_file: CSV with columns: skill_name, outcome (e.g., survival_time, damage)
        df = pd.read_csv(data_file)
        self.encoder = LabelEncoder()
        df['skill_encoded'] = self.encoder.fit_transform(df['skill_name'])
        X = df[['skill_encoded']]
        y = df['outcome']  # e.g., survival_time
        self.model = lgb.LGBMRegressor(n_estimators=100, max_depth=5)
        self.model.fit(X, y)
        self.save()

    def save(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.encoder_path, 'wb') as f:
            pickle.dump(self.encoder, f)

    def load(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)

    def predict_best(self, available_skills):
        # available_skills: list of skill names
        encoded = self.encoder.transform(available_skills)
        preds = self.model.predict(encoded.reshape(-1, 1))
        best_idx = np.argmax(preds)
        return available_skills[best_idx]