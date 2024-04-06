import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]

meta_model = LogisticRegression(random_state=42)

stacking_ensemble = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

model = stacking_ensemble.fit(X, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)