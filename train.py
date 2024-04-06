import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()
#intentionally choosing a bad model
model = KNeighborsClassifier(n_neighbors=1).fit(X, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)