
import joblib

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score

# define data import path
root = os.path.dirname(__file__)
path_df = os.path.join(root, 'data/heart.csv')
data = pd.read_csv(path_df)

# create dummy variables

a = pd.get_dummies(data['cp'], prefix="cp")
b = pd.get_dummies(data['thal'], prefix="thal")
c = pd.get_dummies(data['slope'], prefix="slope")

frames = [data, a, b, c]
data = pd.concat(frames, axis=1)

data = data.drop(columns=['cp', 'thal', 'slope'])

# normalize data

y = data.target.values
x_data = data.drop(['target'], axis=1)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

# split data -- 80% will be training data, 20% will be test data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

# begin Random Forest Classifier

rf = RandomForestClassifier(n_estimators=1000, random_state=1)
rf.fit(x_train.T, y_train.T)

acc = rf.score(x_test.T, y_test.T)
print("Random Forest Accuracy Score : {:.2f}%".format(acc))

#save the trained model
model_path = os.path.join(root, 'models/rfc.sav')
joblib.dump(rf, model_path)

# save the scaler object
pickle_path = os.path.join(root, 'models/rfc.pkl')
with open(pickle_path, 'wb') as pickled_model:
    pickle.dump(pickle_path, pickled_model)
