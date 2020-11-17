import pandas
import xgboost
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from keras.utils import to_categorical
import h5py
from keras.applications.vgg16 import VGG16

#k_data = pd.read_csv('k_data.csv', sep=';', engine='python')

with open('k_data.csv', newline='') as file:
    k = list(csv.reader(file))      # read rows
with open('w_data.csv', newline='') as file:
    w = list(csv.reader(file)) # read rows
w = [float(it) for item in w for it in item]
k = [float(it) for item in k for it in item]

w = np.array(w).reshape(2000,-1)
k = np.array(k).reshape(2000, 5, 4)
model = VGG16(include_top=False, weights=None, input_shape= (2000, 5, 4))
X_train, X_test, y_train, y_test = k[:1500:], k[1500::], w[:1500:], w[1500::]
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
