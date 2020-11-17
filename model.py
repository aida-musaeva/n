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
model = Sequential()
model.add(Dense(32, input_shape=(2000,5,4)))


