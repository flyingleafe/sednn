import numpy as np

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

class DirectEstimatorDNN(Sequential):
    def __init__(self, n_concat, n_freq, n_hid=2048):
        super().__init__()
        self.add(Flatten(input_shape=(n_concat, n_freq)))
        self.add(Dense(n_hid, activation='relu'))
        self.add(Dropout(0.2))
        self.add(Dense(n_hid, activation='relu'))
        self.add(Dropout(0.2))
        self.add(Dense(n_hid, activation='relu'))
        self.add(Dropout(0.2))
        self.add(Dense(n_freq, activation='linear'))
        