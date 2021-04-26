import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Sequential, Model
import pickle
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
#from sklearn.model_selection import train_test_split

df_short = pd.read_csv("data/processed/processed_short.csv")
df_medium = pd.read_csv("data/processed/processed_medium.csv")
df_dank= pd.read_csv("data/processed/processed_dank.csv")

vectorizer = TextVectorization(max_tokens=30000, output_sequence_length=200)

init_matrix = pickle.load(open( "init_matrix.p", "rb" ))

def build_model(hp):
    model = Sequential()
    model.add(Embedding(
        20002,
        100,
        embeddings_initializer=keras.initializers.Constant(init_matrix),
        trainable=False
    ))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])  

print('asd')