# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Sequential, Model
from tensorflow.keras.constraints import max_norm

df_short = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_short.csv").sample(frac=1).reset_index(drop=True)
df_medium = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_medium.csv").sample(frac=1).reset_index(drop=True)
df_dank = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_dank.csv").sample(frac=1).reset_index(drop=True)

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split

vectorizer = TextVectorization(max_tokens=20000)

df_joint = pd.concat([df_short, df_medium, df_dank])
df_joint = df_joint.reset_index(drop=True)
df_train = df_joint.sample(frac=1).reset_index(drop=True)
vectorizer.adapt(np.asarray(df_train["text"]))

def build_model():
    #init_matrix = np.load('universal.npy')
    submodels = []
    for kw in (3, 4, 5):    # kernel sizes
        submodel = Sequential()
        submodel.add(Embedding(
        20002,
        100,
        embeddings_initializer=keras.initializers.Constant(np.load('embed_matrix.npy')),
        trainable=False,))
        submodel.add(Conv1D(100,    
                            kw,
                            padding='valid',
                            activation='relu',
                            strides=1, kernel_constraint=max_norm(3)
                             ))
        submodel.add(GlobalMaxPooling1D())
        submodels.append(submodel)

    submodel1 = submodels[0]
    submodel2 = submodels[1]
    submodel3 = submodels[2]

    x = add([submodel1.output, submodel2.output, submodel3.output])
    
    big_model = Sequential()
    big_model.add(Dropout(0.25))
    big_model.add(Dense(100))
    big_model.add(Dropout(0.6))
    big_model.add(Activation('relu'))
    big_model.add(Dense(1))
    big_model.add(Activation('sigmoid'))

    big_model_output = big_model(x)

    model = Model([submodel1.input, submodel2.input, submodel3.input], big_model_output)

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])  

    print(model.summary())  


    return model

def train_model(df):
    #creating the desired vectors
    text = np.asarray(df['text'])
    y = df["label"]
    text_train, text_test, y_train, y_test = train_test_split(
    text, y, test_size=0.33, random_state=42)

    X_train = vectorizer(text_train)
    X_test = vectorizer(text_test)

    model = build_model()

    model.fit([X_train, X_train, X_train],
                     y_train,
                     epochs=25,
                     validation_data=([X_val, X_val, X_val],
                     y_val))
    
    loss, accuracy =  model.evaluate([X_test, X_test, X_test], y_test)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
        
    return model

train_model(df_medium)

