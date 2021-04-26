import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, randint
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Sequential, Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.model_selection import train_test_split

import requests
import io

response = requests.get('https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/universal.npy')
init_matrix = np.load(io.BytesIO(response.content))

vectorizer = TextVectorization(max_tokens=20000)

df_short = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_short.csv")
df_medium = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_medium.csv")
df_dank = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_dank.csv")

df_joint = pd.concat([df_short, df_medium, df_dank])
df_joint = df_joint.reset_index(drop=True)
df_train = df_joint.sample(frac=1).reset_index(drop=True)
vectorizer.adapt(np.asarray(df_train["text"]))

def data():
    vectorizer = TextVectorization(max_tokens=20000)
    df_short = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_short.csv")
    df_medium = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_medium.csv")
    df_dank = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_dank.csv")

    df_joint = pd.concat([df_short, df_medium, df_dank])
    df_joint = df_joint.reset_index(drop=True)
    df_train = df_joint.sample(frac=1).reset_index(drop=True)
    vectorizer.adapt(np.asarray(df_train["text"]))

    df = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_medium.csv")
    text = np.asarray(df['text'])
    y = df["label"]
    text_train, text_test, y_train, y_test = train_test_split(
    text, y, test_size=0.2, random_state=42)

    X_train = vectorizer(text_train)
    X_test = vectorizer(text_test)

    return X_train, y_train, X_test, y_test

def build_convnet():

  response = requests.get('https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/universal.npy')
  init_matrix = np.load(io.BytesIO(response.content)) 
  submodels = []
  for kw in (3, 4, 5):    # kernel sizes
      submodel = Sequential()
      submodel.add(Embedding(
      20002,
      100,
      embeddings_initializer=keras.initializers.Constant(init_matrix),
      trainable=False))
      submodel.add(Conv1D(100,    
                          kw,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          kernel_constraint=max_norm({{uniform(1,8)}})
                            ))
      submodel.add(GlobalMaxPooling1D())
      submodels.append(submodel)

  submodel1 = submodels[0]
  submodel2 = submodels[1]
  submodel3 = submodels[2]

  x = add([submodel1.output, submodel2.output, submodel3.output])
  
  big_model = Sequential()
  big_model.add(Dropout({{uniform(0,1)}}))
  big_model.add(Dense({{choice([1,2,5,10,25,50,100])}}))
  big_model.add(Dropout({{uniform(0,1)}}))
  big_model.add(Activation('relu'))
  big_model.add(Dense(1))
  big_model.add(Activation('sigmoid'))

  big_model_output = big_model(x)

  model = Model([submodel1.input, submodel2.input, submodel3.input], big_model_output)
  model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

  result = model.fit([X_train,X_train,X_train], y_train,
            batch_size={{choice([32, 64, 128])}},
            epochs=10,
            validation_split=0.1)
  validation_acc = np.amax(result.history['val_accuracy']) 
  print('Best validation acc of epoch:', validation_acc)
  return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def build_rnn():

  response = requests.get('https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/universal.npy')
  init_matrix = np.load(io.BytesIO(response.content))
  model = Sequential()
  model.add(Embedding(
      20002,
      100,
      embeddings_initializer=keras.initializers.Constant(init_matrix),
      trainable=False
  ))
  model.add(Bidirectional(LSTM(100, return_sequences=True)))
  model.add(Dropout({{uniform(0,1)}}))
  model.add(Bidirectional(LSTM({{choice([1,2,5,10,25,50,100])}}, return_sequences=False)))
  model.add(Dropout({{uniform(0,1)}}))
  model.add(Dense({{choice([1,2,5,10,25,50,100])}}, activation='relu'))
  model.add(Dropout({{uniform(0,1)}}))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

  result = model.fit(X_train, y_train,
            batch_size=32,
            epochs=15,
            validation_split=0.1
            )
  validation_acc = np.amax(result.history['val_accuracy']) 
  print('Best validation acc of epoch:', validation_acc)
  return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def optimize():
    best_run, best_model = optim.minimize(model=build_rnn,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=15,
                                        trials=Trials()
                                        )
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    #print(best_model.evaluate([X_test,X_test,X_test], Y_test))
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)


optimize()
