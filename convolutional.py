import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Sequential, Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.metrics import precision_score, f1_score, roc_auc_score, accuracy_score, recall_score
from tf.keras.callbacks import EarlyStopping

import requests
import io

from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

run = neptune.init(project='bocsardigergely/thesis',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZTlkNzc4NC1iYTZjLTQwODktODRiOS1iZjc2NjM3MDQxNWMifQ==')

neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
es = EarlyStopping(monitor='val_loss', patience=4)

df_short = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_short.csv")
df_medium = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_medium.csv")
df_dank = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_dank.csv")

vectorizer = TextVectorization(max_tokens=20000)


df_joint = pd.concat([df_short, df_medium, df_dank])
df_joint = df_joint.reset_index(drop=True)
df_train = df_joint.sample(frac=1).reset_index(drop=True)
vectorizer.adapt(np.asarray(df_train["text"]))

def build_model():
    response = requests.get('https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/universal.npy')
    init_matrix = np.load(io.BytesIO(response.content)) 
    submodels = []
    for kw in (3, 4, 5):    # kernel sizes
        submodel = Sequential()
        submodel.add(Embedding(
        20002,
        100,
        embeddings_initializer=keras.initializers.Constant(init_matrix),
        trainable=False,))
        submodel.add(Conv1D(100,    
                            kw,
                            padding='valid',
                            activation='relu',
                            strides=1, kernel_constraint=max_norm(5.397124926498551)
                             ))
        submodel.add(GlobalMaxPooling1D())
        submodels.append(submodel)

    submodel1 = submodels[0]
    submodel2 = submodels[1]
    submodel3 = submodels[2]

    x = add([submodel1.output, submodel2.output, submodel3.output])
    
    big_model = Sequential()
    big_model.add(Dropout(0.09225974322037533))
    big_model.add(Dense(25))
    big_model.add(Dropout(0.20942239619394942))
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
    text, y, test_size=0.2, random_state=42)

    X_train = vectorizer(text_train)
    X_test = vectorizer(text_test)

    model = build_model()

    model.fit([X_train, X_train, X_train],
                     y_train,
                     batch_size=32,
                     epochs=25,
                     validation_split=0.1,
                     callbacks=[es, neptune_cbk]
                     )
    
    y_pred = model.predict([X_test, X_test, X_test])
    y_pred_bool = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred_bool)
    precision = precision_score(y_test, y_pred_bool)
    recall = recall_score(y_test, y_pred_bool)
    roc = roc_auc_score(y_test, y_pred_bool)
    
    
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, "roc": roc}

    run['performance metrics'] = metrics

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
        
