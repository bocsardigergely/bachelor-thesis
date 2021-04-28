import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_score, f1_score, roc_auc_score, accuracy_score, recall_score
from sklearn.svm import LinearSVC
import neptune.new as neptune


run = neptune.init(project='bocsardigergely/thesis',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZTlkNzc4NC1iYTZjLTQwODktODRiOS1iZjc2NjM3MDQxNWMifQ==')

df_short = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_short.csv")
df_medium = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_medium.csv")
df_dank = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/processed_dank.csv")

df_imdb = pd.read_csv("https://raw.githubusercontent.com/bocsardigergely/bachelor-thesis/main/data/processed/full_proc_imdb.csv")


df_medium_half = df_medium.groupby('label').apply(lambda x: x.sample(frac=0.5)).sample(frac=1).reset_index(drop=True)
df_dank_half = df_dank.groupby('label').apply(lambda x: x.sample(frac=0.5)).sample(frac=1).reset_index(drop=True)
df_short_half = df_short.groupby('label').apply(lambda x: x.sample(frac=0.5)).sample(frac=1).reset_index(drop=True)

tfidf = TfidfVectorizer()


def model_trial(df):
    name =[x for x in globals() if globals()[x] is df][0]
    #creating the desired vectors
    text = df['text']
    y = df["label"]
    text_train, text_test, y_train, y_test = train_test_split(
    text, y, test_size=0.33, random_state=42)
    X_train = tfidf.fit_transform(text_train.astype('U').values)
    X_test = tfidf.transform(text_test.astype('U').values)
    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test = xgb. DMatrix(X_test, label=y_test)



    booster = xgb.XGBClassifier()

    param = {
    'eta': 0.3, 
    'max_depth': 6,  
    'objective': 'binary:hinge'} 
    steps = 15

    model = xgb.train(param, D_train, steps)

    y_pred_bool = model.predict(D_test)
    accuracy = accuracy_score(y_test, y_pred_bool)
    precision = precision_score(y_test, y_pred_bool)
    recall = recall_score(y_test, y_pred_bool)
    roc = roc_auc_score(y_test, y_pred_bool)
    
    
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, "roc": roc}
    run['performance metrics xgb'] = metrics



    svm = LinearSVC()
    svm.fit(X_train, y_train)
    y_pred_bool = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_bool)
    precision = precision_score(y_test, y_pred_bool)
    recall = recall_score(y_test, y_pred_bool)
    roc = roc_auc_score(y_test, y_pred_bool)
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, "roc": roc}
    run['performance metrics svm'] = metrics

model_trial(df_imdb)