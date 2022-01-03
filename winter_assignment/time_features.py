import pandas as pd
import numpy as np
import numpy.random as nr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from nlp import LemmaTokenizer
from sklearn.model_selection import KFold
from tqdm import tqdm
import scipy.sparse
from os.path import exists
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
RANDOM_STATE = 42
nr.seed(RANDOM_STATE)


df_dev = pd.read_csv("dataset/development.csv", index_col="ids", infer_datetime_format=True, parse_dates=[2])
df_dev.drop_duplicates(keep='first', inplace=True)
c = df_dev.index.value_counts()
c = c[c > 1]
indexes_to_remove = c.index.values
df_dev.drop(labels=indexes_to_remove, inplace=True)
df_dev["month"] = df_dev["date"].apply(lambda x: x.month_name())
df_dev["weekday"] = df_dev["date"].apply(lambda x: x.weekday())
df_dev["hour"] = df_dev["date"].apply(lambda x: x.hour)
df_dev.drop(columns=["flag", "user", "date"], inplace=True)
df_dev = pd.get_dummies(df_dev, columns=["month", "weekday", "hour"])

Ydata = df_dev["sentiment"].values
Xdata_onehot = scipy.sparse.csr_matrix(df_dev.values[:, 3:], dtype=int)

kf = KFold(n_splits=5, shuffle=True)
Ypred_tot = np.empty_like(Ydata)

for train_indexes, test_indexes in tqdm(kf.split(Xdata_onehot, Ydata)):
    Xtrain, Xtest = Xdata_onehot[train_indexes], Xdata_onehot[test_indexes]
    Ytrain, Ytest = Ydata[train_indexes], Ydata[test_indexes]
    
    lemmaTokenizer = LemmaTokenizer()
    vectorizer = TfidfVectorizer(stop_words="english", tokenizer=lemmaTokenizer, min_df=0.001)
    vectorizer.fit(df_dev["text"].values[train_indexes])
    Xtrain = scipy.sparse.hstack((vectorizer.transform(df_dev["text"].values[train_indexes]), Xtrain), format='csr')
    Xtest = scipy.sparse.hstack((vectorizer.transform(df_dev["text"].values[test_indexes]), Xtest), format='csr')
    
    # up samp 
    """Xt1 = Xtrain[Ytrain == 1]
    Xt0_idx = nr.choice(np.arange(Xtrain[Ytest == 0].shape[0]), size=Xt1.shape[0], replace=True)
    Xt0 = Xtrain[Xt0_idx]
    Xtrain, Ytrain = scipy.sparse.vstack([Xt0, Xt1], format='csr'), np.concatenate([np.zeros(Xt0.shape[0]),np.ones(Xt1.shape[0])])"""
    print("Xtrain shape: ", Xtrain.shape)
    print("Xtest shape: ", Xtest.shape)
    print("Ytrain shape: ", Ytrain.shape)
    print("Ytest shape: ", Ytest.shape)
    
    #model = LogisticRegression(solver='saga', max_iter=200)
    model = LinearSVC()
    model.fit(Xtrain, Ytrain)
    Ypred_tot[test_indexes] = np.round(model.predict(Xtest), decimals=0).astype(int)
    
print(f1_score(Ypred_tot, Ydata))
