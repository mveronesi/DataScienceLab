import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nlp import LemmaTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tqdm import tqdm

df_dev = pd.read_csv("dataset/development.csv", index_col="ids")
df_dev.drop(columns=["date", "flag", "user"], inplace=True)
df_dev.drop_duplicates(keep='first' ,inplace=True)

# rimozione tweet duplicati con sentiment diverso
c = df_dev.index.value_counts()
c = c[c > 1]
indexes_to_remove = c.index.values
df_dev.drop(labels=indexes_to_remove, inplace=True)

lemmaTokenizer = LemmaTokenizer(correct=False)
vectorizer = TfidfVectorizer(stop_words='english', tokenizer=lemmaTokenizer)
vectorizer.fit(df_dev["text"].values)
Xdata = vectorizer.transform(df_dev["text"].values)
Ydata = df_dev["sentiment"].values

kf = KFold(n_splits=5, shuffle=True)
scores = []
for train_indexes, test_indexes in tqdm(kf.split(Xdata, Ydata)):
    Xtrain, Xtest = Xdata[train_indexes], Xdata[test_indexes]
    Ytrain, Ytest = Ydata[train_indexes], Ydata[test_indexes]
    model = LogisticRegression(solver='sag', max_iter=200)
    model.fit(Xtrain, Ytrain)
    Ypred = np.round(model.predict(Xtest), decimals=0).astype(int)
    scores.append(f1_score(Ytest, Ypred))
print(scores)
print(np.mean(scores))