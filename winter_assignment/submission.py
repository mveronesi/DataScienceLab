import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp import LemmaTokenizer
from sklearn.linear_model import LogisticRegression

df_dev = pd.read_csv("dataset/development.csv", index_col="ids")
df_dev.drop(columns=["date", "flag", "user"], inplace=True)
df_dev.drop_duplicates(keep='first' ,inplace=True)


df_eval = pd.read_csv("dataset/evaluation.csv")
df_eval.drop(columns=["ids", "date", "flag", "user"], inplace=True)

# rimozione tweet duplicati con sentiment diverso
c = df_dev.index.value_counts()
c = c[c > 1]
indexes_to_remove = c.index.values
df_dev.drop(labels=indexes_to_remove, inplace=True)

lemmaTokenizer = LemmaTokenizer()
vectorizer = TfidfVectorizer(stop_words='english', tokenizer=lemmaTokenizer)
vectorizer.fit(df_dev["text"].values)
Xdata = vectorizer.transform(df_dev["text"].values)
Ydata = df_dev["sentiment"].values
model = LogisticRegression(solver='sag', max_iter=200)
model.fit(Xdata, Ydata)

ids = df_eval.index.values.copy().reshape(-1, 1)
Xdata_eval = vectorizer.transform(df_eval["text"])
Ydata_eval = np.round(model.predict(Xdata_eval), decimals=0).reshape(-1, 1)
result = pd.DataFrame(np.hstack((ids, Ydata_eval)), columns=["Id", "Predicted"])
result.to_csv("./submission.csv", index=False)