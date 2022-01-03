import csv
import string
import math

def tokenize(doc):
    """Compute the tokens for a document.
    
    Input: a string, e.g. a document to tokenize.
    Output: a list. Each item is a tokens of the relative document
    """
    for punct in string.punctuation:
        doc = doc.replace(punct, " ") # remove punctuation
    split_doc = [token.lower() for token in doc.split(" ") if token]
    return split_doc

def TF(tokens):
    """Compute the term frequency for the document given in input.
    
    Input: a list of word representing the document
    Output: a dictionary with all the frequencies
    """
    frequencies = {}
    for word in tokens:
        frequencies[word] = frequencies[word] + 1 if word in frequencies else 1
    return frequencies

def DF(TF_s):
    """Compute the document frequency of a dictionary of tokens, i.e. the number of
    documents in which t appears at least once

    Input: a list of dictionaries, each one is a TF for a document
    Output: a dictionary representing the DF for each token
    """
    DF_s = {}
    for document in TF_s:
        for key in document.keys():
            DF_s[key] = DF_s[key] + 1 if key in DF_s.keys() else 1
    return DF_s

def IDF(N, DF_t):
    """Compute the IDF of a document given the number of documents in the corpus
    and the document frequency of a token

    Input: N number of document in the corpus, DF_t the value DF for the token t
    Output: the value of the IDF_t
    
    """
    return math.log(N / DF_t)

def norm(d):
    """Compute the L2-norm of a vector representation"""
    return sum([tf_idf**2 for t, tf_idf in d.items()]) ** 0.5

def dot_product(d1, d2):
    """Compute the dot product between two vector representations"""
    word_set = set(list(d1.keys()) + list(d2.keys()))
    return sum([(d1.get(d, 0.0) * d2.get(d, 0.0)) for d in word_set])

def cosine_similarity(d1, d2):
    """Compute the cosine similarity between documents d1 and d2.

    Input: two dictionaries representing the TF-IDF vectors for documents d1 and d2.
    Output: the cosine similarity.
    """
    return dot_product(d1, d2) / (norm(d1) * norm(d2))

########################## SCRIPT ###############################
dataset = []
with open("../datasets/aclimdb_reviews_train.csv") as f:
    next(f)
    for row in csv.reader(f):
        dataset.append({
            "review": row[0],
            "label": int(row[1])
        })

documents_tokenized = []
TF_s = []
for item in dataset:
    tokens = tokenize(item["review"])
    documents_tokenized.append(tokens)
    TF_s.append(TF(tokens))

DF_s = DF(TF_s)
IDF_s = {}
for DF_t in DF_s.keys():
    IDF_s[DF_t] = IDF(len(dataset), DF_s[DF_t])

# sort dictionary IDF_s by its by values
res = {key: val for key, val in sorted(IDF_s.items(), key = lambda ele: ele[1], reverse = True)}

TF_IDF = []  # contains vector representation for each document
for item in TF_s:
    tmp = {}
    for key in item.keys():
        tmp[key] = item[key] * IDF_s[key]
    TF_IDF.append(tmp)
