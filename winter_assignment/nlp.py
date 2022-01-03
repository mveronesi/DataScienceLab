from nltk.tokenize import wordpunct_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, document):
        lemmas = []
        for t in wordpunct_tokenize(document):
            t = t.strip()
            lemma = self.lemmatizer.lemmatize(t)
            if lemma.isalpha() and len(lemma)>2:
                lemmas.append(lemma)
        return lemmas
