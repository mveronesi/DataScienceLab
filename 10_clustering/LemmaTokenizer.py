from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

class LemmaTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, document):
        lemmas = []
        for t in word_tokenize(document):
            lemmas.append(self.lemmatizer.lemmatize(t.strip()))
        return lemmas