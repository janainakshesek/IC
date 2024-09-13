import numpy as np
import pandas as pd

data_imdb = pd.read_csv("imdb_labelled.txt", delimiter='\t', header=None)
data_imdb.columns = ["Review_text", "Review_class"]

data_amazon = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t', header=None)
data_amazon.columns = ["Review_text", "Review_class"]

data_yelp = pd.read_csv("yelp_labelled.txt", delimiter='\t', header=None)
data_yelp.columns = ["Review_text", "Review_class"]

data = pd.concat([data_imdb, data_amazon, data_yelp])

import re
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_text(df):
    all_reviews = list()
    lines = df["Review_text"].values.tolist()
    for text in lines:
        text = text.lower()
        # Elimina links
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text)
        # Remove os símbolos
        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
        tokens = word_tokenize(text)
        # Remover pontuação
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        PS = PorterStemmer()
        words = [PS.stem(w) for w in words if not w in stop_words]
        words = ' '.join(words)
        all_reviews.append(words)
    return all_reviews

all_reviews = clean_text(data)
all_reviews[0:20]

stopwords.words("english")

from sklearn.feature_extraction.text import CountVectorizer
# Transforma texto em vetores de contagem de tokens.
CV = CountVectorizer(min_df=3)
# Transforma o documento em matriz de termos
X = CV.fit_transform(all_reviews).toarray()
y = data[["Review_class"]].to_numpy()
print(np.shape(X))
print(np.shape(y))
