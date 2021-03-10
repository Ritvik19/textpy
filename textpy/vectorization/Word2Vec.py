from collections import defaultdict

import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.auto import tqdm


class Word2VecVectorizer:
    """Create Word2Vec Embeddings for a corpus

    Args:
    model (string): space language model model, default: en
    embedding_size (int): dimensions of embedding vectors, default: 300
    window (int): size of context window, default: 5
    min_count (int): the minimum number of occurences for a word to considered to train embeddings, default : 5
    algo (0 or 1): 0 -> CBOW 1-> Skip gram, default: 0
    """

    def __init__(
        self, model="en", embedding_size=300, window=5, min_count=5, algo=0, **kwargs
    ):
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        self.algo = algo
        self.kwargs = kwargs

        nlp = spacy.load(model)
        self.tokenizer = Tokenizer(nlp.vocab)

    def _tokenize(self, X):
        return [
            [str(w) for w in doc]
            for doc in tqdm(self.tokenizer.pipe([str(d) for d in X]))
        ]

    def _transform_tokenized(self, X):
        vect = []
        for doc in tqdm(X):
            temp = []
            for word in doc:
                try:
                    temp.append(self.model.wv[word])
                except KeyError:
                    temp.append(np.zeros(self.embedding_size))
            vect.append(np.mean(temp, axis=0))
        return np.array(vect)

    def fit(self, X):
        tokenized_data = self._tokenize(X)
        self.model = Word2Vec(
            tokenized_data,
            size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            workers=-1,
            sg=self.algo,
            **self.kwargs
        )
        return self

    def transform(self, X):
        return self._transform_tokenized(self._tokenize(X))

    def fit_transform(self, X):
        tokenized_data = self._tokenize(X)
        self.model = Word2Vec(
            tokenized_data,
            size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            workers=-1,
            sg=self.algo,
            **self.kwargs
        )
        return self._transform_tokenized(tokenized_data)

    def __getitem__(self, key):
        return self.model.wv[key]


class IdfWeightedWord2VecVectorizer:
    """Create IDF weighted Word2Vec Embeddings for a corpus

    Args:
    model (string): space language model model, default: en
    embedding_size (int): dimensions of embedding vectors, default: 300
    window (int): size of context window, default: 5
    min_count (int): the minimum number of occurences for a word to considered to train embeddings, default : 5
    algo (0 or 1): 0 -> CBOW 1-> Skip gram, default: 0
    """

    def __init__(
        self, model="en", embedding_size=300, window=5, min_count=5, algo=0, **kwargs
    ):
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        self.algo = algo
        self.kwargs = kwargs

        nlp = spacy.load(model)
        self.tokenizer = Tokenizer(nlp.vocab)

    def _tokenize(self, X):
        return [
            [str(w) for w in doc]
            for doc in tqdm(self.tokenizer.pipe([str(d) for d in X]))
        ]

    def _fit_idf(self, x):
        t_vect = TfidfVectorizer().fit(x)
        self.idf = defaultdict(lambda: max(t_vect.idf_))
        for w, i in t_vect.vocabulary_.items():
            self.idf[w] = t_vect.idf_[i]
        return self

    def _transform_tokenized(self, X):
        return np.array(
            [
                np.mean(
                    [
                        self.word2vec[w] * self.idf[w]
                        for w in words
                        if w in self.word2vec
                    ]
                    or [np.zeros(self.embedding_size)],
                    axis=0,
                )
                for words in tqdm(X)
            ]
        )

    def fit(self, X, fit_idf=True):
        if fit_idf:
            self._fit_idf(X)
        tokenized_data = self._tokenize(X)
        model = Word2Vec(
            tokenized_data,
            size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            workers=-1,
            sg=self.algo,
            **self.kwargs
        )
        self.word2vec = {
            w: vec for w, vec in zip(model.wv.index2word, model.wv.vectors)
        }
        return self

    def transform(self, X):
        return self._transform_tokenized(self._tokenize(X))

    def fit_transform(self, X):
        tokenized_data = self._tokenize(X)
        self.model = Word2Vec(
            tokenized_data,
            size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            workers=-1,
            sg=self.algo,
            **self.kwargs
        )
        return self._transform_tokenized(tokenized_data)

    def __getitem__(self, key):
        return self.word2vec[key]
