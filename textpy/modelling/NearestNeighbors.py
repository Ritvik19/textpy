import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors, NearestCentroid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity


class SimilarNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """A learner that finds 'n' most Similar Cluster Centroids for a given text

    Args:
        vectorizer (sklearn TfidfVectorizer or CountVectorizer) used to create the vector space
    """

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X, y, fit_vectorizer=True):
        if fit_vectorizer:
            self.vectorizer = self.vectorizer.fit(X.fillna(""))
        vectors = self.vectorizer.transform(X.fillna(""))
        self.labels = list(sorted(y.unique()))
        self.model = NearestNeighbors(metric="cosine").fit(
            csr_matrix(
                np.concatenate(
                    [vectors[(y[y == val].index)].mean(axis=0) for val in self.labels]
                )
            )
        )
        return self

    def fit_vectorizer(self, X):
        self.vectorizer = self.vectorizer.fit(X.fillna(""))

    def predict(self, X):
        return np.array(
            list(
                map(
                    lambda x: self.labels[x[0]],
                    self.model.kneighbors(
                        self.vectorizer.transform(X.fillna("")),
                        n_neighbors=1,
                        return_distance=False,
                    ),
                )
            )
        )

    def predict_top_n(self, X, n=5):
        distances, indices = self.model.kneighbors(
            self.vectorizer.transform(X.fillna("")), n_neighbors=n
        )
        similarity = 1 - np.ravel(distances)
        indices = np.ravel(indices)
        preds = np.array(list(map(lambda x: self.labels[x[0]], indices)))
        return similarity, preds


class RocchioClassifier(BaseEstimator, ClassifierMixin):
    """A learner that finds the most Similar Cluster Centroid for a given text

    Args:
        vectorizer (sklearn TfidfVectorizer or CountVectorizer) used to create the vector space
    """

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X, y, fit_vectorizer=True):
        if fit_vectorizer:
            self.vectorizer = self.vectorizer.fit(X.fillna(""))
        vectors = self.vectorizer.transform(X.fillna(""))
        self.model = NearestCentroid(metric="cosine").fit(vectors, y)
        return self

    def fit_vectorizer(self, X):
        self.vectorizer = self.vectorizer.fit(X.fillna(""))

    def predict(self, X):
        return self.model.predict(self.vectorizer.transform(X.fillna("")))
