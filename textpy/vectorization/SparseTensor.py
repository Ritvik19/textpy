import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin


class ConvertToSparseTensor(BaseEstimator, TransformerMixin):
    """A utiliy to convert sparse vectors into sparse tensors"""

    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))
