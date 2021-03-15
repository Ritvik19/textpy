import numpy as np
from tqdm.auto import tqdm
from gensim.models.keyedvectors import KeyedVectors
from tensorflow.keras import initializers


class PreTrainedEmbeddings:
    """A keras initializer to fine tune pretrained embeddings

    Args:
        vectors_path (str): path to embeddings
        seq_vectorizer (vectorization.SequenceVectorizer)
        embedding_dim (int): Dimension of Embeddings, Default: 300

    Vectors: https://www.kaggle.com/iezepov/gensim-embeddings-dataset
    """

    def __init__(self, vectors_path, seq_vectorizer, embedding_dim=300):
        embeddings_index = KeyedVectors.load(vectors_path, mmap="r")

        embedding_dim = embedding_dim
        seq_vectorizer = seq_vectorizer

        count_known = 0
        count_unknown = 0

        self.embedding_matrix = np.zeros(
            (seq_vectorizer.tokenizer.num_words, embedding_dim)
        )

        for word, i in tqdm(seq_vectorizer.tokenizer.word_index.items()):
            if i >= seq_vectorizer.tokenizer.num_words:
                continue
            embedding_vector = None
            try:
                embedding_vector = embeddings_index[word]
            except KeyError:
                pass
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
                count_known += 1
            else:
                self.embedding_matrix[i] = np.random.randn(embedding_dim)
                count_unknown += 1

        print(f"{count_known} known vectors\n{count_unknown} random vectors")

    def __call__(self):
        return initializers.Constant(self.embedding_matrix)
