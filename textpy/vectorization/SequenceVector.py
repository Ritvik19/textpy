from tensorflow.keras.preprocessing.text import Tokenizer as SeqTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin


class SequenceVectorizer(BaseEstimator, TransformerMixin):
    """Vectorize a text corpus, by turning each text into either a sequence of integers of a fixed length

    Args:
        vocab_size (int): max number of unique words in the corpus
        max_seq_len (int): max number of words in a sequence
        oov_token (string): Out of Vocabulary token,
            a special token which is used when a new word other than the vocabulary is encountered,
            Default: <unk>
        char_level, default False, it True, tokenizes the data on character level
        padding ('pre' or 'post'): pad either before or after each sequence
        truncating ('pre' or 'post'): remove values from sequences larger than max_seq_len,
            either at the beginning or at the end of the sequences

    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        oov_token="<unk>",
        char_level=False,
        padding="pre",
        truncating="pre",
    ):
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.truncating = truncating
        self.tokenizer = SeqTokenizer(
            num_words=vocab_size, oov_token=oov_token, char_level=char_level
        )

    def fit(self, X):
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X):
        return pad_sequences(
            self.tokenizer.texts_to_sequences(X),
            maxlen=self.max_seq_len,
            padding=self.padding,
            truncating=self.truncating,
        )