from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
import re, spacy


class TextCleaner(BaseEstimator, TransformerMixin):
    """A general purpose text cleaning pipeline which utilizes `spacy` and regex to:
        * lower cases the text
        * removes urls and emails
        * removes html css and js
        * removes stop words
        * performs lemmatization
        * removes numbers, punctuations
        * trims white spaces

    Args:
        model (str): spacy language model, default: en
    """

    def __init__(self, model="en"):
        self.nlp = spacy.load(model, disable=["parser", "ner"])

    def fit(self, X=None):
        return self

    def transform(self, X):
        transformed = []
        for x in tqdm(X):
            x = str(x).strip().lower()  # Lower case the data
            x = re.sub(r"""((http[s]?://)[^ <>'"{}|\^`[\]]*)""", r" ", x)  # remove urls
            x = re.sub(
                r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", r" ", x
            )  # remove emails
            x = re.sub(r"<style.*>[\s\S]+</style>", " ", x)  # remove css
            x = re.sub(r"<script.*>[\s\S]*</script>", " ", x)  # remove js
            x = re.sub(
                r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});", " ", x
            )  # remove html

            if len(x) != 0:
                parsed = self.nlp(x)
                lemmatized = " ".join([w.lemma_ for w in parsed if not w.is_stop])

                # Remove punct
                punct_removed = re.sub(r"\W", " ", str(lemmatized))
                punct_removed = re.sub(r"\d", " ", str(punct_removed))
                punct_removed = re.sub(r"\s+", " ", str(punct_removed))
            else:
                punct_removed = x
            transformed.append(punct_removed)
        return transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)