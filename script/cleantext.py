from sklearn.base import BaseEstimator, TransformerMixin
import nltk

nltk.download('stopwords')


class Minuscule(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self._feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self._feature] = X[self._feature].str.lower()
        return X


# Remove HTML items
class RemoveHTML(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self._feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self._feature] = X[self._feature].str.replace('<.*?>', ' ', regex=True)
        return X


# Remove URL
class RemoveURL(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self._feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self._feature] = X[self._feature].str.replace('https?://\S+|www\.\S+', ' ', regex=True)
        return X


# Remove ponctuation
class RemovePonctuation(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self._feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self._feature] = X[self._feature].str.replace('[^a-z]', ' ', regex=True)
        return X


# Transform in list of tokens
from nltk.corpus import stopwords


class Tokens(BaseEstimator, TransformerMixin):
    def __init__(self, feature, stemming=True):
        self._feature = feature
        self._stops = set(stopwords.words('english'))
        self._stops.add('also')
        self._stemmer = nltk.stem.SnowballStemmer('english')
        self._stemming = stemming

    def fit(self, X, y=None):
        return self

    def remove_stops(self, row):
        meaningful_words = [w for w in row[self._feature] if not w in self._stops]
        return meaningful_words

    def stem(self, row):
        stem_words = [self._stemmer.stem(token) for token in row[self._feature]]
        return stem_words

    def transform(self, X, y=None):
        X[self._feature] = X[self._feature].str.split()
        X[self._feature] = X.apply(self.remove_stops, axis=1)
        if self._stemming:
            X[self._feature] = X.apply(self.stem, axis=1)
        return X


class ListIntoSentence(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self._feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self._feature] = [[' '.join(w)] for w in X[self._feature].values]
        X[self._feature] = X[self._feature].str[0]
        return X


class RemovePonctuationBert(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self._feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self._feature] = X[self._feature].str.replace('[^a-z0-9.]', ' ', regex=True)
        return X


# on enlève le dernier "." car on rajoute un [SEP] tout à la fin
# if "." précédé d'une seule lettre c'est une abréviation et une une fin de phrase

class SentenceBert(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self._feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self._feature] = X[self._feature].str.replace('[.]', ' [SEP] ', regex=True)
        X[self._feature] = '[CLS] ' + X[self._feature]
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self._features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.drop(self._features, axis=1)
        return X
