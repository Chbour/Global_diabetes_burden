"""
Author: Adrian Ahne
Date: 27-06-2018

Helping functions and classes for the classification part

"""

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np


class TextAndMetaDataFeatureExtractor(BaseEstimator, TransformerMixin):
    """ Combine tweets (label tweet) and meta-data (label metadata)
        such as number of followers, number of friends, etc. and the description of the user

        Parameters
        ------------------------------------------------------

        Optional:
            meta_data : array table containing meta-data information (nFollowers, nFriends, etc.)
            user_description : Preprocessed text tokens of the user description of the user's metadata

        Return
        -----------------------------------------------------
        Numpy's recarray containing up to three different types (tweet, metadata, userDescription)
    """

    def __init__(self, meta_data=[], user_description=[]):
        self.meta_data = meta_data
        self.user_description = user_description

    def fit(self, tweets, y=None):
        return self

    def transform(self, tweets):

        # add metadata if true and not user description
        if self.meta_data != [] and self.user_description == []:
            features = np.recarray(shape=(len(tweets),),
                                   dtype=[('tweet', object), ('metadata', object)])

            for i, tweet in enumerate(tweets):
                features['tweet'][i] = tweet
                features['metadata'][i] = np.array(self.meta_data[i].tolist())#.iloc[[i]]

        # add metadata if true and user description
        elif self.meta_data != [] and self.user_description != []:
            features = np.recarray(shape=(len(tweets),),
                                   dtype=[('tweet', object), ('metadata', object), ('userDescription', object)])

            for i, tweet in enumerate(tweets):
                features['tweet'][i] = tweet
                features['metadata'][i] = np.array(self.meta_data[i].tolist())#.iloc[[i]]
                features['userDescription'][i] = self.user_description[i]

        # add no metadata but user description
        elif self.meta_data == [] and self.user_description != []:
            features = np.recarray(shape=(len(tweets),),
                                   dtype=[('tweet', object), ('userDescription', object)])

            for i, tweet in enumerate(tweets):
                features['tweet'][i] = tweet
                features['userDescription'][i] = self.user_description[i]

        # only work with tweets
        else:
            features = np.recarray(shape=(len(tweets),),
                                   dtype=[('tweet', object)])

            for i, tweet in enumerate(tweets):
                features['tweet'][i] = tweet

        return features

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class ItemSelect(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return np.asarray(data[self.key].values.tolist())


class ArrayCaster(BaseEstimator, TransformerMixin):
    """ Transposes meta data matrix so it fits to the tweet matrix in the feature union (pipeline)"""
    def fit(self, x, y=None):
        return self

    def transform(self, metadata):
        metadata = metadata.tolist()
        return csr_matrix(metadata)



class Debug(BaseEstimator, TransformerMixin):
    """
        Debugging in the pipeline
    """

    def __init__(self, message=""):
        self.message = message

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        print("{}".format(self.message))
        print("shape:", X.shape, " type:", type(X))
        import ipdb; ipdb.set_trace()
        #print(X[0])
        return X


class CustomVectorizer(CountVectorizer):
    """
        overwrites the build_analyzer method of scikit-learn's CountVectorizer,
        allowing one to create a custom analyzer for the vectorizer
    """

    def build_analyzer(self):
        # load stop words using CountVectorizer's built in method
        stop_words = self.get_stop_words()

        # create the analyzer that will be returned by this method
        def analyser(tweet):
            #print("tweet")
            #print(tweet)

            tweet = prep.replace_contractions(tweet)
            tweet = prep.replace_hashtags_URL_USER(tweet)
            tweet = prep.tokenize(tweet)
            tweet = prep.remove_punctuation(tweet)

            tweet = prep.preprocess_emojis(tweet)
            tweet = prep.preprocess_emoticons(tweet)
            tweet = prep.remove_non_ascii(tweet)
            tweet = prep.to_lowercase(tweet)

            tweet = prep.remove_stopwords(tweet)
            tweet = prep.lemmatize_verbs(tweet)
            tweet = prep.stem_words(tweet)
#                tweet = [ x for x in tweet if x not in["diabet", "glucos", "insulin", "type", "1", "2", "", "get", "sugar", "would",
#                                                      "go", "know", "take", "give", "say", "one", "could", "would", "people", "look",
#                                                      "year", "test", "see", "oh", "via", "bitch", "daddi", "hi", "w", "b", "n", "c",
#                                                      "ii", "dr", "rt", "bc", "ok", "think", "make"] ]

            return tweet

        return(analyser)
