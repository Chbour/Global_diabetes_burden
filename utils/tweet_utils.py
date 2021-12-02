"""
Author: Adrian Ahne
Date: 14-08-2018

Helping functions and classes to handle / process tweets

"""

import numpy as np
import pandas as pd
import os.path as op
from sklearn.pipeline import Pipeline
from sys_utils import *



# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
load_library(op.join(basename, 'preprocess'))

from sklearn_utils import *


from preprocess import Preprocess
prep = Preprocess()


def preprocess_tweet(tweet, mode="no_preprocessing"):
    """
        Preprocess tweets in the same way the word embeddings (Word2Vec) are trained

        parameters
        -----------------------------------------------------------
        mode: ("full_preprocessing", "partial_preprocessing", "no_preprocessing")
            - "full_preprocessing" : apply all preprocessing functions
            - "partial_preprocessing" : apply only some preprocessing functions
            - "no_preprocessing" : apply only tokenisation
    """
    modes = ["full_preprocessing", "partial_preprocessing", "no_preprocessing"]

    if mode not in modes:
        print("ERROR: Given mode {} not one of the options: {}".format(mode, modes))
        sys.exit(1)


    if mode == "full_preprocessing":
        tweet = prep.replace_contractions(tweet)
        tweet = prep.replace_special_words(tweet)
        tweet = prep.replace_hashtags_URL_USER(tweet, mode_URL="delete", mode_Mentions="replace",
                                               mode_Hashtag="replace")
        tweet = prep.tokenize(tweet)
        tweet = prep.remove_punctuation(tweet)
        tweet = prep.preprocess_emojis(tweet)
        tweet = prep.preprocess_emoticons(tweet)
        tweet = prep.remove_non_ascii(tweet)
        tweet = prep.to_lowercase(tweet)
        tweet = prep.replace_numbers(tweet)
        tweet = prep.remove_stopwords(tweet, include_personal_words=True, include_negations=False)
        tweet = prep.lemmatize_verbs(tweet)
        tweet = prep.stem_words(tweet)

        return tweet

    if mode == "partial_preprocessing":
    #    tweet = prep.replace_contractions(tweet)
    #    tweet = prep.replace_special_words(tweet)
        tweet = prep.replace_hashtags_URL_USER(tweet, mode_URL="delete", mode_Mentions="replace",
                                               mode_Hashtag="replace")
        tweet = prep.tokenize(tweet)
        tweet = prep.remove_punctuation(tweet)
    #    tweet = prep.preprocess_emojis(tweet)
    #    tweet = prep.preprocess_emoticons(tweet)
        tweet = prep.remove_non_ascii(tweet)
        tweet = prep.to_lowercase(tweet)
        tweet = prep.replace_numbers(tweet)
    #    tweet = prep.remove_stopwords(tweet, include_personal_words=True, include_negations=False)
        tweet = prep.lemmatize_verbs(tweet)
    #    tweet = prep.stem_words(tweet)
        return tweet

    if mode == "no_preprocessing":
        tweet = prep.tokenize(tweet)
        return tweet




def tweet_vectorizer(tweet, model):
    """
        Gets word embedding vector for each word in the tweet and calculates
        word embedding for the whole tweet by taking the mean of all word - vectors

        Parameters
        -------------------------------------------------------------------
        tweet:      list of preprocessed tweets
        model:      model with trained word embeddings

        Return
        ---------------------------------------------------------------------
        list containing word embeddings for each tweet
    """

    tweet_vec =[]
    numw = 0
    for w in tweet:
        try:
            if numw == 0:
                tweet_vec = model[w]
            else:
                tweet_vec = np.add(tweet_vec, model[w])
            numw+=1

        except:
            print("no embedding for {} !!!!!!!!!!!!".format(w))

    if tweet_vec == []: return np.zeros((model.vector_size, ))
    else: return np.asarray(tweet_vec) / numw


def get_meta_data_features(tweets_csv, manually_labelled_tweets):
    """
        Get some meta-data of the labeled tweets

        Parameter:
          - tweets_csv : DataFrame with labeled tweets
          - manually_labelled_tweets : Collection in which all raw tweet information
                                       of the labeled tweets are stored
    """

    # define DataFrame
    meta_data_pd = pd.DataFrame(columns=["n_hashtags", "n_urls", "n_user_mentions",
                                         "followers_count", "friends_count"])

    for i, user_name in enumerate(tweets_csv["user_name"]):
        for user in manually_labelled_tweets.find({'user.screen_name' : user_name}):
            meta_data_pd.loc[i] = [len(user["entities"]['hashtags']),
                                   len(user["entities"]['urls']),
                                   len(user["entities"]['user_mentions']),
                                   user["user"]['followers_count'],
                                   user["user"]['friends_count']]

    return meta_data_pd



def do_nothing(tweet):
    """
        Just to overwrite the preprocessing and tokenisation of TfidfVectorizer
    """
    return tweet


def create_pipeline_BoW(model, meta_data=[], user_description=[]):
    """
        Create Pipeline

        Parameters
        -------------------------------------------------------
        - model : algorithm for classification to be used
        - meta_data : meta_data information like number of followers / friends etc.
        - user_description : preprocessed tokens of user's description in Twitter

        Return
        --------------------------------------------------------
        pipeline object
    """

    # meta data given but no user description
    if meta_data != [] and user_description == []:
        print("Create pipeline using meta-data...")
        pipeline  = Pipeline([
            # combine tweets and meta-data with their labels
            ('textMetaDataFeatureExtractor', TextAndMetaDataFeatureExtractor(meta_data=meta_data)),

            ('union', FeatureUnion(
                transformer_list = [

                    # Pipeline handling the tweets
                    ('tweets', Pipeline([
                        ('tweetsSelector', ItemSelector(key='tweet')),
                        ('tfidfvect', TfidfVectorizer(preprocessor=do_nothing, tokenizer=do_nothing))
                    ])),

                    # Pipeline handling meta data
                    ('metadata', Pipeline([
                        ('metadataSelector', ItemSelector(key='metadata')),
                        ('tosparse', ArrayCaster()),
                        ('scale', StandardScaler(with_mean=False)),
                        ('selectKbest', SelectKBest(f_classif)),
                    ]))
                ]
            )),

            ('model', model),
        ])

    # meta data and user description given
    elif meta_data != [] and user_description != []:
        print("Create pipeline using meta-data and Twitter's user description...")

        pipeline  = Pipeline([
            # combine tweets and meta-data with their labels
            ('textMetaDataFeatureExtractor', TextAndMetaDataFeatureExtractor(meta_data=meta_data,
                                                                             user_description=user_description)),

            ('union', FeatureUnion(
                transformer_list = [

                    # Pipeline handling the tweets
                    ('tweets', Pipeline([
                        ('tweetsSelector', ItemSelector(key='tweet')),
                        ('tfidfvect', TfidfVectorizer(preprocessor=do_nothing, tokenizer=do_nothing))
                    ])),

                    # Pipeline handling meta data
                    ('metadata', Pipeline([
                        ('metadataSelector', ItemSelector(key='metadata')),
                        ('tosparse', ArrayCaster()),
                        ('scale', StandardScaler(with_mean=False)),
                        ('selectKbest', SelectKBest(f_classif)),
                    ])),

                    # Pipeline handling the description
                    ('desc', Pipeline([
                        ('descSelector', ItemSelector(key='userDescription')),
                        ('tfidfvect', TfidfVectorizer(preprocessor=do_nothing, tokenizer=do_nothing)),
                        #('Debug1', Debug("desc*****")),

                    ]))
                ]
            )),

            ('model', model),
        ])

    # no meta data given but user description given
    elif meta_data == [] and user_description != []:
        print("Create pipeline using Twitter's user description...")

        pipeline  = Pipeline([
            # combine tweets and meta-data with their labels
            ('textMetaDataFeatureExtractor', TextAndMetaDataFeatureExtractor(user_description=user_description)),

            ('union', FeatureUnion(
                transformer_list = [

                    # Pipeline handling the tweets
                    ('tweets', Pipeline([
                        ('tweetsSelector', ItemSelector(key='tweet')),
                        ('tfidfvect', TfidfVectorizer(preprocessor=do_nothing, tokenizer=do_nothing)),
                    ])),

                    # Pipeline handling the description
                    ('desc', Pipeline([
                        ('descSelector', ItemSelector(key='userDescription')),
                        ('tfidfvect', TfidfVectorizer(preprocessor=do_nothing, tokenizer=do_nothing)),
                        #('Debug1', Debug("desc*****")),

                    ]))
                ]
            )),

            ('model', model),
        ])

    # no meta data and no user description
    else:
        print("Create pipeline...")

        pipeline  = Pipeline([
            # combine tweets and meta-data with their labels
            ('textMetaDataFeatureExtractor', TextAndMetaDataFeatureExtractor()),

            ('union', FeatureUnion(
                transformer_list = [

                    # Pipeline handling the tweets
                    ('tweets', Pipeline([
                        ('tweetsSelector', ItemSelector(key='tweet')),
                        ('tfidfvect', TfidfVectorizer(preprocessor=do_nothing, tokenizer=do_nothing)),
                    ]))
                ]
            )),

            ('model', model),
        ])

    return pipeline
