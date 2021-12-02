"""
Definitions of basic constants

Author: Adrian Ahne
Creation date: 23/04/2018

Inspired by https://github.com/s/preprocessor/tree/master/preprocessor


- 24/08/2018 : Added Emotion terms
"""


import re
import nltk
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
# download library
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from emotion_codes import EMOTICONS_UNICODE
from stopword_def import *

from emotionKeywords import *
from emotion_codes import *
from stopwords_fr import *

class Constants:
    URL = "URL"
    USER = "USER"

    TYPE1 = "TYPEONE"
    TYPE2 = "TYPETWO"

class Patterns:
    URL_PATTERN=re.compile(r"http\S+") # or re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
    MENTION_PATTERN = re.compile(r"(?:@[\w_]+)")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    RESERVED_WORDS_PATTERN = re.compile(r'^(RT|FAV)') #TODO check for this

    EMOTICONS_PATTERN = re.compile(u'(' + u'|'.join(k for k in EMOTICONS_UNICODE) + u')', re.IGNORECASE)
    # TODO create EMOJI PATTERN




class Grammar:
    STOPWORDS_EN = stopwords.words('english')
    STOPWORDS_NO_PERSONAL_EN = stopwords_no_personal_list # excludes personal words like "I", "me", "my" to keep them when filtering personal from institutional tweets
    STOPWORDS_FR = stopwords_fr
    WHITELIST_EN = ["n't", "not", "no", "nor", "never", "nothing", "nowhere", "noone", "none"]
    STEMMER_LANCASTER = LancasterStemmer() # aggressive, fast, sometimes confusing
#    STEMMER_PORTER = PorterStemmer(mode='NLTK_EXTENSIONS') # mode that includes further improvements
    STEMMER_SNOWBALL_EN = SnowballStemmer('english') # improved porter
    STEMMER_SNOWBALL_FR = nltk.stem.snowball.FrenchStemmer()

    LEMMATIZER = WordNetLemmatizer()


class WordLists:
    type1 = ["#type1diabetes", "type one", "#type1", "type1", "Type1", "type 1", "T1", "t1",
             "Type 1", "t1d", "T1D", "#t1d", "#T1D", "typeI", "TypeI", "type I", "Type I"]
    type2 = ["#type2diabetes", "type two", "#type2", "type2", "Type2", "type 2", "T2", "t2",
             "Type 2", "t2d", "T2D", "#t2d", "#T2D", "typeII", "TypeII", "type II", "Type II"]

    TYPE1_WORDS = re.compile(u"|".join(type1))
    TYPE2_WORDS = re.compile(u"|".join(type2))

    excludeTweets = ["#chloegracemoretz",  "#chloegmoretz",  "#chloegrace",  "#chloemoretz",
                     "#moretzgrace", "#chloemoretz"] # list with words whose tweets are to be excluded


class Emotions:

    excludeSynonyms_ = ["go", "like", "get", "pull", "floor",  "know", "give",
                        "longer", "please", "handle", "dear", "deal", "substitute",
                        "crank", "lecture", "starve",  "soreness"]
    addWords_ = []

    parrotsEmotions = get_parrotsEmotions_list_all()
    allEmotions = get_parrotsEmotions_list_all()+DDS_list+PAID_list
    allEmotions_allForms_synonyms = get_synonyms(all_word_forms(allEmotions),
                                                excludeSynonyms_, addWords_)
    # list with all emotions and their synonyms (deleted some which were clearly not an emotion)
    emotions_full_list = emotion_key_words_fulllist

    def get_emotion_synonyms(emotion, excludeWords=excludeSynonyms_, addWords=addWords_,
                             all_emotions_list=emotions_full_list):
        return list(set(get_synonyms(all_word_forms(get_parrotsEmotions_for_emtion(emotion)),
                                                        excludeWords, addWords))
                                     & set(all_emotions_list))



    # take defintions from emotionKeywords
    emotions_synonyms_joy = emotions_synonyms_joy
    emotions_synonyms_love = emotions_synonyms_love
    emotions_synonyms_surprise = emotions_synonyms_surprise
    emotions_synonyms_anger = emotions_synonyms_anger
    emotions_synonyms_sadness = emotions_synonyms_sadness
    emotions_synonyms_fear = emotions_synonyms_fear

#    emotions_synonyms_joy = get_emotion_synonyms("joy")
#    emotions_synonyms_love = get_emotion_synonyms("love")
#    emotions_synonyms_surprise = get_emotion_synonyms("surprise")
#    emotions_synonyms_anger = get_emotion_synonyms("anger")
#    emotions_synonyms_sadness = get_emotion_synonyms("sadness")
#    emotions_synonyms_fear = get_emotion_synonyms("fear")

    EMOTION_CATEGORIES = Emotions_positive.CATEGORIES_POSITIVE+Emotions_negative.CATEGORES_NEGATIVE


ColumnNames = {
    "id" : "id",
    "created_at" : "created_at",
    "lang" : "lang",
    "favorite_count" : "favorite_count",
    "favorited" : "favorited",
    "retweeted" : "retweeted",
    "retweet_count" : "retweet_count",
    "text" : "text",
    "posted_date" : "posted_date",
    "posted_month" : "posted_month",
    "user_id" : "user_id",
    "user_name" : "user_name",
    "user_screen_name" : "user_screen_name",
    "user_followers_count" : "user_followers_count",
    "user_friends_count" : "user_friends_count",
    "user_tweets_count" : "user_statuses_count",
    "user_description" : "user_description",
    "user_time_zone" : "user_time_zone",
    "place_country" : "place_country",
    "place_country_code" : "place_country_code",
    "place_place_type" : "place_place_type",
    "place_name" : "place_name",
    "place_full_name" : "place_full_name",
    "tweet_longitude" : "tweet_longitude",
    "tweet_latitude" : "tweet_latitude",
    "retweeted_user_id" : "retweeted_status_user_id",
    "retweeted_user_name" : "retweeted_status_user_name",
    "retweeted_user_screen_name" : "retweeted_status_user_screen_name",
    "retweeted_user_location" : "retweeted_status_user_location",
    "retweeted_user_created_at" : "retweeted_status_user_created_at",
    "retweeted_user_favourites_count" : "retweeted_status_user_favourites_count",
    "retweeted_user_followers_count" : "retweeted_status_user_followers_count",
    "retweeted_user_friends_count" : "retweeted_status_user_friends_count",
    "retweeted_user_tweet_count" : "retweeted_status_user_tweet_count",
    "retweeted_user_description" : "retweeted_status_user_description",
    "retweeted_user_time_zone" : "retweeted_status_user_time_zone",
    "retweeted_place_country" : "retweeted_status_place_country",
    "retweeted_place_name" : "retweeted_status_place_name",
    "retweeted_place_full_name" : "retweeted_status_place_full_name",
    "retweeted_place_country_code" : "retweeted_status_place_country_code",
    "retweeted_place_place_type" : "retweeted_status_place_place_type",
    "retweeted_created_at" : "retweeted_status_created_at",
    "retweeted_tweet_longitude" : "retweeted_tweet_longitude",
    "retweeted_tweet_latitude" : "retweeted_tweet_latitude",
    "retweeted_text" : "retweeted_status_text"
}
