"""

Preprocessing functions

Author: Adrian Ahne
Creation date: 24/04/2018

"""
import string
import unicodedata

import re
import sys
import contractions # expanding contractions
import inflect # natural language related tasks of generating plurals, singular nouns, etc.
import nltk
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

from emotion_codes import UNICODE_EMOJI
from emotion_codes import EMOTICONS
from emotion_codes import EMOJI_TO_CATEGORY
from defines import *
from contractions_def import *


# assume input matrix contains term frequencies
def tfidf_transform(mat):

    # convert matrix of counts to matrix of normalized frequencies
    normalized_mat = mat / np.transpose(mat.sum(axis=1)[np.newaxis])

    # compute IDF scores for each word given the corpus
    docs_using_terms = np.count_nonzero(mat,axis=0)
    idf_scores = np.log(mat.shape[1]/docs_using_terms)

    # compuite tfidf scores
    tfidf_mat = normalized_mat * idf_scores
    return tfidf_mat

class Preprocess:

    def __init__(self, lang="english"):
        self.TweetTokenizer = TweetTokenizer()
        # Constant words like URL, USER, EMOT_SMILE, etc. that we want to keep in uppercase
        self.Constant_words = [value for attr, value in Constants.__dict__.items()
                               if not callable(getattr(Constants, attr)) and
                               not attr.startswith("__")]+Emotions.EMOTION_CATEGORIES

        self.WN_Lemmatizer_EN = WordNetLemmatizer()
        self.lang = lang

    def get_text(self, raw_tweet):
        """ get text of tweet object in json format """
        return raw_tweet["text"]

    def replace_contractions(self, tweet):
        """ Replace contractions in string of text
            Examples:
              "aren't": "are not",
              "can't": "cannot",
              "'cause": "because",
              "hasn't": "has not",
              "he'll": "he will",

              FIXME: it occurs that
              - "people were in a hurry" is transformed to "we are in a hurry"  !!!
              - "are the main cause of obeisty" -> "are the main because of obesity"
              - "in the U.S are" -> "in the you.S. are"
        """

        if self.lang == "english":
            return  contractions_fix(tweet)
        else:
            return(tweet)

    def replace_hashtags_URL_USER(self, tweet, mode_URL="keep",
                                  mode_Mentions="keep", mode_Hashtag="keep" ):
        """
            Function handling the preprocessing of the hashtags, User mentions
            and URL patterns

            Parameters
            -------------------------------------------------------

            mode_URL : ("replace", "delete")
                       if "replace" : all url's in tweet are replaced with the value of Constants.URL
                       if "delete" : all url's are deleted
                       if "keep" : keep url

            mode_Mentions : ("replace", "delete", "screen_name")
                       if "replace" : all user mentions in tweet are replaced
                                      with the value of Constants.USER
                       if "delete" : all user mentions are deleted
                       if "screen_name" : delete '@' of all user mentions
                       if "keep" : keep user mention

            mode_Hashtag : ("replace", "delete")
                       if "replace" : all '#' from the hashtags are deleted
                       if "delete" : all hashtags are deleted
                       if 'keep' : keep hashtag

            Return
            -------------------------------------------------------------
            List of preprocessed tweet tokens

            https://github.com/yogeshg/Twitter-Sentiment

            Ex.:
            s = "@Obama loves #stackoverflow because #people are very #helpful!, \
                 check https://t.co/z2zdz1uYsd"
            print(replace_hashtags_URL_USER(s))
            >> "USER loves stackoverflow because people are very helpful!, check URL"


        """
        if mode_URL == "replace":
            tweet = Patterns.URL_PATTERN.sub(Constants.URL, tweet)
        elif mode_URL == "delete":
            tweet = Patterns.URL_PATTERN.sub("", tweet)
        elif mode_URL == "keep" :
            tweet = tweet
        else:
            print("ERROR: mode_URL {} not defined!".format(mode_URL))
            exit()

        if mode_Mentions == "replace":
            tweet = Patterns.MENTION_PATTERN.sub(Constants.USER, tweet)
        elif mode_Mentions == "delete":
            tweet = Patterns.MENTION_PATTERN.sub("", tweet)
        elif mode_Mentions == "screen_name":
            mentions = Patterns.MENTION_PATTERN.findall(tweet)
            for mention in mentions:
                tweet = tweet.replace("@"+mention, mention)
        elif mode_Mentions == "keep" :
            tweet = tweet
        else:
            print("ERROR: mode_Mentions {} not defined!".format(mode_Mentions))
            exit()

        if mode_Hashtag == "replace":
            hashtags = Patterns.HASHTAG_PATTERN.findall(tweet)
            for hashtag in hashtags:
                tweet = tweet.replace("#"+hashtag, hashtag)
        elif mode_Hashtag == "delete":
            hashtags = Patterns.HASHTAG_PATTERN.findall(tweet)
            for hashtag in hashtags:
                tweet = tweet.replace("#"+hashtag, "")
        elif mode_Hashtag == "keep" :
            tweet = tweet
        else:
            print("ERROR: mode_Hashtag {} not defined!".format(mode_Hashtag))
            exit()

        return tweet


    def replace_special_words(self, tweet):
        """
            Replace special words

            For ex.: all the type 1 related words like "#type1", "Type 1", "t1d", etc.
                     are transformed to "type1"

        """

        # replace type 1 words
        tweet = WordLists.TYPE1_WORDS.sub(Constants.TYPE1, tweet)

        # replace type 2 words
        tweet = WordLists.TYPE2_WORDS.sub(Constants.TYPE2, tweet)

        return tweet


    def remove_repeating_characters(self, tweet):
        """
            If a word contains repeating characters, reduce it to only two repeating characters
            Ex. "coooooool" => "cool"
        """
        return re.sub(r'(.)\1+', r'\1\1', tweet)


    def remove_repeating_words(self, tweet):
        """
            Remove repeating words and only keep one
            Ex.: "I so need need need to sing" => "I so need to sing"
        """
        return re.sub(r'\b(\w+)( \1\b)+', r'\1', tweet)

    def tokenize(self, tweet):
        """
            Tokenizes tweet in its single components (words, emojis, emoticons)

            Ex.:
            s = "I love:D python 😄 :-)"
            print(tokenize(s))
            >> ['I', 'love', ':D', 'python', '😄', ':-)']
        """
        return list(self.TweetTokenizer.tokenize(tweet))

    def remove_punctuation(self, tweet):
        """
            Remove punctuations from list of tokenized words

            Punctuations: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~...…

            Example:
            >>> text = ['hallo', '!', 'you', '...', 'going', 'reducing', ',']
            >>> remove_punctuation(text)
            >>> ['hallo', 'you', 'going', 'reducing']

            TODO: check if !,? may contain useful information
        """
        def check_punctuation(word):
            return word not in string.punctuation and word not in ['...', '…', '..', "\n", "\t", " ", ""] 

        return [word for word in tweet if check_punctuation(word)]


    def preprocess_emojis(self, tweet, limit_nEmojis=False):
        '''
            Replace emojis with their emotion category
            Example:
                >>> text = "I love eating 😄"
                >>> preprocess_emoji(text)
                >>> "I love eating EMOT_LAUGH"


            Parameters:
            ------------------------------------------------------------
            tweet:          tokenized tweet
            limit_nEmojis:  give maximum number of emojis of the same emotion category
                            that should occur in a tweet. Delete the other ones
                            Default: False, all emojis are considered

            Return
            ---------------------------------------------------------------
            tokenized tweet with replaced emojis by their emotion category
        '''

        if self.lang == "english":

            # counts occurrences of emojis in their emotion category
            emot_counter = {}
            for emotion in Emotions.EMOTION_CATEGORIES:
                emot_counter[emotion] = 0

            cleaned_tweet = []
            for ind, char in enumerate(tweet):
                if char in UNICODE_EMOJI:

                    emot_cat = EMOJI_TO_CATEGORY[UNICODE_EMOJI[char]]
                    if emot_cat != "":

                        if limit_nEmojis is not False:

                            # it is possible that one emoji is categorized into two
                            # different categories, for instance: 'EMOT_SURPRISE EMOT_FEAR'
                            emot_cat = emot_cat.split(" ")
                            for emo in emot_cat:
                                emot_counter[emo] += 1 # counts for the emotion in this tweet
                                if emot_counter[emo] <= limit_nEmojis:
                                    cleaned_tweet.append(emo)
                        else:
                            cleaned_tweet.append(emot_cat)

                    else:
                        print("INFO: No category set for emoji {} -> delete emoji {}".format(char, UNICODE_EMOJI[char]))
                else:
                    cleaned_tweet.append(char)

            return cleaned_tweet

        # other language
        else:
            return(tweet)

    def preprocess_emoticons(self, tweet):
        '''
            Replace emoticons in tweets with their emotion category by searching for
            emoticons with the pattern key word

            Example:
                >>> text = "I like nutella :)"
                >>> preprocess_emoticons(text)
                >>> "I like nutella EMOT_SMILE"
        '''


        if self.lang == "english":
            cleaned_tweet = []
            for word in tweet:
                match_emoticon = Patterns.EMOTICONS_PATTERN.findall(word)
                if not match_emoticon : # if no emoticon found
                    cleaned_tweet.append(word)
                else:
                    if match_emoticon[0] is not ':':
                        if match_emoticon[0] is not word:
                            cleaned_tweet.append(word)
                        else:
                            try:
                                cleaned_tweet.append(EMOTICONS[word])
                            except:
                                print("INFO: Could not replace emoticon: {} of the word: {}".format(match_emoticon[0], word), sys.exc_info())
            return cleaned_tweet

        # other languages
        else:
            return tweet

    def to_lowercase(self, tweet):
        """
            Convert all characters to lowercase from list of tokenized words

            Example:
                >>> text = ["I", "like", "Nutella", "URL"]
                >>> to_lowercase(text)
                >>> ["i", "like", "nutella", "URL"]

            Remark: Do it after emotion treatment, otherwise smiley :D -> :d
        """

        return [word.lower() if word not in self.Constant_words else word for word in tweet]


    def remove_non_ascii(self, tweet):
        """Remove non-ASCII characters from list of tokenized words"""

        return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') \
                for word in tweet if unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') is not ""]


    def replace_numbers(self, tweet, mode="replace"):
        """
            Replace all interger occurrences in list of tokenized words with textual representation

            Example:
            >>> text = ['June', '2017', 'McDougall', '10', 'Day']
            >>> replace_numbers(text)
            >>> ['June', 'two thousand and seventeen', 'McDougall', 'ten', 'Day']

            REMARK: Maybe better to delete numbers of leave them as string: '2017'
        """

        if mode == "replace":
            p = inflect.engine()
            return [p.number_to_words(word) if word.isdigit() else word for word in tweet]

        elif mode == "delete":
            return [word for word in tweet if not word.isdigit() ]


    def remove_stopwords(self, tweet, include_personal_words=False, include_negations=False, list_stopwords_manual=False):
        """
            Remove stop words from list of tokenized words

            Parameter:
                tweet : tokenised list of strings
                include_personal_words : [True, False]
                                        if True, personal stopwords like
                                        "I", "me", "my" are not considered as
                                        stopwords
                include_negations: [True, False]
                                    if True, negation words like "no", "not" ,"nothing"
                                    are included and not considered as stopwords
                ignore_whitelist : whitelist containing words

                list_stopwords_manual : list with stopwords that overwrites the default stop lists if given


            Example:
            >>> text = ['five', 'reasons', 'to', 'eat', 'like', 'a', 'hunter']
            >>> remove_stopwords(text)
            >>> ['five', 'reasons', 'eat', 'like', 'hunter']
        """
        new_tweet = []


        # manual list of stopwords provided
        if list_stopwords_manual != False:
            return [word for word in tweet if word not in list_stopwords_manual]

        else:
            # english language
            if self.lang == "english":
                for word in tweet:
                    if include_personal_words:
                        if include_negations:
                            if word not in Grammar.STOPWORDS_NO_PERSONAL_EN or word in Grammar.WHITELIST_EN: # TODO maybe add manually more stopwords
                                new_tweet.append(word)
                        else:
                            if word not in Grammar.STOPWORDS_NO_PERSONAL_EN: # TODO maybe add manually more stopwords
                                new_tweet.append(word)
                    else:
                        if include_negations:
                            if word not in Grammar.STOPWORDS_EN or word in Grammar.WHITELIST_EN: # TODO maybe add manually more stopwords
                                new_tweet.append(word)
                        else:
                            if word not in Grammar.STOPWORDS_EN: # TODO maybe add manually more stopwords
                                new_tweet.append(word)
                return new_tweet

            # french language
            elif self.lang == "french":
                for word in tweet:
                    if word not in Grammar.STOPWORDS_FR:
                        new_tweet.append(word)
                return new_tweet

            # other languages
            else:
                return tweet



    def lemmatize_verbs(self, tweet):
        """ Lemmatize verbs in list of tokenized words

            Example:
            >>> text = ['americans', 'stopped', 'drinking']
            >>> lemmatize_verbs(text)
            >>> ['americans', 'stop', 'drink']
        """

        # Lemmatization
        def lookup_pos(pos):
            pos_first_char = pos[0].lower()
            if pos_first_char in 'nv':
                return pos_first_char
            else:
                return 'n'

        if self.lang == "english":

            # Part-of-speech tagging
            pos_tags = nltk.pos_tag(tweet)

            return [self.WN_Lemmatizer_EN.lemmatize(word,lookup_pos(pos)) for (word,pos) in pos_tags]

        else:
            return tweet

    def stem_words(self, tweet, stemmer=False):
        """ Stem words in list of tokenized words

            Parameter:
                - tweet :   tokenized list of words of the tweet

                - stemmer : algorithm to use for stemming, options:
                            - Grammar.STEMMER_SNOWBALL_EN
                            - Grammar.PORTER
                            - Grammar.STEMMER_LANCASTER
                            - Grammar.STEMMER_SNOWBALL_FR

            Example:
            >>> text = ['predictive', 'tool', 'for', 'children', 'with', 'diabetes']
            >>> stem_words(text)
            >>> ['predict', 'tool', 'for', 'children', 'diabet']

            Remark: Three major stemming algorithms
                - Porter: most commonly used, oldest, most computationally expensive
                - Snowball / Porter2: better than Porter, a bit faster than Porter
                - Lancaster: aggressive algorithm, sometimes to a fault; fastest algo
                            often not intuitiive words; reduces words space hugely

                - Snowball French
        """


        for ind, word in enumerate(tweet):
            if word not in self.Constant_words: # do not change words like USER, URL, EMOT_SMILE,...
                if stemmer != False:
                    tweet[ind] = stemmer.stem(word)
                elif self.lang == "english":
                    tweet[ind] = Grammar.STEMMER_SNOWBALL_EN.stem(word)
                elif self.lang == "french":
                    tweet[ind] = Grammar.STEMMER_SNOWBALL_FR.stem(word)

        return tweet
