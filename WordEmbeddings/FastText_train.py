"""
Train Fast Text model

Author: Adrian Ahne (AA)

Date: 22-10-2018
"""

import argparse
import pandas as pd
from gensim.models import FastText
import multiprocessing
import os.path as op
import sys
import os


# add path to utils module to python path
basename = op.split(op.dirname(op.realpath(__file__)))[0]
path_utils = op.join(basename , "utils")
sys.path.insert(0, path_utils)

from sys_utils import load_library

load_library(op.join(basename, 'preprocess'))
from preprocess import Preprocess

load_library(op.join(basename, 'readWrite'))
from readWrite import savePandasDFtoFile, readFile



def preprocessTweetsAndSave(args, prep):

    with open(args.tempFile, "w") as f:
        for i, line in readFile(args.filename, columns=args.filenameColumns, sep=args.filenameDelimiter).iterrows():
            # some tweets in the file reduced-tweets.parquet were None
            if line[args.dataColumnName] is not None and line["lang"] == args.lang:
                tweet = line[args.dataColumnName]
                tweet = prep.replace_hashtags_URL_USER(tweet, mode_URL="replace", mode_Mentions="replace") 
                f.write((" ".join(prep.tokenize(tweet)))+"\n")

    f.close()





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train FastText model to create word vectors. \
                                                  Two options: 1) local mode: specify MongoDB host, \
                                                  MongoDB port, MongoDB database, MongoDB collection; \
                                                  2) cluster mode: specify path to data",
                                     epilog='Example usage in local mode : \
                                             python FastText_train.py -m "local" \
                                                                        -lh "localhost" \
                                                                        -lp "27017" \
                                                                        -ldb "tweets" \
                                                                        -lc "en_noRetweets" \
                                                                        --iter 50 \
                                                                        --alpha 0.05 \
                                                                        -fn "hdfs://bgdta1-demy:8020/data/twitter/track-analyse/reduced-tweets.parquet" ')
    parser.add_argument("-m", "--mode", help="Mode of execution (default=local)", choices=["local", "cluster"], required=True, default="local")
    parser.add_argument("-lh", "--localMongoHost", help="Host to connect to MongoDB (default=localhost)", default="localhost")
    parser.add_argument("-lp", "--localMongoPort", help="Port to connect to MongoDB (default=27017)", default="27017")
    parser.add_argument("-ldb", "--localMongoDatabase", help="MongoDB database to connect to")
    parser.add_argument("-lc", "--localMongoCollection", help="MongoDB collection (table) in which data is stored")
    parser.add_argument("-fn", "--filename", help="Path to the data file")
    parser.add_argument("-lfd", "--filenameDelimiter", help="Delimiter used in file (default=',')", default=",")
    parser.add_argument("-lfc", "--filenameColumns", help="String with column names")
    parser.add_argument("-dcn", "--dataColumnName", help="If data stored in tabular form, gives the column of the desired text data (default='tweetText')", default="tweetText")
    parser.add_argument("-l", "--lang", help="Language for which word vectors shall be trained", default="en")
    parser.add_argument("-tf", "--tempFile", help="Temporary file to write preprocessed tweets in and to read directly to FastText training", default="/space/tmp/tmp.cor")
    parser.add_argument("--vecDim", help="Vector dimension of the word embedding (default=300)", default=300, type=int)
    parser.add_argument("--window", help="Maximum distance between the current and predicted word within a sentence (default=5)", default=5, type=int)
    parser.add_argument("--minCount", help="The model ignores all words with total frequency lower than this (default=1)", default=1, type=int)
    parser.add_argument("--localWorkers", help="Number of worker threads to train the model (default=all possible cores of machine)", default=multiprocessing.cpu_count())
    parser.add_argument("--sg", help="Training algo: Skip-gram if sg=1, otherwise CBOW (default=1)", choices=[0,1], default=1)
    parser.add_argument("--hs", help="Hierarchical softmax used for training if hs=1, otherwise negative sampling (default=0)", choices=[0,1], default=0)
    parser.add_argument("--alpha", help="Initial learning rate (default=0.025)", default=0.025)
    parser.add_argument("--seed", help="Seed for random number generator (default=1)", default=1)
    parser.add_argument("--iter", help="Number of iterations (epochs) over the corpus (default=20)", default=20, type=int)
    parser.add_argument("--word_ngrams", help="Uses enriched word vectors with subword information if 1, otherwise this is like Word2Vec if 0 (default=1)", default=1, choices=[0,1])
    parser.add_argument("--min_n", help="Minimum length of char n-grams to be used for training (default=3)", default=3, type=int)
    parser.add_argument("--max_n", help="Maximum length of char n-grams to be used for training (default=6)", default=6, type=int)
    parser.add_argument("-s", "--savePath", help="Path where to save model to", required=True)

    args = parser.parse_args()


    # Preprocessing class
    prep = Preprocess(lang="english")


    # get tweets
    if args.mode == "local":


        # check from which source to read the data
        if args.filename is not None:

            print("Write tokenized tweets to temporary file: {} ...".format(args.tempFile))
            preprocessTweetsAndSave(args, prep)
            print("Writing temporary file finished")

        # Check if necessary arguments are given
        elif args.localMongoDatabase is None and args.localMongoCollection is None:
            sys.stderr.write("ERROR: A MongoDB database and collection need to be provided to extract the data")
            sys.exit(1)

        else:
            print("Local mode: Connect to MongoDB collection..")
            from mongoDB_utils import connect_to_database

            client = connect_to_database()
            db = client[args.localMongoDatabase]
            collection = db[args.localMongoCollection]

            print("Tokenize tweets..")
            tweets = []
            for tweet in collection.find():
                tweets.append(prep.tokenize(tweet))

    elif args.mode == "cluster":

        # Check if necessary arguments are given
        if args.filename is None:
            sys.stderr.write("ERROR: A path to file containing the data needs to be provided")
            sys.exit(1)

        print("Cluster mode: Read parquet files..")
        raw_tweets = readFile(args.filename, columns=args.filenameColumns, sep=args.filenameDelimiter)

        print("Tokenize tweets..")
        tweets = []
        for tweet in raw_tweets[args.dataColumnName].values:
            tweets.append(prep.tokenize(tweet))

    else:
        print("ERROR: Provided mode : {} is not supported. Possible options (local, cluster) ".format(args.mode))
        sys.exit()



    print("Train FastText...")
    model_ft = FastText(corpus_file=args.tempFile, size=args.vecDim, window=args.window, min_count=args.minCount,
                        workers=args.localWorkers ,sg=args.sg, hs=args.hs, iter=args.iter,
                        word_ngrams=args.word_ngrams, min_n=args.min_n, max_n=args.max_n,
                        seed=args.seed, alpha=args.alpha)

    print("Save model to disk...")
    #file_name = "Trained_FastText_{}.model".format(datetime.datetime.now().strftime(DATE_FORMAT))
    if not os.path.isdir(os.path.dirname(args.savePath)):
        os.makedirs(os.path.dirname(args.savePath))
    model_ft.save(args.savePath)
    
    print("Delete temporary file {}".format(args.tempFile))
    os.remove(args.tempFile)
