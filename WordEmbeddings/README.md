# Word embeddings

This directory provides two possibilities to compute word embeddings using the
gensim package:
- Word2Vec
- FastText (uses subword information by taking n-grams into consideration)

This can be done in two modes:
- local mode (-m "local"):
  - Load data
    - from .parquet or .csv to pandas DF (-fn "pathToFile.parquet")
      - Flags:
        - Load only specific columns: --localFileColumns -lfc (-lfc "text, user_name")
        - Specifies column name of text data, default "tweetText": --dataColumnName -dcn (-dcn "text")

    - from MongoDB collection, provide flags:
      - localMongoHost -lh
      - localMongoPort -lp
      - localMongoDatabase -ldb
      - localMongoCollection -lc

- cluster mode (-m "cluster"):
  - load parquet file from hdfs, provide flag (-fn "hdfs://machine:8888/pathToFile.parquet"):
    - dataColumnName -dcn : column in the dataframe containing the text data (default="tweetText")


After loading the data:
- Preprocesses tweets and stores tweet line by line in a temporary file
  (from which the Word2Vec / FastText algorithm can read the data)
  - Path to temporary file (-tf "path")

Provide a path to which the trained model will be stored via the -s flag

## Word2Vec
Possible flags to specify training:
- vecDim: Vector dimension of the word embedding (default=200)
- window: Maximum distance between the current and predicted word within a sentence (default=5)
- minCount: The model ignores all words with total frequency lower than this (default=1)
- localWorkers: Number of worker threads to train the model (default=all possible cores of machine)
- sg {0,1}: Training algo: Skip-gram if sg=1, otherwise CBOW (default=1)
- hs {0,1}: Hierarchical softmax used for training if hs=1,otherwise negative sampling (default=0)
- alpha: Initial learning rate (default=0.025)
- seed: Seed for random number generator (default=1)
- iter: Number of iterations (epochs) over the corpus (default=20)

## FastText
The same flags as for Word2Vec are provided.
Additionally you can specify:
- word_ngrams {0,1}: Uses enriched word vectors with subword information if 1,
                     otherwise this is like Word2Vec if 0 (default=1)
- min_n: Minimum length of char n-grams to be used for training (default=3)
- max_n: Maximum length of char n-grams to be used for training (default=6)


## Example call
python FastText_train.py -m "local" -fn "hdfs://machinename:8888/tweets.parquet" -lfc "text" -dcn "text" -s "/space/Work/spark/FastText_model/ft_wordembeddings_07112018.parquet" --minCount 1 --iter 50
