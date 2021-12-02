"""
    Wrapper of FastText vectors of gensim into following text file format:

    NUMBER_VOCABULARY NUMBER_DIMENSION
    word1 vector1
    word2 vector2
    ...     ...

    18.06.2019
    AA

"""

import argparse
from gensim.models import FastText


def gensim_FastText_model_to_textFile(LoadModelPath, saveModelPath):

    # load FastText_model
    model = FastText.load(LoadModelPath)

    with open(saveModelPath, 'w') as f:
        print("Write to file {}".format(saveModelPath))
        f.write(str(len(model.wv.vocab.keys()))+" "+str(len(model.wv[list(model.wv.vocab.keys())[0]]))+"\n")
        for word in model.wv.vocab.keys():
            f.write(word+" ")
            for value in model.wv[word]: f.write(str(value)+" ")
            f.write("\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Wrap Vectors of gensim to plain text file.",
                                     epilog='Example: \
                                             python gensim_model_to_textFile.py -l "/Users/xyz/vectors.model" \
                                                                        -s "vectors.txt" ')
    parser.add_argument("-l", "--loadModelGensim", help="Path of gensim Model in which vectors are stored (absolute path)", required=True)
    parser.add_argument("-s", "--saveModel", help="Path where to save model (absolute path)", required=True)

    args = parser.parse_args()

    print("Load Model..")
    gensim_FastText_model_to_textFile(args.loadModelGensim, args.saveModel)
    print("Done writing!")
