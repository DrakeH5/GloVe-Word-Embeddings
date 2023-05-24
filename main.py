import pickle

#from torchtext.data import get_tokenizer
#tokenizer = get_tokenizer("basic_english")

from vocabulary import creatingVocab, shuffleVocab
from countCooccuringPairs import countCooccuringPairs
from train import train

def loadTrainingData():
    print("Loading training data...")
    with open("./traningData.pickle", "rb") as f:
        return pickle.load(f)


def main():
    print("Program Opened")
    corpus = loadTrainingData()
    #tokenizer(corpus)
    vocabTokens, vocabCount, token2Index = creatingVocab(corpus)
    #print(vocabTokens[0], vocabCount[0], token2Index[vocabTokens[0]])
    vocabTokens, vocabCount, token2Index = shuffleVocab(vocabTokens, vocabCount, token2Index)
    #print(vocabTokens[0], vocabCount[0], token2Index[vocabTokens[0]])
    countCooccuringPairs(corpus, vocabTokens, token2Index)
    #print(cooccuranceMatrix)
    train(vocabTokens, vocabCount, token2Index)


main()