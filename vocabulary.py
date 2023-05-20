import random

def creatingVocab(corpus):
    print("Creating Vocab...")
    tokens = {}
    convertTokenToIndex = {}
    nmbOfOccurences = []
    for token in corpus:
        if token not in tokens:
            tokens[len(tokens)] = token
            convertTokenToIndex[token] = len(convertTokenToIndex)
            nmbOfOccurences.append(0)
        nmbOfOccurences[convertTokenToIndex[token]]+=1
    return tokens, nmbOfOccurences, convertTokenToIndex 


def shuffleVocab(vocabTokens, vocabCount, token2Index):
    print("Shuffling...")
    NEWtokens = {}
    NEWconvertTokenToIndex = {}
    NEWnmbOfOccurences = []
    for i in range(len(vocabCount)):
        NEWnmbOfOccurences.append(0)
    new_index = [_ for _ in range(len(vocabTokens))]
    random.shuffle(new_index)
    for i in range(len(vocabTokens)):
        NEWtokens[i] = vocabTokens[new_index[i]]
        NEWconvertTokenToIndex[NEWtokens[i]] = i
        NEWnmbOfOccurences[i] = vocabCount[token2Index[vocabTokens[new_index[i]]]]
    return NEWtokens, NEWnmbOfOccurences, NEWconvertTokenToIndex 