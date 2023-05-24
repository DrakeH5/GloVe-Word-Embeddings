import random
import pickle


def shortenVocab(tokens, tokens2Index, nmbOfOccurences, limit):
    for j, i in enumerate(nmbOfOccurences):
        if i < limit:
            del nmbOfOccurences[j]
            del tokens2Index[tokens[j]]
            del tokens[j]
    return tokens, tokens2Index, nmbOfOccurences


def creatingVocab(corpus):
    print("Creating Vocab...")
    tokens = {}
    convertTokenToIndex = {}
    nmbOfOccurences = []
    for token in corpus:
        if token not in convertTokenToIndex:
            tokens[len(tokens)] = token
            convertTokenToIndex[token] = len(convertTokenToIndex)
            nmbOfOccurences.append(0)
        nmbOfOccurences[convertTokenToIndex[token]]+=1
    #shortenVocab(tokens, convertTokenToIndex, nmbOfOccurences, 10)
    tokens, nmbOfOccurences, convertTokenToIndex = shuffleVocab(tokens, nmbOfOccurences, convertTokenToIndex)
    with open("./vocab.pkl", "wb") as file:
        pickle.dump(tokens, file)
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