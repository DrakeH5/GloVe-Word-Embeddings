import random
import pickle


def shortenVocab(tokens, tokens2Index, nmbOfOccurences, limit):
    #https://github.com/pengyan510/nlp-paper-implementation/blob/master/glove/src/vocabulary.py
    inOrder = sorted(list(tokens2Index.keys()), key = lambda token: nmbOfOccurences[tokens2Index[token]], reverse=True)
    tokens2Index={token: index for index, token in enumerate(inOrder[:limit])}
    tokens={index: token for index, token in enumerate(inOrder[:limit])}
    nmbOfOccurences = [nmbOfOccurences[tokens2Index[token]] for token in inOrder[:limit]]
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
    tokens, convertTokenToIndex, nmbOfOccurences = shortenVocab(tokens, convertTokenToIndex, nmbOfOccurences, 100000)  
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