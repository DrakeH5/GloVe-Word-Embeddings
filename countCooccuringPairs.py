import pickle 
import numpy as np
from collections import Counter
from more_itertools import locate
from itertools import count
import time
import math
import h5py

def countCooccuringPairs(corpus, i2Token, t2I):
    print("Counting Cooccuring Pairs...")
    '''cooccuranceMatrix = {}
    defaultVec = {}
    for p in range(len(i2Token)):
        defaultVec[i2Token[p]] = 0
    for k in range(len(i2Token)):
        cooccuranceMatrix[i2Token[k]] = defaultVec
    for i in range(len(corpus)):
        for j in range(-10, 11):
            if i + j >= 0 and i + j < len(corpus) and j != 0:
                cooccuranceMatrix[corpus[i]][corpus[i+j]]+=1
        #print(i)
    dataset = np.zeros((len(i2Token)*len(i2Token), 3))
    index = 0
    for i in cooccuranceMatrix:
        for j in cooccuranceMatrix[i]:
            dataset[index] = (t2I[i], t2I[j], cooccuranceMatrix[i][j])
            print(index, i, j, cooccuranceMatrix[i][j])
            print(dataset[index])
            index+=1
    #with open('./cooccurrenceEntries/cooccurrence.pkl', 'wb') as f:
     #   pickle.dump(cooccuranceMatrix, f)'''
    output = h5py.File('cooccurrenceEntries\cooccurrence.hdf5', 'w')
    corpusGlossary = {}
    for c, i in enumerate(corpus):
        if i not in corpusGlossary:
            corpusGlossary[i] = [c]
        else:
            corpusGlossary[i].append(c)
    initTime = time.time()
    batchSize = 100
    for b in range(math.ceil(len(i2Token) / batchSize)):
        iteration = b * batchSize
        batch = Counter()
        for i in range(batchSize):
            if i+iteration < len(i2Token): 
                batch[i2Token[i+iteration]] = i+iteration
        cooccurance = Counter()
        for i in batch:
            #indexes = list(locate(corpus, lambda x: x == i))
            #indexes = [l for l, m in zip(count(), corpus) if m == i]
            #for j in indexes:
            #for j, m in zip(count(), corpus): 
            #   if m == i:
            for j in corpusGlossary[i]:
                for k in range(max(j-10, 0), min(j+10, len(i2Token))):
                    if k != 0 and corpus[k] in t2I:
                        if (t2I[i], t2I[corpus[k]]) not in cooccurance:
                            cooccurance[(t2I[i], t2I[corpus[k]])] = 0
                        cooccurance[(t2I[i], t2I[corpus[k]])]+=1
        dataset = np.zeros((len(cooccurance), 3))
        for index, ((k, j), cooccurance) in enumerate(cooccurance.items()):
            dataset[index] = (k, j, cooccurance)
            #print(index, k, j, cooccurance)
            #print(dataset[index])
        if b == 0:
            output.create_dataset('cooccurence', (len(dataset), 3), maxshape=(None, 3), chunks=(batchSize, 3), data=dataset)
        else:
            lenStore = output["cooccurence"].len()
            output["cooccurence"].resize(output["cooccurence"].len() + len(dataset), axis=0)
            output["cooccurence"][lenStore: output["cooccurence"].len()] = dataset
        #print(dataset)
    output.close()
    #print(time.time() - initTime, (time.time() - initTime)*(len(i2Token) / batchSize))
    #print(cooccurance)
