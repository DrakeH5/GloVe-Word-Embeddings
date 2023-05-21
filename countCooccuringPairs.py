import pickle 

def countCooccuringPairs(corpus, i2Token):
    print("Counting Cooccuring Pairs...")
    cooccuranceMatrix = {}
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
    with open('./cooccurrenceEntries/cooccurrence.pkl', 'wb') as f:
        pickle.dump(cooccuranceMatrix, f)