def countCooccuringPairs(corpus, i2Token):
    print("Counting Cooccuring Pairs...")
    cooccuranceMatrix = {}
    for k in range(len(i2Token)):
        cooccuranceMatrix[i2Token[k]] = {}
    for i in range(len(corpus)):
        for k in range(len(i2Token)):
            cooccuranceMatrix[corpus[i]][i2Token[k]] = 0
        for j in range(-10, 11):
            if i + j >= 0 and i + j < len(corpus) and j != 0:
                cooccuranceMatrix[corpus[i]][corpus[i+j]]+=1
        #print(i)
    return cooccuranceMatrix