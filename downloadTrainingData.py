import gensim.downloader as api
dataset = api.load("text8")

import itertools
corpus = list(itertools.chain.from_iterable(dataset))

import pickle
with open('traningData.pickle', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
