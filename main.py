import gensim.downloader as api
dataset = api.load("text8")

import itertools
corpus = list(itertools.chain.from_iterable(dataset))