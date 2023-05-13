import gensim.downloader as api
dataset = api.load("text8")

import itertools
corpus = list(itertools.chain.from_iterable(dataset))


vectorizer = Vectorizer.from_corpus(
    corpus=corpus,
    vocab_size=config.vocab_size
)
cooccurrence = CooccurrenceEntries.setup(
    corpus=corpus,
    vectorizer=vectorizer
)
cooccurrence.build(
    window_size=config.window_size,
    num_partitions=config.num_partitions,
    chunk_size=config.chunk_size,
    output_directory=config.cooccurrence_dir
) 