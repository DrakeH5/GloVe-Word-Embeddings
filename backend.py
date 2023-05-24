from pathlib import Path
import os
import argparse
import pickle

import torch
import yaml
from gensim.models.keyedvectors import KeyedVectors
from train import GloVe
import h5py


def load_config():
    config_filepath = Path(__file__).absolute().parents[0] / "config.yaml"
    with config_filepath.open() as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(config, key, value)
    return config


def main():
    config = load_config()
    with open(os.path.join(config.cooccurrence_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    model = GloVe(
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        x_max=config.x_max,
        alpha=config.alpha
    )
    model.load_state_dict(torch.load(config.output_filepath))
    
    keyed_vectors = KeyedVectors(vector_size=config.embedding_size)
    keyed_vectors.add_vectors(
        keys=[vocab.get_token(index) for index in range(config.vocab_size)],
        weights=(model.weight.weight.detach()
            + model.weight_tilde.weight.detach()).numpy()
    )
    
    while True:
        mode = input("Enter a 1 for comparing Similarity of Words \nEnter a 2 for finding most similar words \n")
        if mode == "1":            
            print(keyed_vectors.similarity(input("Words 1?"), input("Word 2?")))
        if mode == "2": 
            word = input("Word? \n")
            print([word for word, _ in keyed_vectors.similar_by_word(word)])



if __name__ == "__main__":
    main()