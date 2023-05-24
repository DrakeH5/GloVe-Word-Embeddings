from pathlib import Path
import os
import argparse
import pickle

import torch
import yaml
from gensim.models.keyedvectors import KeyedVectors
from train import GloVe
import h5py


def main():
    with open("./vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    model = GloVe(
        vocab_size=len(vocab),
        embedding_size=100,
        x_max=100,
        alpha=0.75
    )
    model.load_state_dict(torch.load("./output2.pt"))
    
    keyed_vectors = KeyedVectors(vector_size=100)
    keyed_vectors.add_vectors(
        keys=[vocab[i] for i in range(len(vocab))],
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