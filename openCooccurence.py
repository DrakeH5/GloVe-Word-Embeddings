import h5py
import torch
import random

def openCooccurence():
    with h5py.File("./cooccurrenceEntries\cooccurrence.hdf5", "r") as f:
        dataset = f["cooccurence"][()]  # returns as a numpy array
        random.shuffle(dataset)
        outputData = [
            torch.from_numpy(dataset[:,:2]).long(),
            torch.from_numpy(dataset[:,2]).float()
            ]
        yield outputData

#openCooccurence()