import h5py
filename = "./cooccurrenceEntries\cooccurrence.hdf5"
import numpy as np

hf = h5py.File(filename, 'r')
n1 = hf.get('cooccurrence')
n1 = np.array(n1)
print(n1)