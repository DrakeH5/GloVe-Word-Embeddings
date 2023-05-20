import pickle

with open('./cooccurrenceEntries/cooccurrence.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

print(loaded_dict["anarchy"]["chaos"])