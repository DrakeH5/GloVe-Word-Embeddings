import torch

pt_file = torch.load("./output2.pt")
pt_fileOG = torch.load("./output.pt")

print(pt_file)
print(pt_fileOG)