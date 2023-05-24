import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from openCooccurence import openCooccurence


class GloVe(nn.Module): 
#https://github.com/pengyan510/nlp-paper-implementation/blob/master/glove/src/glove.py
     def __init__(self, vocab_size, embedding_size, x_max, alpha):
        super().__init__()
        self.weight = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            sparse=True
        )
        self.weight_tilde = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            sparse=True
        )
        self.bias = nn.Parameter(
            torch.randn(
                vocab_size,
                dtype=torch.float,
            )
        )
        self.bias_tilde = nn.Parameter(
            torch.randn(
                vocab_size,
                dtype=torch.float,
            )
        )
        self.weighting = lambda x: (x / x_max).float_power(alpha).clamp(0, 1)
    
     def forward(self, i, j, x):
        loss = torch.mul(self.weight(i), self.weight_tilde(j)).sum(dim=1)
        loss = (loss + self.bias[i] + self.bias_tilde[j] - x.log()).square()
        loss = torch.mul(self.weighting(x), loss).mean()
        return loss


def train(vocab, vocabCount, t2I):
    print("Training...")
    batches = openCooccurence()
    model = GloVe(
        vocab_size=len(vocab),
        embedding_size=100,
        x_max=100,
        alpha=0.75
    )
    #model.to(cpu)
    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=0.05
    )  
    model.train()
    totLosses  = []
    for epoch in tqdm(range(10)):
        eLoss = 0
        for batch in batches:
            loss = model(
                batch[0][:, 0],
                batch[0][:, 1],
                batch[1]
            ) 
            eLoss += loss.detach().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        totLosses.append(eLoss)
        print(f"Epoch {epoch}: loss = {eLoss}")
        torch.save(model.state_dict(), "./output2.pt")