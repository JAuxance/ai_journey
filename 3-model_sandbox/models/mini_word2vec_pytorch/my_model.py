import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeding = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            return self.embedding(x)

        