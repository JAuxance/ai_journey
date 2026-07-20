import torch
import torch.nn as nn

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target_idx, context_idx):
        
        target_vec = self.embedding(target_idx)
        context_vec = self.embedding(context_idx)
        similarity = torch.sum(target_vec * context_vec) 
        
        return similarity