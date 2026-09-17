import torch
import torch.nn as nn


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target_idx, context_idx):
        target_vec = self.input_embedding(target_idx)
        context_vec = self.output_embedding(context_idx)

        score = torch.sum(target_vec * context_vec, dim=1)

        return score
