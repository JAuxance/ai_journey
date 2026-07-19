import torch
import torch.nn as nn

word = ["men", "women", "car", "train", "stylo", "crayon", "red", "blue"]
word_idx = {"men": 0, "women": 1, "car": 2, "train": 3, "stylo": 4, "crayon": 5, "red": 6, "blue": 7}
pair_tuple = ((0, 1), (2, 3), (4, 5), (6, 7))

tensor = torch.tensor(pair_tuple)
embedding_layer = nn.Embedding(8, 2)
embedding_layer(tensor)