import torch
import torch.nn as nn
from model import Word2VecModel
import torch.optim as optim


word = ["men", "women", "car", "train", "stylo", "crayon", "red", "blue"]
word_idx = {"men": 0, "women": 1, "car": 2, "train": 3, "stylo": 4, "crayon": 5, "red": 6, "blue": 7}
pair_tuple = [
    (2, 3, 1),
    (6, 5, 0),
    (4, 2, 0),
    (0, 1, 1),
    (3, 5, 0),
    (5, 7, 0),
]

model = Word2VecModel(vocab_size=8, embedding_dim=2)
optimizer = optim.SGD(model.parameters(), lr=0.01)


loss_fn = nn.BCEWithLogitsLoss()
for epoch in range(100):
    total_loss = 0
    for target_idx, context_idx, label in pair_tuple:
        target = torch.tensor([target_idx])
        context = torch.tensor([context_idx])
        label = torch.tensor([label], dtype=torch.float)

        score = model(target, context)
        loss = loss_fn(score, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} - Loss: {total_loss:.4f}")