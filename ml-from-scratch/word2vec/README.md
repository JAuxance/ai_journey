# Word2Vec

Understanding how words become vectors: first the theory (StatQuest + reading),
then a minimal skip-gram model with negative sampling written in PyTorch.

**Status: Tested.** The model trains on a toy vocabulary of 8 words and the loss
goes down. There is no evaluation yet (no real corpus, no similarity check on the
learned vectors).

---

## Files

| File | What it is |
|---|---|
| [`notes.md`](./notes.md) | Theory notes: embeddings, CBOW vs skip-gram, backpropagation, negative sampling |
| [`word2vec.ipynb`](./word2vec.ipynb) | Demo of a pre-trained embedding model (`sentence-transformers`): king/queen vs king/car similarity |
| [`train.ipynb`](./train.ipynb) | **First attempt**, kept as a record (see below) |
| [`model.py`](./model.py) | `Word2VecModel`: two embedding tables (target and context), score = dot product |
| [`train.py`](./train.py) | Training loop: positive and negative pairs, `BCEWithLogitsLoss`, SGD, 100 epochs |

## From the first attempt to the current version

`train.ipynb` used a **single** embedding table and the loss `-score`. With only
positive pairs, the model can always push the score higher, so the loss keeps
falling below zero forever: it learns nothing useful.

`model.py` + `train.py` fix this the way Word2Vec does:

- **two tables**, one for the word as target and one for the word as context
- **labelled pairs**: `1` for words that belong together, `0` for negative samples
- **`BCEWithLogitsLoss`**: the score goes through a sigmoid and is compared to the
  label, so the loss is bounded and has a real minimum

## How to run

```bash
cd ml-from-scratch/word2vec
python train.py        # prints the loss for each epoch
```

`word2vec.ipynb` needs `sentence-transformers` (in `requirements.txt`) and
downloads the `all-MiniLM-L6-v2` model on first run.

## Known issues / next

- `word2vec.ipynb`: the cell uses `model` without creating it. Add
  `model = SentenceTransformer("all-MiniLM-L6-v2")` before `model.encode(...)`.
- Train on a real text: build (target, context) pairs with a sliding window.
- Sample negatives at random instead of writing them by hand.
- Check the result: plot the 2D vectors and look at nearest neighbours.
