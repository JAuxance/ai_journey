# ML from Scratch

Models rebuilt by hand, small enough to understand every line. The point is the
mechanism, not the score: each folder shows how a model works inside before I
use a library version of it in a project.

Code here can be rough or unfinished. Each folder's README states honestly how
far it goes.

**Rule:** when a model from this folder is understood, tested and used for a real
task, it moves into its own folder under [`projects/`](../projects) and gets
cleaned up there.

---

## Content

| Folder | What | Status |
|---|---|---|
| [`word2vec/`](./word2vec) | Skip-gram Word2Vec with negative sampling, in PyTorch, on a toy vocabulary | Tested |
