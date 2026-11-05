# Math

The math behind the models, studied from the source material and checked in code
with NumPy. The goal is to be able to read a paper or a model's `forward()` and
know what every operation means, not just which function to call.

---

## Content

| Folder | Source | Status |
|---|---|---|
| [`linear-algebra/`](./linear-algebra) | [Stanford CS229 — Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf) | Incomplete — sections 1–3.5 done |

### Linear algebra

Each topic follows the same pattern inside
[`linear_algebra.ipynb`](./linear-algebra/linear_algebra.ipynb):
**Definition → Intuition → Example (NumPy)**.

Done so far: notation, dot and outer products, matrix-vector and matrix-matrix
products, identity and diagonal matrices, transpose, symmetric matrices, trace
(with a proof of `tr(AB) = tr(BA)`), vector and matrix norms.

Next in CS229: linear independence and rank, inverse, orthogonal matrices,
range and nullspace, determinant, quadratic forms, eigenvalues and eigenvectors,
then matrix calculus (gradients, Hessians).

---

## Planned

These are the next areas, in order. Each one gets its own folder when work starts.

1. **Linear algebra** — finish CS229 (eigenvalues, matrix calculus)
2. **Probability and statistics** — distributions, expectation, Bayes, MLE
3. **Calculus for ML** — derivatives, chain rule, gradients (the base of backpropagation)

Word2Vec notes, previously in this section, moved to
[`ml-from-scratch/word2vec/`](../ml-from-scratch/word2vec).
