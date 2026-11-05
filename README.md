# AI Journey

A self-directed path into AI, built by hand with the goal of **understanding**,
not just shipping.

## Goal

Build a portfolio strong enough to apply for a **master's or PhD in AI**
(target: **SJTU**). Every folder in this repo is meant to show something I
actually understand: the math, the models rebuilt from scratch, and the projects
where I apply them.

---

## Current focus

**Until the end of December 2026: `math/` and `ml-from-scratch/`.**

- [`math/`](./math): finish the CS229 linear algebra review (eigenvalues, matrix calculus)
- [`ml-from-scratch/`](./ml-from-scratch): take Word2Vec from a toy example to a real corpus

The portfolio bot **Trail** is **paused until January 2027**. Its V2 needs the
foundations above first.

---

## Status

| Folder | Status | Notes |
|---|---|---|
| [`foundations/numpy`](./foundations/numpy) | Done | Learning journal + notes. Knowledge test scored 15/20 |
| [`foundations/pandas`](./foundations/pandas) | Done | Learning journal (11 sections) |
| [`foundations/matplotlib`](./foundations/matplotlib) | Done | Learning journal (8 chart types) + saved figures |
| [`math/linear-algebra`](./math/linear-algebra) | Incomplete | CS229 sections 1–3.5 done (11 topics); rank, inverse, eigenvalues, matrix calculus still to do |
| [`ml-from-scratch/word2vec`](./ml-from-scratch/word2vec) | Tested | Skip-gram + negative sampling in PyTorch; trains on 8 toy words, not evaluated yet |
| [`projects/student-performance-predictor`](./projects/student-performance-predictor) | Done | LinearRegression, test R² ≈ 0.73, 4 models compared |
| [`projects/portfolio-bot-trail`](./projects/portfolio-bot-trail) | Incomplete | V1 frozen (TF-IDF + LogReg, F1 macro 0.629). V2 paused until January 2027 |
| [`journal/`](./journal) | Planned | Format defined, no entries yet |

**Legend**

- **Done**: finished, results written down, nothing pending
- **Understood**: I can explain it without notes, but it is not finished or applied yet
- **Tested**: the code runs and does what it should on a small case, not evaluated seriously
- **Planned**: decided and scoped, work not started
- **Incomplete**: started, with known parts missing (listed in Notes)

---

## Structure

```
ai_journey/
├── foundations/        NumPy, pandas, matplotlib: learning journals
├── math/               Math for AI, from the source material, checked in NumPy
├── ml-from-scratch/    Models rebuilt by hand to understand how they work
├── projects/           Applied projects with a full ML workflow
├── journal/            One dated entry per work session
└── docker/             Reproducible Jupyter environment
```

Each folder has its own README with details and status.

---

## How to run

```bash
# With Docker (recommended — full Jupyter environment)
cd docker
docker compose up --build       # http://localhost:8888

# Locally
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Student performance pipeline (needs `StudentPerformanceFactors.csv` in
`projects/student-performance-predictor/data/`, the CSV is not committed):

```bash
cd docker
docker compose --profile pipeline run --rm pipeline
```

Word2Vec from scratch:

```bash
cd ml-from-scratch/word2vec
python train.py
```

---

## Rules I hold myself to

- **Code clean and documented.** Every script has a purpose statement at the top.
- **Every project includes results and conclusions.** Not "it works" — *what* it does, *how well*, *why*, *what's next*.
- **I do not only train models: I explain them.** A model I can't explain is a model I don't own.
- **Fundamentals over hype.** Hand-coded ML before LLM APIs. Understand before scale.
- **Ship the smallest honest version, then iterate.** V1 < V2 < V3, each one shippable on its own.
