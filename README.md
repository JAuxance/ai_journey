# AI Journey

A structured 24-month journey from software-engineering foundations to a competitive
level in AI — every step logged, every project explained, every conclusion documented.

Target: portfolio strong enough for a top AI-focused university (currently aiming
for **SJTU**), built by hand with the goal of **understanding**, not just shipping.

---

## Current focus

> **Day 56 Focus -> Portfolio Bot (Trail) - V2 embeddings**

```
V1 (TF-IDF) ██████████ frozen at F1 0.629   ·   V2 (embeddings) ░░░░░░░░░░ studying
```

- V1 model trained by hand: TF-IDF + LogisticRegression, **F1 macro 0.629** (5-fold CV)
- `train.py` ships `intent_classifier.joblib` (V1 production-grade)
- Now: studying sentence embeddings (transfer learning, sentence-BERT) per `04-projects/02-portfolio-bot/training/v2/learning-resources.md`
- Next: V2 model with embeddings → expected F1 jump to 0.80+ → deploy V2 directly

---

## Repo map

What actually exists, and where to go:

| Folder | What's there | Status |
|--------|--------------|--------|
| [`00-foundations/`](./00-foundations) | Python scientific stack (numpy, pandas, matplotlib) — journals + exercises | ✅ active |
| [`01-math-for-ai/`](./01-math-for-ai) | Math for AI — currently focused on linear algebra | 🔵 partial (linalg done, calculus/proba/stats to come) |
| [`02-machine-learning/`](./02-machine-learning) | Classical ML — sklearn journal + notebook experiments | ✅ active |
| [`04-projects/`](./04-projects) | End-to-end portfolio projects (the showcase) | ✅ active |
| [`docker/`](./docker) | Containerized Jupyter environment for the whole repo | ✅ available |
| `ROADMAP.md` | The 24-month plan (4 phases) | 📋 reference |

> **Folders not yet present** (Phase 3 onwards): `03-deep-learning/`, `05-notes/`,
> `06-interview-prep/`. They'll be created when the corresponding work actually starts —
> no empty placeholders here, only living folders.

---

## Active projects

### [Project 01 — Student Performance Predictor](./04-projects/01-student-performance-predictor) ✅
First end-to-end ML project. Tabular regression with feature engineering, multiple
models compared, documented results.

### [Project 02 — Portfolio Bot "Trail"](./04-projects/02-portfolio-bot) 🚀
The chatbot **is** the portfolio. A handcrafted intent classifier that visitors
chat with to learn about my work. Current state: V1 trained (F1 0.629), V2 embeddings
study in progress.

- [`data/`](./04-projects/02-portfolio-bot/data) — `intents.yaml`, `responses.yaml` (handwritten)
- [`training/v1/`](./04-projects/02-portfolio-bot/training/v1) — V1 notebook + train.py
- [`training/v2/`](./04-projects/02-portfolio-bot/training/v2) — V2 study plan + experiments
- [`models/`](./04-projects/02-portfolio-bot/models) — joblib artifacts (gitignored)

Production code (FastAPI backend + frontend) lives in a **separate portfolio repo**;
this folder keeps only the ML/training side of Trail.

---

## Roadmap (4 phases over 24 months)

See [`ROADMAP.md`](./ROADMAP.md) for the full plan. Headline:

1. **Foundations** (months 1-2) — scientific Python + math basics + 1st ML project ✅ in progress
2. **Classical ML** (months 3-6) — sklearn algorithms + 2 portfolio projects 🔵 mid-flight
3. **Deep Learning** (months 6-12) — PyTorch, CNNs, RNNs, Transformers ⏳ planned
4. **Research-grade portfolio** (months 12-24) — paper reproduction, original work ⏳ planned

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

---

## Conventions

- `kebab-case` for folder names, `snake_case` for Python files
- Each notable folder has its own `README.md` explaining what's inside
- ML projects follow: `data/` → `notebooks/` (exploration) → `src/` or `training/` (production)
- Versioning when relevant: `v1/`, `v2/`, `v3/` (see portfolio bot for example)
- Commits use Conventional Commits style (`feat:`, `chore:`, `refactor:`, `docs:`, etc.)

---

## Rules I hold myself to

- **Code clean and documented.** Every script has a purpose statement at the top.
- **Every project includes results and conclusions.** Not "it works" — *what* it does, *how well*, *why*, *what's next*.
- **I do not only train models: I explain them.** A model I can't explain is a model I don't own.
- **Fundamentals over hype.** Hand-coded ML before LLM APIs. Understand before scale.
- **Ship the smallest honest version, then iterate.** V1 < V2 < V3, each one shippable on its own.
