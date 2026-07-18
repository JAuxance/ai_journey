# AI Journey

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

## Active projects

### [Portfolio Bot "Trail"](./4-models_released/02-portfolio-bot) 
Interactive chatbot serving as my portfolio. Visitors chat with a hand-coded model to discover my work.

- **V1 (Prod)**: TF-IDF + Logistic Regression (F1: 0.629) 
- **V2 (WIP)**: Upgrading to Sentence Embeddings (Target F1 > 0.80) 

**Structure**: 
- [`data/`](./4-models_released/02-portfolio-bot/data) — Knowledge base (`intents.yaml`, `responses.yaml`)
- [`training/`](./4-models_released/02-portfolio-bot/training) — Exploration notebooks and `train.py` scripts
- [`models/`](./4-models_released/02-portfolio-bot/models) — Serialized artifacts (`.joblib`)
---

##  Repository Structure

This repo follows a strict progression to avoid cognitive debt:

- **`1-foundations`** : Core algorithmics and data fundamentals.
- **`2-math-for-ai`** : Mathematical concepts implemented from scratch (NumPy).
- **`3-model_sandbox`** : ️ Experimentation zone. Work in progress, drafts, tests. Do NOT use in production.
- **`4-models_released`** : ✅ Production zone. Finished, cleaned, documented, and ready-to-use models.
- **`docker/`** : Reproducible development environment.


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


## Rules I hold myself to

- **Code clean and documented.** Every script has a purpose statement at the top.
- **Every project includes results and conclusions.** Not "it works" — *what* it does, *how well*, *why*, *what's next*.
- **I do not only train models: I explain them.** A model I can't explain is a model I don't own.
- **Fundamentals over hype.** Hand-coded ML before LLM APIs. Understand before scale.
- **Ship the smallest honest version, then iterate.** V1 < V2 < V3, each one shippable on its own.
