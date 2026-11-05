# Learning resources V2 — Sentence Embeddings

Comprehensive study plan to replace TF-IDF (V1) with sentence embeddings (V2).
**Be thorough.** Embeddings are the foundational concept for modern NLP — solid here = solid for everything after (BERT, GPT, retrieval, semantic search, RAG, etc.).

---

## The big picture — V1 vs V2

```
V1 (TF-IDF)                         V2 (Sentence Embeddings)
─────────────────────────────       ───────────────────────────────────
text → token counts → vector        text → pretrained model → vector
each dim = a word                    each dim = a meaning feature (latent)
sparse (mostly 0s)                   dense (every value matters)
~300 features for 220 phrases        384-768 features (fixed by model)
ignores synonyms                     captures meaning (synonyms ≈ close)
ceiling ~0.65 here                   realistic target 0.80-0.90+
no understanding                     real semantic generalization
```

The **only** thing that changes in your pipeline: the **vectorizer step**.
Everything else (data loading, LogisticRegression classifier, evaluation, persistence) stays identical. You're upgrading **one component**.

---

## Why this is harder than V1 (and worth it)

You're moving from a transparent token-counting algorithm (TF-IDF you can mentally simulate) to a **neural network output** (768-dim vectors produced by a transformer). It's a conceptual leap that requires understanding 4 things you didn't need for V1:

1. **What an embedding is** (a vector in a learned semantic space)
2. **Vector spaces** (geometry of meaning, cosine similarity)
3. **Pretrained models / transfer learning** (you don't train the embedder, you use one someone else trained on billions of sentences)
4. **The transformer architecture** (just the *intuition*, not implementation)

Don't skip these. A common mistake is to use sentence-transformers as a black box — you can ship something, but you can't debug, can't compare models, and can't explain it in a PhD interview at SJTU.

---

## Phase 1 — Conceptual foundations (~4-6h, do not rush)

### 1.1 What is an embedding (~1h)

The single most important concept. Skip if you skip nothing else.

- **Jay Alammar — "The Illustrated Word2Vec"** : [https://jalammar.github.io/illustrated-word2vec/](https://jalammar.github.io/illustrated-word2vec/)
  ↳ Read slowly, take notes. Understand why `king - man + woman ≈ queen`. That's the visual click.
- **StatQuest — "Word Embedding and Word2Vec, Clearly Explained!!!"** : [search YouTube](https://www.youtube.com/results?search_query=StatQuest+Word+Embedding+Word2Vec)
- **Computerphile — "Vectoring Words (Word Embeddings)"** : [search YouTube](https://www.youtube.com/results?search_query=Computerphile+Vectoring+Words)

**Comprehension checkpoint**: you must be able to say out loud *"an embedding is a vector in an N-dimensional space where the distance between 2 vectors reflects their semantic similarity"*. If you can't → re-read Alammar.

### 1.2 Vector spaces + cosine similarity (~30 min)

The geometric intuition. No heavy math needed, just the vibe.

- **3Blue1Brown — "Vectors | Chapter 1, Essence of linear algebra"** : [search YouTube](https://www.youtube.com/results?search_query=3Blue1Brown+Vectors+Essence+of+linear+algebra)
- **3Blue1Brown — "Dot products and duality"** : [search YouTube](https://www.youtube.com/results?search_query=3Blue1Brown+Dot+products+and+duality)
- **Cosine similarity intuition**: [search "cosine similarity intuition"](https://www.google.com/search?q=cosine+similarity+intuition+explained)

**Checkpoint**: you can answer *"why do we use cosine similarity and not Euclidean distance to compare 2 embeddings?"*. (Hint: magnitude vs direction.)

### 1.3 Transformers — intuition, not implementation (~2h)

You don't need to implement a transformer. You need to understand **what it does** at the intuition level. That's what separates a junior from a future PhD.

- **Jay Alammar — "The Illustrated Transformer"** : [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
  ↳ The absolute gold standard. Many people say this single article = 80% of understanding transformers.
- **Jay Alammar — "The Illustrated BERT"** : [https://jalammar.github.io/illustrated-bert/](https://jalammar.github.io/illustrated-bert/)
- **3Blue1Brown — "But what is a GPT? Visual intro to transformers"** : [search YouTube](https://www.youtube.com/results?search_query=3Blue1Brown+But+what+is+a+GPT)

**Checkpoint**: you can explain *"what is the attention mechanism in 2 sentences"* (answer: each word looks at all the others and computes how relevant they are to itself). And *"BERT vs GPT — what's the usage difference"* (bidirectional encoding vs autoregressive generation).

### 1.4 Sentence embeddings specifically (~1h)

How we go from word embeddings to sentence embeddings.

- **Sentence-BERT paper (abstract + intro only)**: [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
  ↳ Read the abstract and the intro (5-10 pages max). You don't need to fully understand the math, just the idea: "BERT is too slow to compare sentences → we train it to produce one vector per sentence".
- **James Briggs — "Sentence Transformers Crash Course"**: [search YouTube](https://www.youtube.com/results?search_query=James+Briggs+sentence+transformers)
- **Sentence-Transformers official docs**: [https://sbert.net/](https://sbert.net/) — section "Pretrained Models"

**Checkpoint**: you can explain why *"all-MiniLM-L6-v2 produces 384-dim vectors while BERT-base produces 768"* (distilled model, smaller, preserves performance).

### 1.5 Transfer learning — the key concept for V2 (~30 min)

Why you can use a model pretrained by others and it works on your tiny dataset.

- **"What is Transfer Learning?" — overview articles**: [search](https://www.google.com/search?q=transfer+learning+NLP+explained)
- **Krish Naik / Andrew Ng** on transfer learning: [search YouTube](https://www.youtube.com/results?search_query=transfer+learning+NLP)

**Checkpoint**: you can answer *"why can I use a model trained on 1 billion web sentences for my classifier on 220 handwritten phrases?"* (answer: semantic embeddings are *general* — they capture sentence meaning independently of domain, so they transfer to any small downstream classifier).

---

## Phase 2 — Practical hands-on (~3-4h)

### 2.1 Installation + hello world (~15 min)

```bash
pip install sentence-transformers
```

In your notebook, load a model and encode 3 sentences. Check `.shape` (should be `(3, 384)` for MiniLM).

**Main doc**: [https://sbert.net/docs/quickstart.html](https://sbert.net/docs/quickstart.html)

### 2.2 Mini similarity experiment (~30 min)

The pedagogical experiment that seals the click. To do in your notebook.

1. Encode 4 sentences from your dataset (2 from the same intent, 2 from different intents)
2. Compute **cosine similarity** between all pairs (4x4 matrix)
3. Verify: intra-intent pairs have high similarity (~0.6-0.9), inter-intent pairs have low similarity (~0-0.3)

→ You'll see the embeddings "understand" your intents BEFORE you train anything. That's THE click.

**Cosine similarity doc in sentence-transformers**: [https://sbert.net/docs/usage/semantic_textual_similarity.html](https://sbert.net/docs/usage/semantic_textual_similarity.html)

### 2.3 Embeddings + LogisticRegression (~1h)

The direct upgrade from your V1.

**Simple strategy** (recommended to start): encode ALL your X upfront, then sklearn classifier on those embeddings as features.

```
Conceptual workflow (you implement it):
1. model = SentenceTransformer("all-MiniLM-L6-v2")
2. X_emb = model.encode(X)                      # shape (220, 384)
3. cross_val_score(LogisticRegression(...), X_emb, Y, cv=5, scoring="f1_macro")
4. Compare to your V1 baseline (0.629)
```

⚠️ **You don't use `Pipeline` this time** (unless you wrap SentenceTransformer in a custom class). It's simpler to encode upfront.

**Classification example with embeddings**: [https://sbert.net/examples/training/sts/README.html](https://sbert.net/examples/training/sts/README.html) (adapt the idea to your classification)

### 2.4 Compare V1 vs V2 (~30 min)

Metrics side by side:
- F1 macro (cross_val)
- Cross_val std (stability)
- Confusion matrix (where it improves, where it stagnates)

Re-use your V1 tools (`classification_report`, `ConfusionMatrixDisplay`) on V2 predictions.

### 2.5 Production: `train_v2.py` (~1h)

When you're happy with V2, write a clean `train_v2.py` (like `train.py` but for V2). It:
1. Loads the sentence-transformers model
2. Encodes X
3. Fits LogisticRegression
4. Saves **both the classifier** (joblib) **and the embedding model name** (json/yaml) — important: your backend will have to load BOTH

⚠️ **Crucial detail for serving**: your backend will have to
1. Load the SentenceTransformer (~80 MB for MiniLM)
2. Load the LogisticRegression (the .joblib)
3. Pipeline: `text → model.encode → classifier.predict`

That's slightly heavier than V1 (TF-IDF was inside the .joblib directly).

---

## Phase 3 — Optimization & comparison (~2-3h, optional but formative)

### 3.1 Try multiple embedding models

Compare `all-MiniLM-L6-v2` vs `all-mpnet-base-v2` vs `multi-qa-MiniLM-L6-cos-v1`. Comparison table: F1 macro, model size, latency.

**Model list + benchmarks**: [https://sbert.net/docs/pretrained_models.html](https://sbert.net/docs/pretrained_models.html)

### 3.2 Visualize your embeddings (~1h)

Reduce 384 dim → 2 dim with UMAP or PCA, plot with matplotlib coloring by intent.

→ If your 10 clusters are visually separated, your model has an easy job. If some overlap, those are your future confusion pairs.

- **UMAP**: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
- **StatQuest — UMAP**: [search YouTube](https://www.youtube.com/results?search_query=StatQuest+UMAP)
- **StatQuest — PCA**: [search YouTube](https://www.youtube.com/results?search_query=StatQuest+PCA)

This is DEEPLY useful: you'll literally *see* how your model organizes your intents.

### 3.3 Fine-tuning (advanced, V3 later)

You can fine-tune the embedding model on your own sentence pairs. That's V3 territory. Skip for V2.

For your reference: [https://sbert.net/docs/training/overview.html](https://sbert.net/docs/training/overview.html)

---

## Suggested attack order (1 week, 2h/day)

| Day | Focus | Deliverable |
|------|-------|-------------|
| **Day 1** | Phase 1.1 (Illustrated Word2Vec) + 1.2 (vectors) | You can explain "what is an embedding" |
| **Day 2** | Phase 1.3 (Illustrated Transformers + BERT) | You know the role of attention |
| **Day 3** | Phase 1.4 (Sentence-BERT) + 1.5 (transfer learning) | You understand why pretrained works |
| **Day 4** | Phase 2.1-2.2 (install + hello + similarity) | Mini cosine experiment succeeded in your notebook |
| **Day 5** | Phase 2.3-2.4 (classifier + compare V1/V2) | You have a V2 vs V1 F1 number |
| **Day 6** | Phase 2.5 (train_v2.py) | V2 production script ready |
| **Day 7** | Phase 3.2 (UMAP visualization) — bonus | You have a 2D color plot of your intents |

**~12-15h total spread over 1 week.** No rush. Phase 1 is the foundation.

---

## Bonus — Resources to go further (useful for SJTU PhD interview)

### Free full courses

- **Hugging Face NLP Course**: [https://huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)
  ↳ The standard. Chapters 1-3 for V2, chapters 4+ for V3.
- **Stanford CS224N — NLP with Deep Learning**: [search YouTube](https://www.youtube.com/results?search_query=Stanford+CS224N+NLP+Deep+Learning)
  ↳ Free lectures on YouTube. Very dense, master/PhD level. Watch progressively.

### Papers to know (skim at least the abstract)

1. **Word2Vec — Mikolov et al. 2013**: "Efficient Estimation of Word Representations in Vector Space" — [arxiv 1301.3781](https://arxiv.org/abs/1301.3781)
2. **Attention Is All You Need — Vaswani et al. 2017**: THE transformers paper — [arxiv 1706.03762](https://arxiv.org/abs/1706.03762)
3. **BERT — Devlin et al. 2018**: [arxiv 1810.04805](https://arxiv.org/abs/1810.04805)
4. **Sentence-BERT — Reimers & Gurevych 2019**: [arxiv 1908.10084](https://arxiv.org/abs/1908.10084)

You don't need to read them in full. But **knowing they exist and what they propose** matters.

### For your general ML culture

- **3Blue1Brown — Neural Networks series**: [search YouTube](https://www.youtube.com/results?search_query=3Blue1Brown+Neural+Networks) (4 videos, visual foundation of NNs)
- **Andrej Karpathy — Neural Networks: Zero to Hero**: [search YouTube](https://www.youtube.com/results?search_query=Andrej+Karpathy+Neural+Networks+Zero+to+Hero) (gold series for V3+)

---

## Self-evaluation checkpoints

At the end of each phase, ask yourself:

**Phase 1 (concepts)**:
- [ ] I can explain what an embedding is to a beginner
- [ ] I understand why cosine similarity > Euclidean distance for embeddings
- [ ] I know the role of the attention mechanism in transformers
- [ ] I know why a pretrained model can classify my 220 phrases

**Phase 2 (practice)**:
- [ ] I encoded my sentences and looked at `.shape`
- [ ] I computed cosine similarity between 4 sentences and saw the intra-intent vs inter-intent pattern
- [ ] I trained LogReg on the embeddings and compared to V1 via cross_val_score
- [ ] I have a `train_v2.py` that saves the .joblib + the embedding model name

**Phase 3 (optional)**:
- [ ] I tested 2-3 different models
- [ ] I have a 2D UMAP plot of my embeddings colored by intent

---

## The golden rule of this phase

> **Understand > implement.** If you can implement without understanding, you only copied. If you understand without having implemented, you can always implement later. Always prioritize understanding.

What distinguishes a future SJTU PhD student from a junior: they **explain why their code works**. Not just *that* it works.

Good learning.
