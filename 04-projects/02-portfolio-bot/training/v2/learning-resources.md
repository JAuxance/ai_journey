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

The **only** thing that changes in your pipeline : the **vectorizer step**.
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

- 🎥 **Jay Alammar — "The Illustrated Word2Vec"** : [https://jalammar.github.io/illustrated-word2vec/](https://jalammar.github.io/illustrated-word2vec/)
  ↳ Lecture lente, prend des notes. Comprends pourquoi `king - man + woman ≈ queen`. C'est le déclic visuel.
- 🎥 **StatQuest — "Word Embedding and Word2Vec, Clearly Explained!!!"** : [search YouTube](https://www.youtube.com/results?search_query=StatQuest+Word+Embedding+Word2Vec)
- 🎥 **Computerphile — "Vectoring Words (Word Embeddings)"** : [search YouTube](https://www.youtube.com/results?search_query=Computerphile+Vectoring+Words)

**Checkpoint comprehension** : tu dois pouvoir expliquer à voix haute *"un embedding c'est un vecteur dans un espace de N dimensions où la distance entre 2 vecteurs reflète leur similarité sémantique"*. Si tu ne peux pas → relis Alammar.

### 1.2 Vector spaces + cosine similarity (~30 min)

L'intuition géométrique. Pas besoin de maths lourdes, juste la vibe.

- 🎥 **3Blue1Brown — "Vectors | Chapter 1, Essence of linear algebra"** : [search YouTube](https://www.youtube.com/results?search_query=3Blue1Brown+Vectors+Essence+of+linear+algebra)
- 🎥 **3Blue1Brown — "Dot products and duality"** : [search YouTube](https://www.youtube.com/results?search_query=3Blue1Brown+Dot+products+and+duality)
- 📖 **Cosine similarity intuitive** : [search "cosine similarity intuition"](https://www.google.com/search?q=cosine+similarity+intuition+explained)

**Checkpoint** : tu sais répondre *"pourquoi on utilise la cosine similarity et pas la distance euclidienne pour comparer 2 embeddings ?"*. (Indice : magnitude vs direction.)

### 1.3 Transformers — l'intuition, pas l'implémentation (~2h)

Tu n'as pas besoin d'implémenter un transformer. Tu as besoin de comprendre **ce qu'il fait** au niveau intuitif. C'est ce qui sépare un junior d'un futur PhD.

- 🎥 **Jay Alammar — "The Illustrated Transformer"** : [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
  ↳ Le gold standard absolu. Pas mal de monde dit que cet article seul = 80% de la compréhension transformers.
- 🎥 **Jay Alammar — "The Illustrated BERT"** : [https://jalammar.github.io/illustrated-bert/](https://jalammar.github.io/illustrated-bert/)
- 🎥 **3Blue1Brown — "But what is a GPT? Visual intro to transformers"** : [search YouTube](https://www.youtube.com/results?search_query=3Blue1Brown+But+what+is+a+GPT)

**Checkpoint** : tu sais expliquer *"l'attention mechanism, c'est quoi en 2 phrases"* (réponse : chaque mot regarde tous les autres et calcule à quel point ils sont pertinents pour lui-même). Et *"BERT vs GPT, c'est quoi la différence d'usage"* (bidirectional encoding vs autoregressive generation).

### 1.4 Sentence embeddings spécifiquement (~1h)

Comment on passe d'embeddings de mots à embeddings de phrases.

- 📖 **Sentence-BERT paper (abstract + intro seulement)** : [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
  ↳ Lis l'abstract et l'intro (5-10 pages max). Tu n'as pas besoin de comprendre les maths complètes, juste l'idée : "BERT est trop lent pour comparer des phrases → on entraîne pour produire un vecteur par phrase".
- 🎥 **James Briggs — "Sentence Transformers Crash Course"** : [search YouTube](https://www.youtube.com/results?search_query=James+Briggs+sentence+transformers)
- 📖 **Sentence-Transformers official docs** : [https://sbert.net/](https://sbert.net/) — section "Pretrained Models"

**Checkpoint** : tu sais expliquer pourquoi *"all-MiniLM-L6-v2 produit des vecteurs de 384 dim alors que BERT-base produit du 768"* (modèle distillé, plus petit, conserve la perf).

### 1.5 Transfer learning — le concept-clé pour V2 (~30 min)

Pourquoi tu peux utiliser un modèle pré-entraîné par d'autres et que ça marche sur ton dataset minuscule.

- 📖 **"What is Transfer Learning?" — overview articles** : [search](https://www.google.com/search?q=transfer+learning+NLP+explained)
- 🎥 **Krish Naik / Andrew Ng** sur transfer learning : [search YouTube](https://www.youtube.com/results?search_query=transfer+learning+NLP)

**Checkpoint** : tu sais répondre *"pourquoi je peux utiliser un modèle entraîné sur 1 milliard de phrases du web pour mon classifier sur 220 phrases handwritten ?"* (réponse : les embeddings sémantiques sont *généralistes*, ils capturent le sens des phrases indépendamment du domaine, donc ils transfèrent à n'importe quel petit classifier downstream).

---

## Phase 2 — Practical hands-on (~3-4h)

### 2.1 Installation + hello world (~15 min)

```bash
pip install sentence-transformers
```

Dans ton notebook, charge un modèle et encode 3 phrases. Regarde `.shape` (doit être `(3, 384)` pour MiniLM).

📖 **Doc principale** : [https://sbert.net/docs/quickstart.html](https://sbert.net/docs/quickstart.html)

### 2.2 Mini-expérience de similarité (~30 min)

L'expérience pédagogique qui scelle le déclic. À faire dans ton notebook.

1. Encode 4 phrases de ton dataset (2 du même intent, 2 d'intents différents)
2. Calcule la **cosine similarity** entre toutes les paires (matrice 4x4)
3. Vérifie : les paires intra-intent ont des similarités élevées (~0.6-0.9), les paires inter-intent ont des similarités basses (~0-0.3)

→ Tu verras les embeddings "comprendre" tes intents AVANT même d'entraîner quoi que ce soit. C'est ÇA le déclic.

📖 **Doc cosine similarity dans sentence-transformers** : [https://sbert.net/docs/usage/semantic_textual_similarity.html](https://sbert.net/docs/usage/semantic_textual_similarity.html)

### 2.3 Embeddings + LogisticRegression (~1h)

Le upgrade direct de ton V1.

**Stratégie simple** (recommandée pour démarrer) : encode TOUT ton X upfront, puis classifier sklearn sur ces embeddings comme features.

```
Workflow conceptuel (à coder par toi) :
1. model = SentenceTransformer("all-MiniLM-L6-v2")
2. X_emb = model.encode(X)                      # shape (220, 384)
3. cross_val_score(LogisticRegression(...), X_emb, Y, cv=5, scoring="f1_macro")
4. Compare à ton V1 baseline (0.629)
```

⚠️ **Tu n'utilises pas `Pipeline` cette fois** (sauf si tu wraps le SentenceTransformer dans une classe custom). C'est plus simple d'encoder upfront.

📖 **Exemple de classif avec embeddings** : [https://sbert.net/examples/training/sts/README.html](https://sbert.net/examples/training/sts/README.html) (adapter l'idée à ta classif)

### 2.4 Compare V1 vs V2 (~30 min)

Métriques côte à côte :
- F1 macro (cross_val)
- Std de cross_val (stabilité)
- Confusion matrix (où ça s'améliore, où ça stagne)

📖 Réutilise tes outils du V1 (`classification_report`, `ConfusionMatrixDisplay`) sur les prédictions V2.

### 2.5 Production : `train_v2.py` (~1h)

Quand tu es content de V2, écris `train_v2.py` propre (comme `train.py` mais pour V2). Il :
1. Charge sentence-transformers model
2. Encode X
3. Fit LogisticRegression
4. Sauve **et le classifier** (joblib) **et le nom du modèle d'embedding** (json/yaml) — important : ton backend devra charger les DEUX

⚠️ **Détail crucial pour le serving** : ton backend devra
1. Charger le SentenceTransformer (~80 MB pour MiniLM)
2. Charger le LogisticRegression (le .joblib)
3. Pipeline : `text → model.encode → classifier.predict`

C'est légèrement plus lourd qu'en V1 (TF-IDF était dans le .joblib direct).

---

## Phase 3 — Optimization & comparison (~2-3h, optionnel mais formateur)

### 3.1 Essayer plusieurs modèles d'embedding

Compare `all-MiniLM-L6-v2` vs `all-mpnet-base-v2` vs `multi-qa-MiniLM-L6-cos-v1`. Tableau de comparaison : F1 macro, taille du modèle, latence.

📖 **Liste des modèles + benchmarks** : [https://sbert.net/docs/pretrained_models.html](https://sbert.net/docs/pretrained_models.html)

### 3.2 Visualiser tes embeddings (~1h)

Réduire 384 dim → 2 dim avec UMAP ou PCA, plot avec matplotlib en colorant par intent.

→ Si tes 10 clusters sont visuellement séparés, ton modèle a un boulot facile. Si certains se mélangent, c'est tes paires de confusion futures.

- 📖 **UMAP** : [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
- 🎥 **StatQuest — UMAP** : [search YouTube](https://www.youtube.com/results?search_query=StatQuest+UMAP)
- 🎥 **StatQuest — PCA** : [search YouTube](https://www.youtube.com/results?search_query=StatQuest+PCA)

C'est PROFONDÉMENT utile : tu vas littéralement *voir* comment ton modèle organise tes intents.

### 3.3 Fine-tuning (advanced, V3 plus tard)

Tu peux fine-tuner le modèle d'embedding sur tes propres paires de phrases. C'est V3 territory. Skip pour V2.

📖 Pour ta culture : [https://sbert.net/docs/training/overview.html](https://sbert.net/docs/training/overview.html)

---

## Ordre d'attaque suggéré (1 semaine, 2h/jour)

| Jour | Focus | Livrable |
|------|-------|----------|
| **Jour 1** | Phase 1.1 (Word2Vec illustré) + 1.2 (vecteurs) | Tu peux expliquer "qu'est-ce qu'un embedding" |
| **Jour 2** | Phase 1.3 (Transformers illustré + BERT) | Tu connais le rôle de l'attention |
| **Jour 3** | Phase 1.4 (Sentence-BERT) + 1.5 (transfer learning) | Tu comprends pourquoi pretrained marche |
| **Jour 4** | Phase 2.1-2.2 (install + hello + similarity) | Mini-expérience cosine réussie dans ton notebook |
| **Jour 5** | Phase 2.3-2.4 (classifier + compare V1/V2) | Tu as un chiffre F1 V2 vs V1 |
| **Jour 6** | Phase 2.5 (train_v2.py) | Script de prod V2 prêt |
| **Jour 7** | Phase 3.2 (visualisation UMAP) — bonus | Tu as un plot 2D coloré de tes intents |

**~12-15h total réparties sur 1 semaine.** Pas de précipitation. La phase 1 est la fondation.

---

## Bonus — Ressources pour aller plus loin (utile pour SJTU PhD interview)

### Cours complets gratuits

- **Hugging Face NLP Course** : [https://huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)
  ↳ Le standard. Chapitres 1-3 pour V2, chapitres 4+ pour V3.
- **Stanford CS224N — NLP with Deep Learning** : [search YouTube](https://www.youtube.com/results?search_query=Stanford+CS224N+NLP+Deep+Learning)
  ↳ Lectures gratuites sur YouTube. Très costaud, niveau master/PhD. À regarder progressivement.

### Papers à connaître (skim au moins l'abstract)

1. **Word2Vec — Mikolov et al. 2013** : "Efficient Estimation of Word Representations in Vector Space" — [arxiv 1301.3781](https://arxiv.org/abs/1301.3781)
2. **Attention Is All You Need — Vaswani et al. 2017** : LE paper transformers — [arxiv 1706.03762](https://arxiv.org/abs/1706.03762)
3. **BERT — Devlin et al. 2018** : [arxiv 1810.04805](https://arxiv.org/abs/1810.04805)
4. **Sentence-BERT — Reimers & Gurevych 2019** : [arxiv 1908.10084](https://arxiv.org/abs/1908.10084)

Tu n'as pas besoin de les lire en entier. Mais **savoir qu'ils existent et ce qu'ils proposent** est important.

### Pour ta culture générale ML

- **3Blue1Brown — série Neural Networks** : [search YouTube](https://www.youtube.com/results?search_query=3Blue1Brown+Neural+Networks) (4 vidéos, fondation visuelle des NNs)
- **Andrej Karpathy — Neural Networks: Zero to Hero** : [search YouTube](https://www.youtube.com/results?search_query=Andrej+Karpathy+Neural+Networks+Zero+to+Hero) (série gold pour V3+)

---

## Checkpoints d'auto-évaluation

À la fin de chaque phase, demande-toi :

**Phase 1 (concepts)** :
- ✅ Je peux expliquer ce qu'est un embedding à un débutant
- ✅ Je comprends pourquoi la cosine similarity > la distance euclidienne pour les embeddings
- ✅ Je connais le rôle du mécanisme d'attention dans les transformers
- ✅ Je sais pourquoi un modèle pretrained peut classer mes 220 phrases

**Phase 2 (pratique)** :
- ✅ J'ai encodé mes phrases et regardé le `.shape`
- ✅ J'ai calculé la cosine similarity entre 4 phrases et vu le pattern intent-intra vs intent-inter
- ✅ J'ai entraîné LogReg sur les embeddings et comparé à V1 via cross_val_score
- ✅ J'ai un `train_v2.py` qui sauve le .joblib + le nom du modèle d'embedding

**Phase 3 (optionnel)** :
- ✅ J'ai testé 2-3 modèles différents
- ✅ J'ai un plot UMAP 2D de mes embeddings colorés par intent

---

## La règle d'or de la phase

> **Comprendre > implémenter.** Si tu peux implémenter sans comprendre, tu n'as fait que recopier. Si tu comprends sans avoir implémenté, tu peux toujours implémenter après. Toujours privilégier comprendre.

Ce qui distingue un futur étudiant SJTU PhD d'un junior, c'est qu'il **explique pourquoi son code marche**. Pas juste *que* ça marche.

Bon apprentissage. 🚀
