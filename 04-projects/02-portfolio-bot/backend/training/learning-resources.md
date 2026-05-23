# Learning resources — `train.py` prep

Click in order. Each link opens what you need next.

---

## 🎯 Deadline : Tuesday 2026-05-26

Goal: `train.py` runs, all core concepts (TF-IDF, multinomial LogReg, confusion matrix, cross-validation) can be explained out loud. Residual deep-math questions are OK and expected.

| Day | Time | Plan |
|-----|------|------|
| **Sat 2026-05-23** | 2h | Video TF-IDF (Machine Learnia) + StatQuest Logistic Regression. Take notes. |
| **Sun 2026-05-24** | 4h | Morning: 4 remaining videos (Multinomial LogReg, Confusion Matrix, Precision/Recall, Cross Validation). Afternoon: sklearn "Working with Text Data" tutorial — read + reproduce in a notebook. |
| **Mon 2026-05-25** | 3-4h | Paper-sketch `train.py`. Code the base: load YAML → split → Pipeline → first `.fit()` + accuracy. |
| **Tue 2026-05-26** | 2-3h | Add `classification_report`, `confusion_matrix`, `cross_val_score`, `joblib.dump`. Manual sanity-check on a few sentences. ✅ |

Total: ~12h over 4 days. Skip-day = catch up the next day, don't push the deadline.

---

## Phase 1 — Videos (intuition, ~1h30 total)

YouTube search links — pick the top result.

1. [TF-IDF — Machine Learnia (FR)](https://www.youtube.com/results?search_query=Machine+Learnia+NLP+TF-IDF)
   *Alt EN:* [TF-IDF — Ritvikmath](https://www.youtube.com/results?search_query=Ritvikmath+TF-IDF+Explained+Clearly)
2. [Logistic Regression — StatQuest](https://www.youtube.com/results?search_query=StatQuest+Logistic+Regression+Clearly+Explained)
3. [Multinomial Logistic Regression — StatQuest](https://www.youtube.com/results?search_query=StatQuest+Multinomial+Logistic+Regression)
4. [Confusion Matrix — StatQuest](https://www.youtube.com/results?search_query=StatQuest+Confusion+Matrix)
5. [Precision & Recall — StatQuest](https://www.youtube.com/results?search_query=StatQuest+Precision+and+Recall)
6. [Cross Validation — StatQuest](https://www.youtube.com/results?search_query=StatQuest+Cross+Validation)

---

## Phase 2 — The killer tutorial (~45 min, read fully)

7. ⭐ [sklearn — Working With Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

Almost exactly this project. Read with an editor open.

---

## Phase 3 — Reference docs (keep open while coding)

**YAML loading**
8. [PyYAML documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)

**TF-IDF**
9. [sklearn — TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
10. [sklearn — Text feature extraction (deep dive)](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

**Logistic Regression**
11. [sklearn — LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

**Pipeline**
12. [sklearn — Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
13. [sklearn — Composite estimators](https://scikit-learn.org/stable/modules/compose.html#pipeline)

**Train/test split**
14. [sklearn — train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

**Evaluation**
15. [sklearn — classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
16. [sklearn — confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
17. [sklearn — ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)

**Cross-validation**
18. [sklearn — cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
19. [sklearn — Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)

**Persist the model**
20. [sklearn — Model persistence](https://scikit-learn.org/stable/model_persistence.html)

---

## Bonus — full FR ML course

21. [Machine Learnia — Apprendre le Machine Learning de A à Z](https://www.youtube.com/results?search_query=Machine+Learnia+Apprendre+le+Machine+Learning+de+A+%C3%A0+Z)
    ~30h free. Target the logistic regression and NLP episodes.

---

## Pipeline you're building

```
YAML  →  list of (text, label)
            ↓
     train_test_split (stratified, 80/20)
            ↓
        Pipeline
   ┌──────────────────┐
   │ TfidfVectorizer  │   text → numeric vectors
   │ LogisticRegression │ vectors → predicted intent
   └──────────────────┘
            ↓
   fit → classification_report + confusion_matrix
            ↓
   joblib.dump(model, "intent_classifier.joblib")
```

---

## Key params to know (cheat sheet)

**TfidfVectorizer**
- `ngram_range=(1, 2)` — captures bigrams like `"ai journey"`
- `analyzer="char_wb"` — char n-grams help with typos (`"helo"`, `"hii"`)
- `min_df`, `max_df` — filter rare / frequent tokens

**LogisticRegression**
- `class_weight="balanced"` — important: `out_of_scope` has more examples than others
- `C=1.0` — regularization (smaller = stronger)
- `max_iter=1000` — bump if convergence warning

**train_test_split**
- `stratify=y` — keep class proportions in both splits
- `random_state=42` — reproducibility

---

## Total time

- Videos 1→6 : ~1h30
- Tutorial 7 : ~45 min
- Coding : ~2-3h

≈ 5-6h for a working `train.py`.
