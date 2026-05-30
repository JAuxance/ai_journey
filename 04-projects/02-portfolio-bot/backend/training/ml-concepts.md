# ML concepts for `train.py` — what you need to understand

Conceptual reference for the Trail intent classifier (Phase 2).
This explains the **ideas**. You write the actual `train.py` yourself.

---

## 0. The big picture

```
intents.yaml
     │  (parse + flatten)
     ▼
  X = [texts]          Y = [intent labels]      <- two aligned lists
     │
     │  TfidfVectorizer  (text -> numbers)
     ▼
  matrix of weights  (rows = sentences, cols = words)
     │
     │  LogisticRegression  (learns: weights -> label)
     ▼
  trained model
     │
     │  evaluate (accuracy, precision/recall, confusion matrix)
     │  persist (joblib)
     ▼
  intent_classifier.joblib   <- used later by the API
```

**One sentence:** turn text into numbers, let a model learn which number-pattern
maps to which intent, then measure how well it learned and save it.

---

## 1. From YAML to `X` and `Y`

The classifier learns a mapping:

```
text  ->  intent
```

- **`X`** = the list of all training texts (visitor utterances).
- **`Y`** = the matching intent label for each text (the YAML key it sat under).
- They must stay **aligned**: `X[i]` is the text, `Y[i]` is its label. `len(X) == len(Y)`.

`responses.yaml` is **not** used here — it's a lookup table used later, at serving
time. Training only needs `intents.yaml`.

> **Sanity checks after building them**
> - `len(X) == len(Y)` -> must be `True`
> - `len(X)` -> the total (sum of examples across all 10 intents), ~220
> - `set(Y)` -> exactly your 10 intent names, nothing more, nothing less

---

## 2. Why text must become numbers

ML models do math. They cannot read `"who is he"`. Every model input must be a
vector of numbers. So before any learning, text is **vectorized**: each sentence
becomes a row of numbers. That's the job of TF-IDF.

---

## 3. TF-IDF — the core idea

TF-IDF turns a sentence into a weighted vector. The weight of each word answers:
*how important is this word, here?* Two ingredients:

### TF — Term Frequency
How often a word appears **in this sentence**. More occurrences -> higher TF.
This is **per sentence**.

### IDF — Inverse Document Frequency
How **rare** a word is across **all** sentences (the whole corpus).
- Word in almost every sentence (`"what"`, `"is"`, `"his"`) -> **low** IDF.
- Rare, distinctive word (`"sjtu"`, `"discord"`, `"doctorate"`) -> **high** IDF.
This is **global** — one value per word, independent of any single sentence.

### TF-IDF = TF × IDF
A word scores high in a sentence when it appears **here** (high TF) **and** is
**rare overall** (high IDF). Common filler words get crushed; distinctive words
stand out.

> **The distinction that trips beginners**
> `v.idf_` gives the **IDF only** — one number per word (global rarity).
> The full **TF-IDF** lives in the matrix returned by `fit_transform` — per
> sentence. `idf_` is one ingredient; the matrix is the finished dish.

### The matrix
`fit_transform(X)` returns a matrix of shape **(n_sentences, n_words)**:

```
                "who"  "is"  "he"  "sjtu" ...
"who is he"   [ 0.62  0.51  0.59  0.00  ... ]   <- row = one sentence
"why sjtu"    [ 0.00  0.00  0.00  0.81  ... ]
   ...
```

- **Row** = one of your sentences, as a vector of word-weights.
- **Column** = one word from the learned vocabulary.
- **Cell** = the TF-IDF weight of that word in that sentence.

It's **sparse** (mostly zeros, since each sentence uses few words) — use
`.toarray()` to view it in full.

**Why this helps the classifier:** distinctive words (high weight) become strong,
reliable signals for an intent; noise words contribute almost nothing.

---

## 4. `TfidfVectorizer` — API reference

```
from sklearn.feature_extraction.text import TfidfVectorizer
```

### Methods
| Method | What it does |
|---|---|
| `fit_transform(X)` | Learns the vocabulary **and** returns the TF-IDF matrix. Use on training data. |
| `transform(X)` | Vectorizes new text using the **already learned** vocabulary. Use at predict time. |
| `get_feature_names_out()` | The list of words (columns), in column order. |

### Useful fitted attributes (note the trailing `_`)
| Attribute | What it holds |
|---|---|
| `vocabulary_` | dict `word -> column index`. (`vocabulary` without `_` is the input param = `None`.) |
| `idf_` | the IDF value of each word, indexed by column. |

> **Rule:** in scikit-learn, anything **learned during `.fit()` ends with `_`**
> (`vocabulary_`, `idf_`, `classes_`, `coef_`). No underscore = a setting you
> passed in, not a result.

### Key parameters (for tuning later)
| Param | Effect | For this project |
|---|---|---|
| `ngram_range=(1,2)` | use single words **and** word pairs | captures `"ai journey"` as one signal |
| `analyzer="char_wb"` | use character n-grams | robust to typos (`"helo"`, `"hii"`) |
| `min_df`, `max_df` | drop too-rare / too-frequent words | denoise the vocabulary |
| `lowercase=True` | lowercases text (default) | keep on |

---

## 5. The classifier — Logistic Regression (multiclass)

```
from sklearn.linear_model import LogisticRegression
```

You know binary logistic regression (2 classes). With 10 intents, scikit-learn
does **multinomial** logistic regression (softmax) automatically: for an input it
outputs a probability for **each** of the 10 intents and picks the highest.

### Key parameters
| Param | Why it matters here |
|---|---|
| `class_weight="balanced"` | your classes are uneven (`out_of_scope` has more examples). This stops the model from favoring big classes. |
| `C=1.0` | regularization strength (smaller = stronger = simpler model). Tune if overfitting. |
| `max_iter=1000` | raise it if you get a "did not converge" warning. |

Useful fitted attribute: `classes_` (the label order the model uses internally).

---

## 6. Pipeline — glue the vectorizer and classifier together

```
from sklearn.pipeline import Pipeline
```

A `Pipeline` chains steps into one object. Conceptually:

```
Pipeline([
    ("tfidf", TfidfVectorizer(...)),     # step 1: text -> numbers
    ("clf",   LogisticRegression(...)),  # step 2: numbers -> intent
])
```

Why it's worth it:
- **One object** does `.fit()`, `.predict()`, `.score()` end to end — you pass raw
  text in, it vectorizes then classifies.
- **No leakage:** the vectorizer is fit only on the training split, automatically.
- **One file to save:** `joblib.dump(pipeline, ...)` persists vectorizer + model
  together, so the API loads a single artifact.

> In your exploration notebook you call `fit_transform` by hand to *see* what TF-IDF
> does. In `train.py`, let the Pipeline handle it — don't vectorize manually.

---

## 7. `train_test_split` — measure honestly

```
from sklearn.model_selection import train_test_split
```

You can't judge a model on data it trained on (it could just memorize). So hold
out a slice it never sees during training, and test on that.

- `test_size=0.2` -> keep 20% for testing.
- `stratify=Y` -> **keep each intent's proportion** in both train and test. Critical
  with small, uneven classes — without it, a rare intent might vanish from one side.
- `random_state=42` -> makes the split reproducible (same split every run).

> **Caveat for you:** ~220 examples over 10 classes is small. A single 20% test set
> is noisy. That's why cross-validation (section 9) gives a more trustworthy number.

---

## 8. Evaluating — never trust accuracy alone

| Metric | Question it answers | Watch out |
|---|---|---|
| **Accuracy** | overall, what fraction is correct? | misleading on uneven classes — a lazy model can look "good". |
| **Precision** (per class) | when it predicts intent X, how often is it right? | low = too many false alarms for X. |
| **Recall** (per class) | of the real X cases, how many did it catch? | low = it misses real X. |
| **F1** (per class) | balance of precision and recall | the single number to compare per class. |

```
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
```

- **`classification_report(y_true, y_pred)`** -> precision/recall/F1 **per intent**.
  Always read it per class, not just the global score.
- **`confusion_matrix` / `ConfusionMatrixDisplay`** -> a grid showing which intents
  get **confused with which**. Off-diagonal cells = mistakes. This is your map for
  the 2nd dataset pass: if `current_focus` is often misread as `ai_journey`, you
  know exactly where to add/clarify examples.

---

## 9. Cross-validation — for small datasets

```
from sklearn.model_selection import cross_val_score
```

Instead of one train/test split, CV makes K splits (e.g. 5), trains and tests K
times, and averages. You get a **stable** estimate plus a sense of variance.

- `cv=5` -> 5 folds.
- `scoring="f1_macro"` -> average F1 across classes, treating each class equally
  (good for uneven data).
- Report it as `mean ± std` — the spread tells you how shaky the estimate is.

---

## 10. Persisting the model — `joblib`

```
import joblib
```

- `joblib.dump(pipeline, "models/intent_classifier.joblib")` -> save to disk.
- `joblib.load(path)` -> load it back (in the API) and call `.predict([...])`.

joblib is preferred over `pickle` for scikit-learn objects (handles big numpy
arrays efficiently). Save the **whole Pipeline**, not just the classifier — you
need the fitted vectorizer too, or new text can't be vectorized the same way.

---

## 11. The full mental flow (recap)

```
1. load intents.yaml          ->  nested dict
2. flatten                    ->  X (texts), Y (labels), aligned
3. train_test_split(stratify) ->  X_train/Y_train, X_test/Y_test
4. Pipeline(TfidfVectorizer, LogisticRegression)
5. pipeline.fit(X_train, Y_train)
6. y_pred = pipeline.predict(X_test)
7. classification_report + confusion_matrix(Y_test, y_pred)
8. cross_val_score(pipeline, X, Y, cv=5)   (optional, recommended)
9. joblib.dump(pipeline, "models/intent_classifier.joblib")
```

---

## 12. Beginner traps (ones you already met — keep them in mind)

| Trap | Fix |
|---|---|
| `from pathlib import path` | capital `Path` — Python is case-sensitive. |
| Cell shows no `[N]:` | it's a Markdown cell, not Code — convert with `Y`. |
| Edited a cell, nothing changed | re-run **that** cell; the kernel only knows what it executed. |
| `intents.items()` gives weird keys | the YAML has a root wrapper — use `intents["intents"].items()`. |
| `X.append(text_list)` | append each **text**, not the whole list — that's the inner loop. |
| `v.vocabulary` prints `None` | learned attribute needs the underscore: `v.vocabulary_`. |
| `idf_` "is the TF-IDF" | `idf_` is IDF only; full TF-IDF is the `fit_transform` matrix. |

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Feature** | one input column the model sees (here: one word's weight). |
| **Label / target** | the answer to predict (here: the intent, your `Y`). |
| **Vectorize** | turn text into a numeric vector. |
| **Corpus** | the whole collection of training texts (your `X`). |
| **Sparse matrix** | a matrix stored efficiently because it's mostly zeros. |
| **Fit** | learn parameters from data. |
| **Transform** | apply a learned transformation to data. |
| **Stratify** | preserve class proportions when splitting. |
| **Overfitting** | memorizing training data instead of generalizing. |
| **Regularization** | penalty that keeps the model simple (param `C`). |
