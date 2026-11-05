# Projects

Applied projects. Each one goes through a complete ML workflow: data exploration, feature engineering, modeling, evaluation, and interpretation. The status of each project is stated at the top of its README.

---

## Projects

### [Student Performance Predictor](./student-performance-predictor/)

**Status**: Done.

**Goal**: Predict student exam scores from 19 tabular features (study habits, environment, background).

**Approach**: LinearRegression baseline with two encoding strategies (one-hot vs. mixed ordinal encoding).

**Results**: R² ≈ 0.73. Encoding choice has minimal impact; linear structure dominates.

 [Full details →](./student-performance-predictor/README.md)

---

### [Portfolio Bot "Trail"](./portfolio-bot-trail/)

**Status**: **PAUSED until January 2027** (focus is on `math/` and `ml-from-scratch/` until the end of December 2026).

**Goal**: Interactive chatbot serving as my portfolio. Visitors chat with a hand-coded model to discover my work.

**Current state**:
- **V1 (frozen)**: TF-IDF + Logistic Regression (F1 macro: 0.629, 5-fold CV)
- **V2 (paused)**: Sentence embeddings (target F1 > 0.80). Only the first exploration notebooks exist.

 [Full details →](./portfolio-bot-trail/README.md)