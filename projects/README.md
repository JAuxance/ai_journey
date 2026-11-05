# 4 Models Released

This folder contains finished, production-ready projects. Each project demonstrates a complete ML workflow: data exploration, feature engineering, modeling, evaluation, and interpretation.

---

## Projects

### [01 - Student Performance Predictor](./01-student-performance-predictor/)

**Goal**: Predict student exam scores from 19 tabular features (study habits, environment, background).

**Approach**: LinearRegression baseline with two encoding strategies (one-hot vs. mixed ordinal encoding).

**Results**: R² ≈ 0.73. Encoding choice has minimal impact; linear structure dominates.

 [Full details →](./01-student-performance-predictor/README.md)

---

### [02 - Portfolio Bot "Trail"](./02-portfolio-bot/)

**Goal**: Interactive chatbot serving as my portfolio. Visitors chat with a hand-coded model to discover my work.

**Current state**:
- **V1 (Prod)**: TF-IDF + Logistic Regression (F1: 0.629) 
- **V2 (WIP)**: Upgrading to Sentence Embeddings (Target F1 > 0.80) 

 [Full details →](./02-portfolio-bot/README.md)