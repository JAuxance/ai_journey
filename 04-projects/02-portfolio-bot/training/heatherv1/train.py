#!/usr/bin/env python3
"""Train a model to classify user intents for the Portfolio Bot.
This script loads training data, preprocesses it, and trains a logistic regression model.
The trained model is saved to disk for later use in the bot.
"""

import yaml
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib


DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
INTENT_PATH = DATA_DIR / "intents.yaml"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"

def load_data(path):
    """Load training data from a YAML file."""
    X = []
    Y = []
    intents = yaml.safe_load(path.read_text())
    for (key, text_list) in intents["intents"].items():
        for text in text_list:
            X.append(text)
            Y.append(key)
    return X, Y

def build_pipeline():
    """Build a machine learning pipeline for intent classification."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)),
    ])
    return pipeline

def main():
    # Load and preprocess data
    X, Y = load_data(INTENT_PATH)

    # Build and train the model
    pipeline = build_pipeline()
    pipeline.fit(X, Y)


    # Save the trained model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":    
    main()