# Project 02 - Portfolio Bot ("Trail") — ML & Training

> The chatbot **is** the portfolio.

This folder is the **ML / training side** of Trail: dataset, training notebook,
and learning notes. The deployed code (FastAPI backend + HTML/CSS/JS frontend)
lives in a **separate portfolio repo** so this `ai-journey` repo stays focused
on the learning trail.

A small, handcrafted NLP chatbot that replaces the traditional portfolio website.
Visitors land on a chat interface and discover who I am, what I'm learning,
and what I've built — through conversation, not navigation.

## Status

- **V1 frozen.** F1 macro ≈ **0.629** (5-fold cross-validation)
- **Pipeline:** TF-IDF + LogisticRegression (scikit-learn), joblib-persisted
- **Dataset:** 225 utterances, 10 intents, all hand-written
- **Approach:** ML built by hand — no AI code assistant on the model side
- **Started:** 2026-05-11

## What's in this folder

```
02-portfolio-bot/
├── README.md
├── data/
│   ├── intents.yaml      training utterances (the X, the labels)
│   ├── responses.yaml    Trail's reply templates (3-6 per intent)
│   └── _inspiration/     external dataset samples for reference (gitignored)
├── training/
│   ├── learning-resources.md   videos + docs + study plan for train.py
│   ├── ml-concepts.md          concepts reference (TF-IDF, Pipeline, eval)
│   └── heatherv1/
│       └── heather-experiments.ipynb   full ML pipeline experiments
└── models/                joblib outputs from experiments (gitignored)
```

## Why this project

I am 48+ days into a self-directed AI journey targeting a top AI-focused
university. A conventional portfolio site would describe my work; this one
**demonstrates** it. The bot itself is the proof of work.

The project also gives me a long horizon to grow:

- **V1:** classical ML stack (TF-IDF + logistic regression) ← current
- **V2:** word embeddings (planned)
- **V3:** small PyTorch classifier (planned)
- **V4:** small local LLM for generation (long-term)

Each upgrade is built and measured here, then the artifact ships to the
production repo. The portfolio improves in parallel with the skills it
advertises.

## V1 scope - 10 intents

8 product intents + 2 social glue:

| # | Intent           | Purpose                                          |
|---|------------------|--------------------------------------------------|
| 1 | greet            | "hi", "hey", "yo"                                |
| 2 | about_him        | who I am, my background (present)                |
| 3 | current_focus    | what I am learning right now                     |
| 4 | ai_journey       | long-term goal, school target, motivation        |
| 5 | projects         | Manager, ai-journey, this bot                    |
| 6 | skills_tech      | languages, tools, what I can actually do         |
| 7 | contact          | email, GitHub, LinkedIn                          |
| 8 | out_of_scope     | anything outside the trained scope               |
| 9 | thanks           | "thanks", "cheers"                               |
|10 | goodbye          | "bye", "talk later"                              |

## Persona - Trail

- **Name:** Trail
- **Voice:** warm, friendly companion. Third person ("Auxance is…", "He's…")
- **Tone:** humble, curious, calmly confident. Never corporate.
- **Self-aware:** Trail knows it is a small ML model and says so when relevant.
- **Signature opener:** *"Hey, I'm Trail. I walk visitors through Auxance's journey. What would you like to know?"*

## How the dataset works (POV system)

The classifier learns `text → intent`. Inside the data:

- **`I` / `me`** → the visitor (asking questions)
- **`he` / `his`** → Auxance (subject of the conversation)
- **`you`** → Trail (the bot being addressed)

So a training utterance is what a visitor types **to Trail**, asking **about Auxance**.

`intents.yaml` = the X + y (visitor utterances + their intent label).
`responses.yaml` = the lookup table the policy uses after the classifier predicts.

## Decisions log

| Date       | Decision                                                  | Reason |
|------------|-----------------------------------------------------------|--------|
| 2026-05-11 | Ambition V1 = rules + intent classifier                   | Matches current skill (sklearn). Clear path to upgrade. |
| 2026-05-11 | English only for V1                                       | Cleaner training set, target audience. |
| 2026-05-11 | Text replies only (no tool calling)                       | Keeps the ML problem honest. |
| 2026-05-11 | The bot IS the portfolio (not a sidekick)                 | Show, don't tell. |
| 2026-05-11 | 10 intents (8 + thanks + goodbye)                         | Compact dataset, expressive enough. |
| 2026-05-11 | Persona "Trail", warm companion, third person             | Friendly without being cute. |
| 2026-05-29 | Split repo: ML stays in ai-journey, deployed code in portfolio repo | Keeps ai-journey learning-focused; portfolio repo owns deployment. |
| 2026-05-29 | V1 frozen at F1 macro 0.629                               | TF-IDF plateau on 225 examples / 10 classes. V2 needs embeddings. |

## Rules I hold myself to (from the journey's root README)

- Code clean and documented.
- Every project includes results and conclusions.
- I do not only train models: I explain them.
- Fundamentals over hype.
