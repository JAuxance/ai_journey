# Project 02 - Portfolio Bot ("Trail")

> The chatbot **is** the portfolio.

A small, handcrafted NLP chatbot that replaces the traditional portfolio website.
Visitors land on a chat interface and discover who I am, what I'm learning,
and what I've built — through conversation, not navigation.

## Status

- Phase: **V1 - in design**
- Started: 2026-05-11
- Approach: built by hand, no AI code assistant, learning-first.

## Why this project

I am 42 days into a self-directed AI journey targeting a top AI-focused
university. A conventional portfolio site would describe my work; this one
**demonstrates** it. The bot itself is the proof of work.

The project also gives me a long horizon to grow:

- V1: classical ML stack (TF-IDF + logistic regression).
- V2: word embeddings.
- V3: a small PyTorch model.

Each upgrade ships into the same live product. The portfolio improves
in parallel with the skills it advertises.

## Architecture (V1)

```
+------------------------------+
|  Frontend (HTML/CSS/JS)      |
|  full-page chat UI           |
|  quick-start chips on load   |
+--------------+---------------+
               | POST /chat
               v
+------------------------------+
|  Backend (FastAPI)           |
|  /chat   : message -> reply  |
|  /intents: list (debug)      |
|  /log    : low-confidence    |
+--------------+---------------+
               v
+------------------------------+
|  Intent classifier (sklearn) |
|  TF-IDF (word + char n-gram) |
|  LogisticRegression          |
|  joblib-serialized           |
|  confidence threshold        |
+--------------+---------------+
               v
+------------------------------+
|  Response policy             |
|  YAML: intent -> templates   |
|  random pick per call        |
+------------------------------+
```

All persistent artifacts (model, logs, datasets) live on disk. No external
database needed for V1.

## Stack

| Layer    | Choice                  | Why |
|----------|-------------------------|-----|
| Frontend | HTML + CSS + JS vanilla | Fundamentals over frameworks. Migrate to React later if needed. |
| Backend  | Python + FastAPI        | Same language as the ML stack, modern async, auto OpenAPI docs. |
| ML       | scikit-learn            | Already comfortable, transparent, fast to iterate. |
| Storage  | YAML + joblib + JSON logs | No DB needed. Everything is text and reviewable. |

## V1 scope - 10 intents

8 product intents + 2 social glue:

| # | Intent           | Purpose                                          |
|---|------------------|--------------------------------------------------|
| 1 | greet            | "hi", "hey", "yo"                                |
| 2 | about_him        | who I am, my background                          |
| 3 | current_focus    | what I am learning right now (Day, topic)        |
| 4 | ai_journey       | long-term goal, school target, motivation        |
| 5 | projects         | Manager, ai-journey, this bot                    |
| 6 | skills_tech      | languages, tools, what I can actually do         |
| 7 | contact          | email, GitHub, LinkedIn                          |
| 8 | out_of_scope     | anything outside the trained scope               |
| 9 | thanks           | "thanks", "cheers"                               |
|10 | goodbye          | "bye", "talk later"                              |

Target dataset size: ~30 examples per intent = ~300 total. Hand-written.

## Persona - Trail

- Name: **Trail**.
- Voice: warm, friendly companion. Third person ("Auxance is…", "He's…").
- Tone: humble, curious, calmly confident. Never corporate.
- Self-aware: Trail knows it is a small ML model and says so when relevant.
- Signature opener (placeholder): "Hey, I'm Trail. I walk visitors through Auxance's journey. What would you like to know?"

## Phase plan

| Phase | Goal                                                  | New concepts learned                          |
|-------|-------------------------------------------------------|-----------------------------------------------|
| 0     | Skeleton: FastAPI hello + static frontend + CORS + wire end-to-end | client/server, HTTP, JSON, async Python      |
| 1     | Dataset: write intents.yaml + responses.yaml          | annotation discipline, dataset balance        |
| 2     | ML pipeline: TF-IDF + LogReg, train + eval + persist  | text vectorization, evaluation, confusion matrix |
| 3     | Integration: load model in API, /chat route, confidence threshold, out_of_scope routing | model serving, threshold tuning   |
| 4     | Polish: /log endpoint, quick-start chips, fallback link | continuous improvement loop, UX of chat       |
| 5+    | Upgrades: word embeddings -> PyTorch classifier       | aligned with later phases of the AI roadmap   |

## Folder structure (target)

```
02-portfolio-bot/
|-- README.md
|-- backend/
|   |-- app/
|   |   |-- main.py          FastAPI app and routes
|   |   |-- classifier.py    model loading + predict
|   |   |-- policy.py        intent -> response selection
|   |   `-- schemas.py       Pydantic request/response
|   |-- data/
|   |   |-- intents.yaml     intent -> list of training utterances
|   |   `-- responses.yaml   intent -> list of reply templates
|   |-- training/
|   |   |-- train.py
|   |   |-- evaluate.py
|   |   `-- notebook.ipynb   exploration + plots
|   |-- models/              trained joblib artifacts (gitignored)
|   `-- logs/                low-confidence message logs (gitignored)
`-- frontend/
    |-- index.html
    |-- styles.css
    `-- chat.js
```

## Decisions log

| Date       | Decision                                          | Reason                                  |
|------------|---------------------------------------------------|-----------------------------------------|
| 2026-05-11 | Ambition V1 = rules + intent classifier           | Matches current skill (sklearn). Clear path to upgrade. |
| 2026-05-11 | English only for V1                               | Cleaner training set, target audience.  |
| 2026-05-11 | Text replies only (no tool calling)               | Keeps the ML problem honest.            |
| 2026-05-11 | The bot IS the portfolio (not a sidekick)         | Show, don't tell.                       |
| 2026-05-11 | 10 intents (8 + thanks + goodbye)                 | Compact dataset, expressive enough.     |
| 2026-05-11 | Persona "Trail", warm companion, third person     | Friendly without being cute.            |

## Rules I'm holding myself to (from the journey's root README)

- Code clean and documented.
- Every project includes results and conclusions.
- I do not only train models: I explain them.
- Fundamentals over hype.
