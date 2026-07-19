#  Learning Notes: Word2Vec & Embeddings

*Source: StatQuest (Josh Starmer) & Personal Research*  
*Objective: Understand the mathematical mechanics behind transforming text into vectors.*

---

## 1. Core Principle: Embedding + Neural Network

To transform words into mathematics, we assign a **weight vector** (a list of numbers) to each word.

- **Step 1: Weighted Sum**  
  For a given phrase, we calculate the sum of the weights of each word.
- **Step 2: Activation Function**  
  We apply a function (like ReLU) to the result of this sum.  
  *Role: Introduce non-linearity so the network can learn complex patterns, not just straight lines.*
- **Step 3: Softmax**  
  At the very end of the network, we use the **Softmax** function. It transforms the raw scores into **percentages (probabilities)** that total 100%. The word with the highest percentage is the model's prediction.
- **Visualization**  
  Thanks to these weights, we can plot each word on a 2D or 3D graph. Words with similar meanings end up physically close to each other on this graph.

---

## 2. The Two Word2Vec Architectures

*(Note: What is sometimes loosely called "Bag of Words" in this context is actually CBOW, the first half of Word2Vec).*

- **CBOW (Continuous Bag of Words)**  
  We provide the context to guess the missing center word.  
  *Example*: Context = "I am ___ here". The model must guess the center word: **"not"**.
- **Skip-gram**  
  The exact opposite. We provide the center word to guess the surrounding context.  
  *Example*: Center word = **"not"**. The model must guess the neighboring words: "I", "am", "here".

---

## 3. Training & Optimization

- **Backpropagation**  
  This is the mechanism that adjusts the weights. If the model makes a mistake, the error is calculated and sent backward to slightly modify the weights. This makes semantically similar words closer together in the vector space.
- **The Scaling Problem**  
  In reality, we use hundreds of activation functions and train the model on billions of words (e.g., the entire Wikipedia). Calculating the probability for *all* 50,000+ words in the dictionary at every single step is far too slow.
- **The Solution: Negative Sampling**  
  Instead of updating the weights for the entire dictionary at each step, we ask the model to:
  1. **Maximize** the score of the **correct word** (e.g., "not").
  2. **Minimize** the score of a **few random words** (e.g., "car", "banana", "sky").  
  *Result*: Training is drastically accelerated while maintaining excellent prediction quality.

---

## 4. Code Example (Bridge to Practice)

How to use a modern embedding model (based on these principles, but improved, like Sentence-BERT) in Python:

```python
# 1. Import the library (install via: pip install sentence-transformers)
from sentence_transformers import SentenceTransformer

# 2. Load a lightweight pre-trained model (it already contains the optimized weights)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Transform words/phrases into mathematical vectors (embeddings)
words = ["king", "queen", "car"]
vectors = model.encode(words)

# 4. Calculate mathematical similarity (cosine similarity) between "king" (index 0) and "queen" (index 1)
similarity_king_queen = model.similarity(vectors[0], vectors[1])

# 5. Display the result (close to 1.0 = highly similar, close to 0.0 = not similar at all)
print(f"Similarity between 'king' and 'queen': {similarity_king_queen.item():.4f}")