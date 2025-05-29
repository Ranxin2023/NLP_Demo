# NLPDEMO: Natural Language Processing Hands-On Project
## Table of Contents
- [Introduction](#intruduction)
- [Project Structure](#project-structure)
- [Concepts](#nlp-concepts):
    - [What is NLP](#1-what-is-nlp)
    - [What are the main challenges of NLP](#2-what-are-the-main-challenges-in-nlp)
    - [Tokenization in NLP](#3-tokenization-in-nlp)
    - [Preprocessing](#4-what-are-some-common-preprocessing-techniques-used-in-nlp)
    - [Part of Speech](#5-part-of-speech-pos-tagging-in-nlp)
    - [Three Approaches](#7-what-are-the-differences-between-rule-based-statistical-based-and-neural-based-approaches-in-nlp)
    - [NLTK Package](#8-nltk-natural-language-toolkit)
    - [Bag of Words](#9-what-is-the-bag-of-words-bow-model)
    - [Parsing in NLP](#10-what-are-the-different-types-of-parsing-in-nlp)
    - [TF IDF](#11-what-is-tf-idf-in-nlp)
    - [Machine Learning Algorithms](#12-machine-learning-algorithms-used-in-nlp)
    - [Transformer](#13-transfomers)
    - [Word Embeddings](#14-word-embeddings)
    - [BERT](#15-bert)
    - [OOV(out of words)](#17-oovout-of-vocabularywords)
    - [Machine Translation](#18-machine-translation)
    - [Sequence Labeling](#19-what-is-sequence-labeling)
- [Python Dependencies](#python-dependencies)
- [Setup](#setup)

## Intruduction
This project is a hands-on demo of various Natural Language Processing (NLP) techniques, organized in modular Python scripts. It walks through fundamental concepts such as preprocessing, normalization, text classification, lemmatization, different NLP model approaches, and more.

## Project Structure
```graphql
NLPDemo/
│
├── datasets/                          # (Optional) Datasets used in demos (not shown yet)
│
├── demo_code/
│   ├── GeneralNLP/                    # General NLP tasks and preprocessing
│   │   ├── nlp_challenges.py
│   │   ├── nlp_tasks.py
│   │   ├── preprocessing.py
│   │   └── three_approaches.py
│   │
│   ├── ML_in_NLP/                     # ML models used in NLP tasks
│   │   ├── transformer/
│   │   │   ├── datacamp_tutorial.py
│   │   │   ├── self_implementation.py
│   │   │   └── Transformer.py
│   │   ├── Decision_Tree.py
│   │   ├── ML_algorithms.py
│   │   ├── Naive_Bayes.py
│   │   └── SVM_demo.py
│   │
│   ├── NLPUtils/                      # Utility demos and examples
│   │   ├── bow_demo.py
│   │   ├── cos_similarity_examples.py
│   │   ├── nltk_demo.py
│   │   ├── parsing_demo.py
│   │   ├── text_normalization.py
│   │   └── tf_idf_demo.py
│   │
│   ├── Sequence_Labelling/           # Sequence labeling tasks
│   │   ├── chunking.py
│   │   ├── NER_demo.py
│   │   ├── semantic_role_labeling.py
│   │   └── tagging_demo.py
│   │
│   └── Word_Embeddings/              # Word embedding algorithm demos
│       ├── BERT.py
│       ├── GloVe_demo.py
│       ├── word2vec.py
│       ├── word_embeddings.py
│       └── word_embedding_algorithms.py
│
├── nlp_env/                          # Python virtual environment (ignored in Git)
│
├── output_results/                   # Text outputs and logs from all demos
│   ├── bow_demo.txt
│   ├── cos_similarity_demo.txt
│   ├── ML_demo.txt
│   ├── nlp_challenges.txt
│   ├── nlp_tasks.txt
│   ├── nltk_demo.txt
│   ├── parsing_result.txt
│   ├── SRL_demo.txt
│   ├── tag_demo.txt
│   ├── text_normalization.txt
│   ├── text_preprocessing.txt
│   ├── tf_idf_result.txt
│   ├── three_approach_result.txt
│   ├── transformer_datacamp.txt
│   ├── word_embeddings.txt
│   └── word_embeddings_algorithms.txt
│
├── main.py                           # Entry point to run selected demos
├── readme.md                         # Project documentation
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Files and folders ignored by Git


```

## NLP Concepts
### 1. What is NLP?
Natural Language Processing (NLP) is a subfield of Artificial Intelligence that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate natural language in a way that is meaningful. NLP encompasses tasks such as translation, sentiment analysis, question answering, and more.

### 2. What are the main challenges in NLP?
The complexity and variety of human language create numerous difficult problems for the study of Natural Language Processing (NLP). The primary challenges in NLP are as follows:
#### Challenge 1: **Semantics and Meaning**:
- Example: "He saw a bat in the dark."
    - **Why it’s hard**: The word **"bat"** can mean an **animal** or a **sports object**. Determining which meaning is correct requires understanding the **context**.
    - **What we do**: We use spaCy to analyze each token’s **part of speech (POS)** and lemma to explore how it processes such ambiguity.

#### Challenge 2: Ambiguity (Lexical & Syntactic)
- Example: "She couldn't bear the pain."

    - **Why it’s hard**: The word "bear" could mean endure or refer to the animal. Without context, it's ambiguous.
    - **What we do**: Use TextBlob to extract sentiment, showing how ambiguity can influence interpretation.
    - **Real-world impact**: Ambiguity can mislead **emotion analysis**, **translation**, and **recommendation systems**.

#### Challenge 3: Contextual Understanding (Coreference Resolution)
- Example: "Alex told Jordan that he would win."

    - **Why it’s hard**: Who does **"he"** refer to—Alex or Jordan? This is called a coreference problem.
    - **What we do**: Analyze dependency parsing with spaCy to show how it tracks grammatical relationships—but note that spaCy doesn't resolve coreferences by default.
    - **Real-world impact**: Coreference errors can break summarization, QA systems, or AI assistants.
        - Coreference occurs when multiple expressions (words or phrases) in a text refer to the same real-world entity.
        - 🔍 Simple Example:
        “Emma went to the store. She bought apples.”
        - Here, “She” refers to “Emma”. This is a coreference relationship.

#### Challenge 4: Language Diversity
- Example: `"मैं घर जा रहा हूँ।"` (Hindi for “I am going home.”)
    - **Why it’s hard**: Most NLP models are trained on English. When given non-English input, models often fail.
    - **What we do**: Feed a Hindi sentence to spaCy's English model and show it fails to assign meaningful POS tags.
    - **Real-world impact**: Lack of multilingual support limits accessibility of NLP tools for non-English speakers.

#### Challenge 5: Data Limitations and Bias
- Example: `"The nurse helped the doctor because she was tired."`
    - **Why it’s hard**: NLP systems often learn societal biases from training data (e.g., assuming "nurse" = female).
    - **What we do**: Show the dependency parse and discuss how gender pronouns often reinforce stereotypes in text understanding.
    - **Real-world impact**: Bias can lead to unethical outputs in tasks like resume screening, chatbots, and search.

### 3. Tokenization in NLP
#### 3.1   What is a Token?
A **token** is the smallest unit in a text that carries meaning. Depending on the level of tokenization, a token could be:
- A word (e.g., “tokenization”)

- A subword (e.g., “token”, “##ization” in BERT)

- A character (e.g., “t”, “o”, “k”, ...)
#### 3.2 types of token
| Type                       | Description                                                              | Example                                     |
| -------------------------- | ------------------------------------------------------------------------ | ------------------------------------------- |
| **Word Tokenization**      | Splits text by words, considering punctuation and spacing                | `"NLP is great"` → `["NLP", "is", "great"]` |
| **Subword Tokenization**   | Breaks rare or complex words into frequent sub-units (used in BERT, GPT) | `"tokenization"` → `["token", "##ization"]` |
| **Character Tokenization** | Breaks every character                                                   | `"NLP"` → `["N", "L", "P"]`                 |

### 4. What are some common preprocessing techniques used in NLP?
- Tokenization: Splitting text into words or phrases.

- Stop Word Removal: Removing common, less informative words.

- Text Normalization:

    - Lowercasing
    - Lemmatization
    - Stemming
    - Date/Time normalization

- Punctuation/Special Character Removal

- HTML Tag Removal

- Spell Correction

- Sentence Segmentation

### 5. Part-of-Speech (POS) Tagging in NLP
**Part-of-speech tagging** is the process of assigning grammatical categories (like noun, verb, adjective) to each word in a sentence. It’s essential for understanding sentence structure and meaning.

In this project, we demonstrated three types of POS tagging:
✅ **Rule-Based/Statistical Tagging (NLTK)**
We used NLTK's `pos_tag()` which combines handcrafted rules and statistical probabilities.
```python
from nltk import pos_tag, word_tokenize
pos_tag(word_tokenize("The quick brown fox jumps over the lazy dog"))
```

| **Word**  | **POS Tag** | **Meaning**             |
| --------- | ----------- | ----------------------- |
| The       |  DT         | Determiner              |
| quick     |  JJ         | Adjective               |
| brown     |  NN ❌     | Noun (should be ADJ)    |
| fox       |  NN         | Noun                    |
| jumps     |  VBZ        | Verb (3rd person sing.) |
| over      |  IN         | Preposition             |
| the       |  DT         | Determiner              |
| lazy      |  JJ         | Adjective               |
| dog       |  NN         | Noun                    |


📊 **Summary Comparison**:

| Model     | Strengths                      | Weaknesses                             |
| --------- | ------------------------------ | -------------------------------------- |
| **NLTK**  | Fast, simple, rule/statistical | Lacks deep context (e.g., “brown”)     |
| **spaCy** | Neural, context-aware          | Less accurate on very complex syntax   |
| **BERT**  | Deep context, phrase-sensitive | Slower, more complex, may merge tokens |


🔍 Issue: "brown" is misclassified as a noun (NN) instead of an adjective (JJ). This is a known limitation of rule-based systems that lack context understanding.
### 7. What are the differences between rule-based, statistical-based, and neural-based approaches in NLP?
- **Rule-Based Approach:**

    - Uses manually crafted linguistic rules.

    - Highly interpretable, but hard to scale.

    - Good for simple patterns (e.g., regex email extraction).

- **Statistical-Based Approach:**

    - Learns from statistical patterns in labeled data.
    - Example: Naive Bayes for sentiment classification.
    - Flexible and scalable with enough data.

- **Neural-Based Approach:**

    - Uses deep learning models like BERT, GPT.
    - Learns hierarchical language features automatically.
    - High performance, especially with large datasets.

### 8. NLTK: Natural Language Toolkit
#### 📦 What is NLTK?
NLTK stands for **Natural Language Toolkit**. It is a powerful, open-source Python library for Natural Language Processing (NLP). Developed at the University of Pennsylvania, NLTK provides tools and resources for working with human language data (text), making it one of the most widely used libraries in education and research.
- 📌 Official Website: https://www.nltk.org
- 🐍 Language: Python
- 👨‍🏫 Primary Audience: Students, educators, researchers, beginners
#### 📚 Educational Focus
NLTK is often called a "teaching toolkit" because:
- It includes **built-in corpora and lexical resources**, like WordNet.
- It comes with **detailed documentation and examples**.
- It’s ideal for **learning NLP concepts** without needing huge datasets or deep learning infrastructure.

#### 🛠️ What Can You Do with NLTK?
| Task                         | Description                                  |
| ---------------------------- | -------------------------------------------- |
| **Tokenization**             | Split text into words or sentences           |
| **Stemming & Lemmatization** | Reduce words to their root form              |
| **Part-of-Speech Tagging**   | Identify nouns, verbs, adjectives, etc.      |
| **Parsing & Syntax Trees**   | Understand grammar and structure             |
| **Named Entity Recognition** | Detect entities like names, dates, locations |
| **Corpora Access**           | Load sample datasets like Gutenberg, Brown   |
| **Language Modeling**        | Create n-gram models and generate text       |
| **Text Classification**      | Build Naive Bayes or other classifiers       |

#### ✅ When to Use NLTK
| Use NLTK If...                           | Consider Alternatives If...                          |
| ---------------------------------------- | ---------------------------------------------------- |
| You are **learning** NLP or linguistics  | You need **high performance** at scale (e.g., spaCy) |
| You want to **prototype quickly**        | You need **neural embeddings or transformers**       |
| You need access to **classic NLP tools** | You want **modern, deep-learning-based NLP**         |

### 9. What is the **Bag-of-Words (BoW)** Model? 
BoW is one of the simplest and most widely used techniques to convert text into numerical feature vectors.
How It Works (Step-by-Step):
- **Tokenization**:
Break the sentence into words (tokens).
Example:
```python
"I love apples." → ['I', 'love', 'apples']
```
- **Vocabulary Creation**:
Collect all unique words across all documents.
Example:
```python
["I love apples.", "I love mangoes too."] → Vocabulary: ['I', 'love', 'apples', 'mangoes', 'too']
```
- **Vectorization**:
Turn each document into a vector of word counts from the vocabulary.
Each dimension of the vector corresponds to a word in the vocabulary.
Example: 
```python
Doc1: "I love apples."   → [1, 1, 1, 0, 0]
Doc2: "I love mangoes too." → [1, 1, 0, 1, 1]

```
- **Why Use BoW?**:
BoW transforms human language into a form that machine learning models can understand.

### 10. What are the different types of parsing in NLP?
Constituency Parsing, Dependency Parsing, Top-down Parsing, Bottom-up Parsing
- **Constituency Parsing:**
Constituency parsing breaks down a sentence into nested phrases (or constituents) using a context-free grammar (CFG). The output is a tree where:
    - Leaf nodes = actual words
    - Internal nodes = grammatical phrases (like NP, VP)

```yaml
S
├── NP (Noun Phrase)
│   ├── DT: The
│   └── NN: cat
├── VP (Verb Phrase)
│   ├── VBD: sat
│   └── PP (Prepositional Phrase)
│       ├── IN: on
│       └── NP
│           ├── DT: the
│           └── NN: mat

```

- **Dependency Parsing:**
Dependency parsing builds a directed graph where:

Each word in a sentence depends on another word

It shows grammatical relations (like subject → verb, verb → object)


| **Word** | **Head** | **Relation**                        |
|----------|----------|-------------------------------------|
| The      | cat      | determiner (det)                    |
| cat      | sat      | nominal subject (nsubj)             |
| sat      | ROOT     | main verb                           |
| on       | sat      | preposition (prep)                  |
| the      | mat      | determiner                          |
| mat      | on       | object of preposition (pobj)        |

- **Top-Down Parsing**:
Top-down parsing starts at the root of the parse tree (S = Sentence) and recursively applies grammar rules to predict what should come next (i.e., expand S → NP + VP).
Starts at root → breaks into sub-parts recursively

Example:
```mathematica
S → NP VP
NP → Det N
VP → V PP
PP → P NP

```
```yaml
S
├── NP
│   ├── Det: The
│   └── N: cat
└── VP
    ├── V: sat
    └── PP
        ├── P: on
        └── NP
            ├── Det: the
            └── N: mat

```

### 11. What is TF-IDF in NLP?
**TF-IDF** stands for:

- TF: Term Frequency

- IDF: Inverse Document Frequency

TF-IDF is a way to numerically represent text data — used widely in Natural Language Processing (NLP) to determine how important a word is in a document relative to a collection (corpus) of documents.

#### What is Term Frequency (TF)?
TF measures how often a word appears in a document.
- Formula:
**TF(t, d)** = (Number of times term *t* appears in document *d*) / (Total number of terms in document *d*)

#### What is **Inverse Document Frequency (IDF)?**
IDF measures how unique or rare a word is across **all documents** in the corpus.
- Formula:
**IDF(t)** = log( *N* / (1 + *df(t)*) )
Where:
- *N* = Total number of documents  
- *df(t)* = Number of documents containing term *t*

#### 🧾 Example:
If "cat" appears in 2 out of 3 documents:
**IDF("cat")** = log( 3 / (1 + 2) ) = log(1) = **0**

> 🔸 So common words get **low IDF scores**, while rare ones get **high scores**.

#### 🧠 How Is TF-IDF Used in NLP?
TF-IDF is used to convert text into numbers that can be used for:

- 🧪 Text classification (e.g., spam detection)
- 🗂 Information retrieval (e.g., search engines)
- 📊 Clustering or topic modeling
- 🤖 Machine learning models that take numerical input

#### 🧮 TF-IDF Score = TF × IDF
Words that appear often in a document but rarely elsewhere get a high score.

#### 🧠 Intuition:
Words like "the" and "is" appear everywhere → low IDF → low TF-IDF.

Words like "machine", "neural" appear in some docs but not all → higher IDF → useful for distinguishing docs.

#### 🧮 TF-IDF Score = TF × IDF
Words that appear often in a document but rarely elsewhere get a high score.

### 12. Machine Learning Algorithms used in NLP

#### Naive Bayes
- **Type**: Probabilistic classifier based on Bayes' Theorem.
- How it works: Assumes features (like words) are independent. Calculates the probability that a given text belongs to a class (like spam or not spam).
- **Common Use**: Text classification, spam detection, sentiment analysis.
- **Pros**:
    - Simple to implement and interpret
    - Very fast even with large feature sets
    - Performs surprisingly well on small datasets
- **Cons**:
    - **Unrealistic independence assumption** — words in natural language are highly dependent on context.
    - Can be biased if a word never appeared in training for a class (though smoothing helps).

#### Support Vector Machines (SVM)
- **Type**: Supervised learning algorithm.
- **How it works**: Finds a hyperplane that best separates data into classes in a high-dimensional space.
- **Common Use**: Text classification, NER (Named Entity Recognition), sentiment analysis.
- **Pros**: 
    - Handles high-dimensional spaces well (like TF-IDF vectors).
    - Effective when the number of samples is small compared to features.
    - With kernel tricks, it can model complex boundaries.

- **Cons**:
    - **Not probabilistic** — only gives class decision, not confidence (unless calibrated).
    - Can be slow on very large datasets.
    - Hard to interpret directly (no clear tree or rules).

- **Based on the demo**:
    - Used `SVC` with a linear kernel and TfidfVectorizer.
    - All samples became support vectors — common when you have very few, diverse inputs.
    - The model predicted `NLP is bad` as positive, because `NLP` had stronger TF-IDF weights than `bad` and was associated with positive classes in training.

#### Decision Trees
- **Type**: Rule-based supervised learning model.
- **How it works**: Builds a tree structure where each internal node is a decision on a feature, and each leaf node is a class label.
- **Common Use**: Information extraction, binary text classification.
- **Pros**: 
    - Highly interpretable — you can visualize and understand decisions.
    - Can handle mixed feature types (text + numeric).
- **Cons**:
    - Prone to overfitting, especially on small datasets.
    - Less robust to small variations in data (a small change can alter the tree structure drastically).
- **Based on the demo**:
    - You trained a DecisionTreeClassifier using CountVectorizer.
    - Your model produced a rule like:

    ```cpp
        if "nlp" appears more than 0.5 → class 1 (positive)
        else → class 0 (negative)
    ```
    - Simple and interpretable, but fragile — e.g., `"Spam is amazing"` could confuse it since "spam" = negative but "amazing" = positive.


#### Random Forests
- **Type**: Ensemble learning (collection of decision trees).
- **How it works**: Builds multiple decision trees and averages their results to improve accuracy and reduce overfitting.
- **Common Use**: Named entity recognition, sentiment classification.
- **Pros**: More accurate and stable than a single decision tree.

#### Transfomer
- **Type**: Attention-based deep learning architecture.
- **How it works**: Uses self-attention to weigh the importance of each word in a sentence relative to others.
- **Common Use**: BERT, GPT, and similar models for classification, translation, Q&A, summarization.
- **Pros**: State-of-the-art results; captures long-range dependencies better than LSTMs/RNNs.

### 13. Transfomers
#### What is transformer in NLP
The Transformer is a deep learning architecture introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). It has become the foundation of modern NLP models like BERT, GPT, and RoBERTa.

Unlike RNNs or LSTMs, which process sequences token by token, Transformers use self-attention to analyze an entire sequence in parallel. This allows them to capture global context, making them faster and more effective for long sequences.
#### 🧠 How the Transformer Works (Simplified Overview):
1. **Token Embedding**
Each word is converted into a dense vector using an `nn.Embedding` layer.
2. **Positional Encoding**
Since Transformers don't have sequence order by default, positional encodings (sinusoidal functions) are added to the embeddings to represent the position of each token in the sentence.
3. **Self-Attention**
The self-attention mechanism is the core of the Transformer architecture. It allows the model to weigh the importance of different words in a sequence when encoding each word — enabling the model to understand context from surrounding tokens.

For a word `𝑞`, self-attention is computed as:
    Attention(q, k, v) = softmax((q · kᵀ) / √dₖ) · v

Where:
- `q`, `k`, `v`: query, key, and value vectors
- `dₖ`: dimension of keys (for scaling)
- The result is a weighted sum of value vectors, based on how relevant each word is to the query
4. **Multi-Head Attention**
Instead of computing a single attention score, multiple attention heads capture different relationships in parallel, and their results are concatenated and projected back to the embedding space.
5. **Feedforward Network (FFN)**
After self-attention, each token's vector goes through a fully connected feedforward layer with non-linearity.
6. **Residual Connections + Layer Normalization**
Each block includes skip connections (residuals) and layer normalization to stabilize training.


### 14. Word Embeddings
Word embeddings are **dense vector representations** of words in a continuous space, where semantically or syntactically similar words are placed closer together. Unlike one-hot encoding or BoW (Bag of Words), embeddings capture contextual relationships and meaning.

#### 🔧 How it Works (Demo Summary)
In this project, we used `gensim.downloader` to load the pre-trained word2vec-google-news-300 model and explored embeddings with:
```python
model = api.load("word2vec-google-news-300")
```
This model maps each word to a 300-dimensional vector trained on Google News.

#### ✅ What We Explored
- **Semantic Similarity:**:
    ```text
        Similarity between 'king' and 'queen': 0.6511
    ```
- **Syntactic Analogy**:
    ```text
    'king' - 'man' + 'woman' = 'queen' (score: 0.7118)
    ```
- **Nearest Neighbors**:
    ```text
        Words most similar to 'apple':
    apples, pear, fruit, berry, pears

    ```
- **Out-of-Vocabulary Detection:**:
    ```text
        datascience is OOV (Out-of-Vocabulary)
    ```
#### Several popular algorithms to generate word embeddings:
There are several widely used algorithms to generate word embeddings. These methods differ in how they capture the meaning and context of words:

1. **word2vec**: 
A shallow neural network that learns word associations based on local context in a corpus. It offers two architectures:
- **CBOW**(Continuous Bag of Words): Predicts a target word from its surrounding context.
- **Skip-gram**: Predicts surrounding words from a target word.
                These embeddings are **static** (same vector regardless of sentence).
2. **GloVe (Global Vectors)**:
Constructs word embeddings by factoring a word co-occurrence matrix across the entire corpus. It captures **global statistical information** and produces **static embeddings** like word2vec, but based on overall co-occurrence patterns.
- **FastText**: Extends word2vec by including subword (character n-gram) information to better handle rare and misspelled words.
- **ELMo**: Produces context-dependent embeddings using a bidirectional LSTM trained on a language modeling objective.
#### 🧠 Why Word Embeddings Matter
- They **capture meaning** more effectively than BoW or TF-IDF.

- Enable **vector operations** (e.g., analogies).

- Useful for downstream tasks: classification, translation, question answering, etc.    

#### 📦 Tools Used
- `gensim` for pretrained embedding loading.

- `contextlib.redirect_stdout` to write output to a file.

- `redirect_stderr` to silence progress bar clutter.

### 15. BERT
#### 15.1 🧠 What is BERT?
BERT (Bidirectional Encoder Representations from Transformers) is a **pretrained language model** developed by Google in 2018. 
It is based on the **Transformer encoder architecture** and is trained to understand the **context of a word in a sentence by looking both left and right (bidirectional)**.

#### 15.2 📌 Key Characteristics of BERT:
- 1. **Only uses the Transformer encoder** (not the decoder)
- 2. **Pretrained on a massive text corpus** (Wikipedia + BookCorpus)
- 3. **Fine-tunable** on downstream tasks: sentiment analysis, question answering, NER, etc.
- 4. Uses **[MASK]** token to train via **Masked Language Modeling (MLM)**

#### 15.3 ⚙️ How BERT Works – Step by Step
- 1. **Transformer Encoder Architecture**:
BERT uses only the **encoder part** of the Transformer (from the original "Attention is All You Need" paper). The encoder focuses on capturing contextual relationships between words in a sentence.
- 2. **Bidirectional Attention**:
Unlike earlier models like GPT (which are left-to-right), BERT reads text in **both directions**.
**Example**:
- Sentence: `"The bank raised the interest rate."`
    - Left-to-right model: sees “The bank raised the...”
    - BERT: sees both “The bank raised...” and “...the interest rate.” simultaneously.

This enables BERT to **understand context deeply**, especially for polysemous words (words with multiple meanings).
- 3. **Input Representation**:
    - Each input to BERT is a combination of:
        - **Token embeddings** (from WordPiece tokenizer)
        - **Segment embeddings** (to differentiate Sentence A from Sentence B)
        - **Position embeddings** (to capture order)

- 4. **Pretraining Tasks**:
    - BERT is pretrained on large corpora (like Wikipedia and BooksCorpus) using two unsupervised tasks:
    - **(a) Masked Language Modeling (MLM)**:
        - Randomly masks 15% of input tokens
        - Model must predict the masked words using context from both sides
    - **(b) Next Sentence Prediction (NSP)**:
        - Given a pair of sentences, predict if the second sentence logically follows the first.
        ```text
            Sentence A: "She opened the book."
            Sentence B: "It was her favorite novel." → Label: True

            Sentence B: "He went to the store." → Label: False
        ```
- 5. **Fine-tuning for Downstream Tasks**
    - Once pretrained, BERT can be fine-tuned by adding a task-specific layer on top (e.g., classification head) and training on a smaller dataset.

    - Example fine-tuning:

        - Add a classification head on top of [CLS] token's representation.

        - Fine-tune entire model + new head using task-specific labeled data.

#### 🔍 15.3 Summary Table
| Component         | Details                                                         |
| ----------------- | --------------------------------------------------------------- |
| Architecture      | Transformer Encoder (only)                                      |
| Context           | **Bidirectional**                                               |
| Tokenization      | WordPiece with `[CLS]`, `[SEP]`, and `[MASK]` tokens            |
| Pretraining Tasks | Masked Language Modeling (MLM), Next Sentence Prediction (NSP)  |
| Fine-tuning       | Add task-specific layers; train entire model                    |
| Strengths         | Deep context, state-of-the-art accuracy, easy transfer learning |
| Limitations       | Large size, expensive to train from scratch                     |

### 16. Gensim


### 17. OOV(Out-of-Vocabulary)Words
#### 🧠 What Are OOV Words?
#### ✅ Techniques to Handle OOV Words
- **Character-Level Models**:
    - **What It Is**: Character-level models represent text **one character at a time,** instead of by full words or subwords.
    - **How It Works**: 
        - Input: `"xenobot"` → `["x", "e", "n", "o", "b", "o", "t"]`
        - The model learns **character sequences** rather than whole-word meanings.
        - It can generalize to OOV words **if they contain familiar patterns**.
    - **❌ Limitations:**:
        - Slower to train and decode.
        - May lose long-range semantic information.
- **Subword Tokenization (BPE, WordPiece)**:
    - Instead of full words, text is broken into **frequent subword units** using algorithms like:
        - **Byte Pair Encoding** (BPE): Used in GPT
        - **WordPiece**: Used in BERT
        - **Unigram**: Used in SentencePiece and T5
    - **🛠️ How It Works:**
        - `"xenobot"` → `["x", "##eno", "##bot"]`
        - These subwords exist in the vocabulary even if the full word doesn't.
    - **🧠 Why It Helps:**:
        - Allows the model to handle **unknown or compound words**.
        - Prevents vocabulary from growing too large.
        - Most modern transformer models use this.
- **Unknown Token ([UNK])**:
    - **📌 What It Is**:
        - Use a special placeholder token like `[UNK]` to represent any OOV word.
    - **🛠️ How It Works:**: 
        - `"xenobot" → [UNK]`
        - During training or inference, the model substitutes OOVs with this token.
    - **🧠 Why It Helps**:
        - Simple and prevents errors/crashes.
        - Still allows the model to make a **best guess** using surrounding context.
    - **❌ Limitations**:
        - Loses all **semantic meaning** of the unknown word.
        - Can lead to poor or generic predictions.
- **External Knowledge**:
    - **📌 What It Is**: Use external sources like:
        - **WordNet**, **Wikipedia**, or knowledge graphs (e.g., DBpedia, ConceptNet)
        - Custom dictionaries or glossaries
    - **🛠️ How It Works:**:
        - When encountering `"neuralink"`, look up its meaning externally.
        - Integrate this information into model inference.
    - **🧠 Why It Helps:**:
        - Useful in domains like **biomedicine**, **law**, or **education**, where models often miss terms.
        - Supports **explainability** and human-in-the-loop systems.
    - **❌ Limitations**:
        - Requires access to a live or pre-built knowledge source.
        - Integration with models can be complex.
- **Fine-Tuning with OOV Data**:
    - **What It Is**: Retrain or fine-tune a pre-trained model with **examples that contain OOV words**, so it learns their usage.
    - **🛠️ How It Works**:
        - Use domain-specific text with new vocabulary.
        - Fine-tune a language model like BERT or GPT on this data.
    - **🧠 Why It Helps:**:
        - The model **learns the context and meaning** of new words.
        - Becomes more specialized and accurate for the domain.
    - **✅ Real-World Use Cases:**:
        - **Medical NLP** (new drugs, diseases)
        - **Tech support bots** (new products)
        - **ChatGPT fine-tuned for company-specific terminology**

### 18. Machine Translation
Machine translation is the process of automatically translating text or speech from one language to another using a computer or machine learning model.

There are three techniques for machine translation:
#### 1. Rule-Based Machine Translation (RBMT)
- 📌 Concept:
RBMT systems translate text based on **manually defined linguistic rules** and **bilingual dictionaries**. It operates by analyzing the grammatical structure of the source sentence and applying syntactic and semantic rules to generate a grammatically correct target sentence.
-  In Practice:
```python
# Dictionary-based simulation
dictionary = {"hello": "bonjour", "world": "monde"}
``` 
####  2. Statistical Machine Translation (SMT)
- 📌 Concept:
SMT systems translate by using probability and statistics derived from large aligned bilingual corpora (e.g., English-French sentence pairs). These systems learn patterns based on how often words or phrases co-occur.

- 🧠**How it works**:
    - Align source and target sentences
    - Learn probability distributions of phrase translations
    - Use language models to ensure fluency
    - Translation = most probable target sentence given source

- 💡**Key Concepts**:
    - Phrase-based SMT: Translates phrases rather than individual words
    - Language Model (LM): Helps ensure fluent output (e.g., “a red car” is more likely than “a car red”)

- ✅ **Pros**:
    - Data-driven, adaptable to new languages with corpora
    - Outperforms RBMT in general cases

- ❌ **Cons**:
    - Still **rule-blind** (just statistical co-occurrence)
    - Needs huge datasets
    - Outputs often lack fluency or naturalness
    - Cannot easily handle long-range dependencies

#### 🧠 3. Neural Machine Translation (NMT)
- 📌 **Concept**:
NMT uses **deep learning models**, particularly **sequence-to-sequence architectures** and **Transformers**, to learn to translate from large datasets. It captures meaning and context more holistically.
- 🧠 **How it works**:
    - Input sentence → tokenized (e.g., subwords)
    - Encoder processes sentence → creates vector representation
    - Decoder generates translation token-by-token
    - Attention mechanism allows model to focus on relevant parts of input during decoding
- 🔬 **Key Tech**:
    - **Transformer** architecture (BERT, GPT, T5, MarianMT)
    - **Word embeddings** to capture meaning
    - **Attention** to focus on the right parts of input
- 🔧 **Tools**:
    - Hugging Face Transformers (e.g., `Helsinki-NLP/opus-mt-en-fr`)
    - Fairseq, OpenNMT, Tensor2Tensor
- ✅ **Pros**:
    - Best-in-class fluency and accuracy

    - Captures long-distance dependencies

    - Learns language patterns automatically
- ❌ **Cons**:
    - Requires a lot of data and compute to train

    - Can be opaque (not easily interpretable)

    - Sometimes hallucinates (outputs fluent but incorrect translations)

### 19. What is Sequence Labeling?
#### 2. Named Entity Recognition (NER)
Recognizes and classifies **named entities** (real-world objects) into predefined categories:
#### 3. Chunking (Shallow Parsing)
Groups words into syntactic "chunks" like **noun phrases (NP)** or **verb phrases (VP)**.
🔹 **Purpose**: Captures structure between words (e.g., “The quick brown fox” = NP).
   
## 🧩 Python Dependencies

Below are the main libraries required to run this NLP demo project:

| Package            | Version    | Description                                                                   |
|--------------------|------------|-------------------------------------------------------------------------------|
| `nltk`             |  3.8.1     | Natural Language Toolkit for classic NLP tasks like tokenization and POS      |
| `spacy`            |  3.7.2     | Industrial-strength NLP library used with `benepar` for parsing               |
| `textblob`         |  0.17.1    | Simple NLP API for sentiment analysis and noun phrase extraction              |
| `beautifulsoup4`   |  4.12.3    | For parsing HTML/XML data                                                     |
| `transformers`     |  4.39.3    | Hugging Face Transformers library (e.g., BERT, GPT, RoBERTa)                  |
| `torch`            |  >=2.0.0   | PyTorch backend, required for transformer models                              |
| `benepar`          |  0.2.0     | Berkeley Neural Parser, used for constituency parsing with spaCy              |
| `scikit-learn	`    |  1.4.2     | Machine learning toolkit; used here for `TfidfVectorizer` and classifiers     |
| `pandas`           |  2.2.2     | Data analysis and manipulation; used for displaying TF-IDF results            |
| `gensim`           |  4.3.2     | Topic modeling and word embeddings (Word2Vec, FastText, GloVe via downloader) |
| `scipy`            |	1.12.0    | Scientific computing; used for similarity metrics and matrix operations       |
| `smart_open`       |	>=6.3.0	  | Streams pretrained GloVe and other large files in gensim                      |

## Setup
1. clone the repository
```sh
git clone https://github.com/Ranxin2023/NLP_Demo
```
2. install the requirement
```sh
pip install -r requirements.txt

```
3. Open a python shell
```sh
python -m nltk.downloader all
python -m spacy download en_core_web_sm
python -m benepar.download_en3
``` 
4. Open main.py, choose the demo function you want to run, then run
```sh
python main.py
```