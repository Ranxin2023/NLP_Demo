# NLPDEMO: Natural Language Processing Hands-On Project
## Intruduction
This project is a hands-on demo of various Natural Language Processing (NLP) techniques, organized in modular Python scripts. It walks through fundamental concepts such as preprocessing, normalization, text classification, lemmatization, different NLP model approaches, and more.

## Project Structure
```graphql
NLPMDEMO/
├── datasets/
│   └── text_preprocessing.html               # HTML resource for preprocessing
├── demo_code/                                # All demo Python scripts
│   ├── __pycache__/
│   ├── cos_similarity_examples.py            # Cosine similarity demo
│   ├── nlp_challenges.py                     # Challenges in NLP
│   ├── nlp_tasks.py                          # Common NLP tasks
│   ├── nltk_demo.py                          # NLTK-based parsing demo
│   ├── parsing_demo.py                       # Constituency & dependency parsing
│   ├── preprocessing.py                      # Text preprocessing pipeline
│   ├── text_normalization.py                 # Normalization: lowercase, stemming, etc.
│   ├── tf_idf_demo.py                        # TF-IDF demonstration
│   └── three_approaches.py                   # Top-down, bottom-up, constituency parsing
├── output_results/                           # Output files from the demos
│   ├── cos_similarity_demo.txt
│   ├── nlp_challenges.txt
│   ├── nlp_tasks.txt
│   ├── nltk_demo.txt
│   ├── parsing_result.txt
│   ├── text_normalization.txt
│   ├── text_preprocessing.txt
│   └── three_approach_result.txt
├── .gitignore                                # Git ignore file
├── main.py                                   # Optional entry point script
├── README.md                                 # Project overview
└── requirements.txt                          # Python dependencies


```

## NLP Concepts
### 1. What is NLP?
Natural Language Processing (NLP) is a subfield of Artificial Intelligence that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate natural language in a way that is meaningful. NLP encompasses tasks such as translation, sentiment analysis, question answering, and more.

### 2. What are the main challenges in NLP?

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

### 6. What are the differences between rule-based, statistical-based, and neural-based approaches in NLP?
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

### 7. What is the **Bag-of-Words (BoW)** Model? 
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

### 8. What are the different types of parsing in NLP?
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

### 9. What is TF-IDF in NLP?
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
### 🧾 Example:
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

🧮 TF-IDF Score = TF × IDF
Words that appear often in a document but rarely elsewhere get a high score.
## 🧩 Python Dependencies

Below are the main libraries required to run this NLP demo project:

| Package            | Version     | Description                                                                 |
|--------------------|-------------|-----------------------------------------------------------------------------|
| `nltk`             | 3.8.1       | Natural Language Toolkit for classic NLP tasks like tokenization and POS   |
| `spacy`            | 3.7.2       | Industrial-strength NLP library used with `benepar` for parsing            |
| `textblob`         | 0.17.1      | Simple NLP API for sentiment analysis and noun phrase extraction           |
| `beautifulsoup4`   | 4.12.3      | For parsing HTML/XML data                                                  |
| `transformers`     | 4.39.3      | Hugging Face Transformers library (e.g., BERT, GPT, RoBERTa)               |
| `torch`            | >=2.0.0     | PyTorch backend, required for transformer models                           |
| `benepar`          | 0.2.0       | Berkeley Neural Parser, used for constituency parsing with spaCy           |

### 10. Machine Learning Algorithms used in NLP
#### Naive Bayes
- **Type**: Probabilistic classifier based on Bayes' Theorem.
- How it works: Assumes features (like words) are independent. Calculates the probability that a given text belongs to a class (like spam or not spam).
- **Common Use**: Text classification, spam detection, sentiment analysis.
- **Pros**: Simple, fast, works well on small datasets.

#### Support Vector Machines (SVM)
- **Type**: Supervised learning algorithm.
- **How it works**: Finds a hyperplane that best separates data into classes in a high-dimensional space.
- **Common Use**: Text classification, NER (Named Entity Recognition), sentiment analysis.
- **Pros**: Works well for high-dimensional data (e.g., text vectors), effective even with limited samples.

#### Decision Trees
- **Type**: Rule-based supervised learning model.
- **How it works**: Builds a tree structure where each internal node is a decision on a feature, and each leaf node is a class label.
- **Common Use**: Information extraction, binary text classification.
- **Pros**: Interpretable, easy to visualize.


### 11. Transfomers
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
