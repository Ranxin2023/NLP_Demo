# NLPDEMO: Natural Language Processing Hands-On Project
## Intruduction
This project is a hands-on demo of various Natural Language Processing (NLP) techniques, organized in modular Python scripts. It walks through fundamental concepts such as preprocessing, normalization, text classification, lemmatization, different NLP model approaches, and more.

## Project Structure
```graphql
NLPDEMO/
│
├── datasets/                      # Contains input HTML files
│   └── text_preprocessing.html
│
├── demo_code/                     # Source code for different NLP topics
│   ├── nlp_challenges.py
│   ├── nlp_tasks.py
│   ├── nltk_demo.py
│   ├── preprocessing.py
│   ├── text_normalization.py
│   └── three_approaches.py
│
├── output_results/               # Auto-generated result files
│   ├── nlp_challenges.txt
│   ├── nlp_tasks.txt
│   ├── text_normalization.txt
│   ├── text_preprocessing.txt
│   └── three_approach_result.txt
│
├── main.py                        # Main entry point to run modules
├── requirements.txt               # Project dependencies
└── README.md                      # You're here!

```

## NLP Concepts
1. What is NLP?
Natural Language Processing (NLP) is a subfield of Artificial Intelligence that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate natural language in a way that is meaningful. NLP encompasses tasks such as translation, sentiment analysis, question answering, and more.

2. What are the main challenges in NLP?

4. What are some common preprocessing techniques used in NLP?
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

6. What are the differences between rule-based, statistical-based, and neural-based approaches in NLP?
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

7. What is the **Bag-of-Words (BoW)** Model? 
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
- **Top-Down Parsing**:

8.  What are the different types of parsing in NLP?
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

```markdown
| **Word** | **Head** | **Relation**                        |
|----------|----------|-------------------------------------|
| The      | cat      | determiner (det)                    |
| cat      | sat      | nominal subject (nsubj)             |
| sat      | ROOT     | main verb                           |
| on       | sat      | preposition (prep)                  |
| the      | mat      | determiner                          |
| mat      | on       | object of preposition (pobj)        |

```
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

## Setup
```sh
pip install -r requirements.txt
python -m nltk.downloader all
python -m spacy download en_core_web_sm
```