from contextlib import redirect_stdout
from sklearn.feature_extraction.text import CountVectorizer


def manual_bow_vectorization():
    import re
    doc1 = "I love apples."
    doc2 = "I love mangoes too."
    # Step 1: Tokenization
    # ['i', 'love', 'apples']
    tokens1 = re.findall(r'\b\w+\b', doc1.lower())  
    # ['i', 'love', 'mangoes', 'too']
    tokens2 = re.findall(r'\b\w+\b', doc2.lower())  

    # Step 2: Vocabulary Creation
    # ['apples', 'i', 'love', 'mangoes', 'too']
    vocab = sorted(set(tokens1 + tokens2))  

    # Step 3: Vectorization
    def vectorize(tokens, vocab):
        return [tokens.count(word) for word in vocab]

    vec1 = vectorize(tokens1, vocab)
    vec2 = vectorize(tokens2, vocab)

    print("Vocabulary:", vocab)
    print("Document 1 Vector:", vec1)
    print("Document 2 Vector:", vec2)

def count_vectorizer_example1():
    
    print("count vectorizer example 1...")
    docs = [
        "I love apples.",
        "I love mangoes too."
    ]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)

    # Convert to array
    print("Vocabulary:", vectorizer.get_feature_names_out())
    print("Vectors:\n", X.toarray())

def count_vectorizer_example2():
    print("count vectorizer example 2...")

    docs = [
        "I love NLP",
        "NLP is awesome",
        "Data science is fun",
        "I love data",
        "Data is powerful"
    ]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)

    # Show vocabulary and matrix
    print("Vocabulary:", vectorizer.get_feature_names_out())
    print("BoW Vectors:\n", X.toarray())

def bow_demo():
    with open("./output_results/bow_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            manual_bow_vectorization()
            count_vectorizer_example1()
            count_vectorizer_example2()