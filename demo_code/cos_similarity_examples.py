from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import redirect_stdout
def example1():
    print("example 1...")
    '''
    The sentences are somewhat related (fox, dog, brown, quick), so the score is moderate.
    '''
    sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown dog outpaces a fast fox.",
    ]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)

    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    print(f"Cosine Similarity: {similarity[0][0]:.4f}")

def example2():
    print("example2...")
    documents = [
    "I love deep learning and neural networks.",
    "Convolutional networks are a type of deep learning model.",
    "I play football every weekend with my friends."
    ]

    tfidf = TfidfVectorizer().fit_transform(documents)

    sim_0_1 = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    sim_0_2 = cosine_similarity(tfidf[0:1], tfidf[2:3])[0][0]

    print(f"Similarity between Doc 0 and 1 (same topic): {sim_0_1:.4f}")
    print(f"Similarity between Doc 0 and 2 (different topic): {sim_0_2:.4f}")
def example3():
    print("example3...")
    texts = [
        "Machine learning is awesome.",
        "Machine learning is awesome!",
        "Machine learning is very interesting."
    ]

    tfidf = TfidfVectorizer().fit_transform(texts)

    print("Sim(text 0, text 1):", cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    print("Sim(text 0, text 2):", cosine_similarity(tfidf[0:1], tfidf[2:3])[0][0])

def cos_sililarity_demo():
    with open("./output_results/cos_similarity_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            example1()
            example2()
            example3()
