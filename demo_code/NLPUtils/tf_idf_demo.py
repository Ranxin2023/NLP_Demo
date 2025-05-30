from sklearn.feature_extraction.text import TfidfVectorizer
from contextlib import redirect_stdout
import pandas as pd
def professional_article_tf_idf():
    print("professional article tf demo...")

    # Sample article
    text = """Water is a chemical compound consisting of two hydrogen atoms and one oxygen atom.
    It is essential for all known forms of life. Water molecules are polar, which allows them to form hydrogen bonds.
    This property makes water an excellent solvent, especially for ionic and polar substances."""

    # Custom stopwords (domain-neutral terms)
    custom_stopwords = {
        'known', 'forms', 'life', 'essential', 'especially',
        'them', 'all', 'it', 'is', 'for', 'this', 'makes'
    }

    # Step 1: Initialize vectorizer
    default_vectorizer = TfidfVectorizer(stop_words='english')
    built_in_stopwords = set(default_vectorizer.get_stop_words())

    # Step 2: Combine built-in + custom stopwords
    all_stopwords = built_in_stopwords.union(custom_stopwords)

    # Step 3: Create new vectorizer using combined stopwords
    vectorizer = TfidfVectorizer(stop_words=list(all_stopwords), token_pattern=r'\b\w+\b')

    # TF-IDF transform
    X = vectorizer.fit_transform([text])
    tfidf_scores = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Sort and display
    sorted_scores = tfidf_scores.T.sort_values(by=0, ascending=False)
    print("📘 professional article tf demo")
    print("▼ Top TF-IDF Terms:\n")
    print(sorted_scores.head(15))



def tf_idf_implementation():
    print("tf idf implementation...")
    import math
    from collections import Counter

    # Sample documents
    docs = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat chased the dog"
    ]

    # Step 1: Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in docs]

    # Step 2: Calculate term frequency (TF)
    def compute_tf(doc):
        tf_dict = Counter(doc)
        total_terms = len(doc)
        return {term: count / total_terms for term, count in tf_dict.items()}

    # Step 3: Calculate inverse document frequency (IDF)
    def compute_idf(corpus):
        N = len(corpus)
        all_terms = set(term for doc in corpus for term in doc)
        idf_dict = {}
        for term in all_terms:
            containing_docs = sum(1 for doc in corpus if term in doc)
            idf_dict[term] = math.log(N / (1 + containing_docs)) + 1
        return idf_dict

    # Step 4: Calculate TF-IDF
    def compute_tfidf(tf, idf):
        return {term: tf[term] * idf[term] for term in tf}

    # Process all documents
    idf = compute_idf(tokenized_docs)
    tfidf_scores = [compute_tfidf(compute_tf(doc), idf) for doc in tokenized_docs]

    # Print TF-IDF for each document
    for i, scores in enumerate(tfidf_scores):
        print(f"Document {i+1} TF-IDF:")
        for term, score in scores.items():
            print(f"  {term}: {score:.4f}")
        print()

def scikit_learn_for_idf():
    print("scikit learn for idf")
    docs = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat chased the dog"
    ]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)

    # Get feature names and TF-IDF matrix
    features = vectorizer.get_feature_names_out()
    dense = X.todense()

    # Print TF-IDF scores
    for i, doc in enumerate(dense):
        print(f"Document {i+1} TF-IDF:")
        for word, score in zip(features, doc.tolist()[0]):
            if score > 0:
                print(f"  {word}: {score:.4f}")
        print()

def tf_idf():
    with open("./output_results/tf_idf_result.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            tf_idf_implementation()
            scikit_learn_for_idf()
            professional_article_tf_idf()
