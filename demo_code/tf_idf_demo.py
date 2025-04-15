from sklearn.feature_extraction.text import TfidfVectorizer
def tf_idf_demo():
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