from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
texts = ["I love NLP", "NLP is amazing", "I hate spam", "Spam is bad"]
labels = [1, 1, 0, 0]  # 1 = positive, 0 = negative
def SVM_demo():
    print("SVM demo....")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    # 2. Show vocabulary
    print("📘 Vocabulary (word -> index):")
    for word, idx in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
        print(f"{idx:2d}: {word}")
    print()
    # 3. Show TF-IDF matrix
    print("🧮 TF-IDF Matrix:")
    print(X.toarray())
    print()
    '''
    Each row = a sentence
    Each column = a word from the vocabulary
    Each value = TF-IDF score (importance of the word in the sentence)

    Example:
    Row 0 = "I love NLP"

    TF-IDF for "love" (index 4): 0.78528828

    TF-IDF for "nlp" (index 5): 0.6191303

    Other columns are 0 because those words didn't appear in this sentence.
    '''
    # 4. Train SVM model
    clf = SVC(kernel='linear')
    clf.fit(X, labels)
    '''
    These are the indices of the support vectors — the most important training points used by the SVM to define the decision boundary.

    All 4 training examples ended up as support vectors here, likely because the dataset is very small and each example is critical.
    '''
    # 5. Support vectors
    print("🧠 Support vector indices (in training set):", clf.support_)
    print()

    # 6. Predict new input
    test_input = "NLP is bad"
    prediction = clf.predict(vectorizer.transform([test_input]))[0]
    label = "Positive" if prediction == 1 else "Negative"
    print(f"🔎 Prediction for \"{test_input}\": {label}")
