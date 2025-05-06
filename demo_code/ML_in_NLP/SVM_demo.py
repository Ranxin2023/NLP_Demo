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
    # 4. Train SVM model
    clf = SVC(kernel='linear')
    clf.fit(X, labels)

    # 5. Support vectors
    print("🧠 Support vector indices (in training set):", clf.support_)
    print()

    # 6. Predict new input
    test_input = "NLP is bad"
    prediction = clf.predict(vectorizer.transform([test_input]))[0]
    label = "Positive" if prediction == 1 else "Negative"
    print(f"🔎 Prediction for \"{test_input}\": {label}")
