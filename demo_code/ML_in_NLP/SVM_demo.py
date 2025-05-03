from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
texts = ["I love NLP", "NLP is amazing", "I hate spam", "Spam is bad"]
labels = [1, 1, 0, 0]  # 1 = positive, 0 = negative
def SVM_demo():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    clf = SVC(kernel='linear')
    clf.fit(X, labels)

    print(clf.predict(vectorizer.transform(["NLP is bad"])))
