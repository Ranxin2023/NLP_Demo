from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
def Naive_Bayes_demo():
    print("Naive Bayes Demo.......")
    texts = ["I love NLP", "NLP is amazing", "I hate spam", "Spam is bad"]
    labels = [1, 1, 0, 0]  # 1 = positive, 0 = negative

    vec = CountVectorizer()
    X = vec.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)
    feature_names = vec.get_feature_names_out()
    for i, class_label in enumerate(model.classes_):
        print(f"\nClass {class_label} log probabilities:")
        for j, log_prob in enumerate(model.feature_log_prob_[i]):
            print(f"  {feature_names[j]}: {log_prob:.4f}")


    print(model.predict(vec.transform(["I love spam"])))  # Example prediction
