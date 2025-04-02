
def rule_based():
    import re

    text = "Contact us at support@example.com or hr@example.org for help."

    emails = re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
    print("Rule-Based Extracted Emails:", emails)

def statistical_based():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    X_train = ["I love this product", "This is terrible", "Absolutely fantastic", "Worst experience ever"]
    y_train = ["positive", "negative", "positive", "negative"]

    vectorizer = CountVectorizer()
    X_vect = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_vect, y_train)

    test = ["I hate it"]
    test_vect = vectorizer.transform(test)
    prediction = model.predict(test_vect)

    print("Statistical Prediction:", prediction[0])

def neural_based_approach():
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    result = classifier("This product is unbelievably good!")
    print("Neural-Based Prediction:", result)

def three_processing_approach():
    rule_based()
    statistical_based()
    neural_based_approach()
