from contextlib import redirect_stdout
import warnings
warnings.filterwarnings("ignore")

def rule_based():
    import re

    text = "Contact us at support@example.com or hr@example.org myname@gmail.com for help."

    emails = re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
    print("Rule-Based Extracted Emails:", emails)

def statistical_based():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    X_train = [
    "I love this product",
    "I hate bad pizza",          
    "This is terrible",
    "Absolutely fantastic",
    "Worst experience ever",
    "Great service and food",       
    "I love pizza",    
    "Great pizza and staff",          
    "Delicious and amazing",        
    "Bad taste and quality"         
    ]
    y_train = ["positive", "negative", "negative", "positive", "negative", "positive", "positive", "positive","positive", "negative"]


    '''
    
    CountVectorizer is a class from scikit-learn (sklearn.feature_extraction.text) that converts a collection of text documents into a matrix of token counts.

    In other words:

    It transforms raw text into a numerical format that a machine learning model can understand.
    '''
    # Convert text documents to a matrix of token counts
    vectorizer = CountVectorizer()
    X_vect = vectorizer.fit_transform(X_train)

    # Create a Naive Bayes classifier and train it on the vectorized data
    '''
    MultinomialNB is a version of the Naive Bayes classifier that is particularly suited for discrete features 
    — such as word counts or term frequencies from documents.
    Why “Multinomial”?
    It models the distribution of word counts across documents:

    If you turn text into a bag-of-words count vector, MultinomialNB estimates the probability of each class (label) based on how many times words appear in that class.

    For example:

    “I love it” → [love: 1, hate: 0]

    “I hate it” → [love: 0, hate: 1]
    '''
    model = MultinomialNB()
    model.fit(X_vect, y_train)
    # New test sentence to classify
    test = ["I hate it", "It's great pizza!"]
    # Transform the test sentence using the same vectorizer
    test_vect = vectorizer.transform(test)
    # Predict the sentiment using the trained model
    prediction = model.predict(test_vect)

    print(f"Statistical Prediction:{prediction[0]} {prediction[1]}")

def neural_based_approach():
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = classifier("This product is unbelievably good!")
    print("Neural-Based Prediction:", result)

def three_processing_approach():
    with open("./output_results/three_approach_result.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            rule_based()
            statistical_based()
            neural_based_approach()
