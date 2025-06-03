from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def random_forest_demo():
    print("Random Forest Demo......")
    texts = ["I love NLP", "NLP is amazing", "I hate spam", "Spam is bad"]
    labels = [1, 1, 0, 0]
    # Step 1: Vectorize
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    # Step 2: Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X, labels)

    # Step 3: Test Prediction
    test_sentence = ["I love NLP and hate spam"]
    test_vector = vectorizer.transform(test_sentence)
    
    print("\n🧪 Test Vector:")
    print(test_vector.toarray())

    print("\n🔍 Individual Tree Predictions:")
    for i, tree in enumerate(rf_model.estimators_):
        pred = tree.predict(test_vector)[0]
        print(f"Tree {i+1}: {'Positive' if pred == 1 else 'Negative'}")

    final_pred = rf_model.predict(test_vector)[0]
    print(f"\n✅ Final Random Forest Prediction: {'Positive' if final_pred == 1 else 'Negative'}")

    # Step 4: Feature Importance
    importances = rf_model.feature_importances_
    print("\n📈 Feature Importances:")
    feature_names = vectorizer.get_feature_names_out()
    for name, score in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"{name}: {score:.4f}")
