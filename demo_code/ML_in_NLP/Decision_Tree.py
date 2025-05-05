from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction.text import CountVectorizer
def decision_tree_demo():
    print("Decision Tree Demo......")
    texts = ["I love NLP", "NLP is amazing", "I hate spam", "Spam is bad"]
    labels = [1, 1, 0, 0]  # 1 = positive, 0 = negative

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    # Show transformed vectors
    '''
    vectorizer.vocabulary_
    | Word    | Index |
    | ------- | ----- |
    | amazing | 0     |
    | hate    | 1     |
    | is      | 2     |
    | love    | 3     |
    | nlp     | 4     |
    | spam    | 5     |
    | bad     | 6     |

    '''
    print("🧮 Vectorized input (bag-of-words counts):")
    print(X.toarray(), "\n")  # convert sparse matrix to dense
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X, labels)
     # Show decision tree structure
    print("🌳 Decision Tree Rules:")
    '''
    Tree rules are:
    |--- nlp <= 0.50
    |   |--- class: 0
    |--- nlp > 0.50
    |   |--- class: 1
    This means:

    If the word "nlp" does not appear in the sentence (nlp <= 0.5 → count = 0) → classify as class 0 (negative).

    If "nlp" does appear (nlp > 0.5) → classify as class 1 (positive).

    So the decision tree is only splitting based on the presence of the word “nlp” — because that was enough to separate your small training set!

    '''
    feature_names = vectorizer.get_feature_names_out()
    tree_rules = export_text(tree_model, feature_names=list(feature_names))
    print(f"Tree rules are:\n{tree_rules}")
    print(tree_model.predict(vectorizer.transform(["Spam is amazing"])))
