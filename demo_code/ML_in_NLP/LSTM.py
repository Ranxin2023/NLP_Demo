import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
def lstm_demo():
    print("LSTM demo......")
    texts = ["I love NLP", "NLP is amazing", "I hate spam", "Spam is bad"]
    labels = [1, 1, 0, 0]  # 1 = positive, 0 = negative

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    # 3. Model Definition
    print("📐 Defining LSTM Model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=50, output_dim=8, input_length=X.shape[1]),
        tf.keras.layers.LSTM(8),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()  # Shows the model architecture
    print()

    # 4. Compilation
    print("⚙️ Compiling Model...\n")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 5. Training
    print("🚀 Training Model...\n")
    model.fit(X.toarray(), labels, epochs=10, verbose=2)

    # 6. Prediction Example
    test_text = ["Spam is amazing"]
    X_test = vectorizer.transform(test_text)
    prediction = model.predict(X_test.toarray())
    print("\n🔍 Testing on sentence:", test_text[0])
    print(f"✅ LSTM Prediction: {'Positive' if prediction[0][0] > 0.5 else 'Negative'} (Confidence: {prediction[0][0]:.4f})")