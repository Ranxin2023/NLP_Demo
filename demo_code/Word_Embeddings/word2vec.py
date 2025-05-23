from gensim.models import Word2Vec

def word2vec_demo():
    print("word2vec demo......")
    sentences = [
    ["nlp", "is", "fun"],
    ["deep", "learning", "is", "a", "subset", "of", "machine", "learning"],
    ["word", "embeddings", "capture", "semantic", "meaning"]
    ]

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=2)

    # Define words to inspect
    target_words = ["nlp", "fun", "learning", "semantic", "machine", "embeddings"]

    # Print embeddings
    for word in target_words:
        print(f"\nEmbedding for '{word}':")
        print(word2vec_model.wv[word])
