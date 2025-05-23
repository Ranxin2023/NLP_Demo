from gensim.models import word2vec

def word2vec_demo():
    sentences = [
    ["nlp", "is", "fun"],
    ["deep", "learning", "is", "a", "subset", "of", "machine", "learning"],
    ["word", "embeddings", "capture", "semantic", "meaning"]
    ]

    # Train Word2Vec model
    word2vec_model = word2vec(sentences, vector_size=50, window=3, min_count=1, workers=2)

    # Get embedding for a word
    print("Word2Vec Embedding for 'nlp':")
    print(word2vec_model.wv['nlp'])