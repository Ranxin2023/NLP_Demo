from gensim.models import word2vec
def word2vec_demo():
    sentences = [["this", "is", "a", "sample", "sentence"], ["word", "embeddings", "are", "useful"]]
    model = word2vec(sentences, vector_size=100, window=5, min_count=1, sg=1)  # sg=1: skip-gram, sg=0: CBOW
    vector = model.wv['sample']
    print(vector)