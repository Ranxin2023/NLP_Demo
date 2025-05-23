from demo_code.Word_Embeddings.BERT import BERT_demo
from demo_code.Word_Embeddings.GloVe_demo import Glove_demo
from demo_code.Word_Embeddings.word2vec import word2vec_demo
from contextlib import redirect_stdout, redirect_stderr
def word_embeddings_algorithms():
    with open("./output_results/word_embeddings_algorithms.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f), redirect_stderr(f):
            BERT_demo()
            Glove_demo()
            word2vec_demo()
