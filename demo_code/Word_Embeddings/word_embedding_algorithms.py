from demo_code.Word_Embeddings.BERT import BERT_demo
from contextlib import redirect_stdout
def word_embeddings_algorithms():
    with open("./output_results/word_embeddings_algorithms.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            BERT_demo()