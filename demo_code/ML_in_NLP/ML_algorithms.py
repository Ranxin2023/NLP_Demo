from demo_code.ML_in_NLP.Naive_Bayes import Naive_Bayes_demo
from demo_code.ML_in_NLP.SVM_demo import SVM_demo
# from demo_code.Transformer.transformer_demo import transfomer_demo
from demo_code.ML_in_NLP.Decision_Tree import decision_tree_demo
from demo_code.ML_in_NLP.Random_Forest import random_forest_demo
from contextlib import redirect_stdout
def ML_algorithms():
    with open("./output_results/ML_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            Naive_Bayes_demo()
            SVM_demo()
            decision_tree_demo()
            # transfomer_demo()
            random_forest_demo()
