# from allennlp.predictors.predictor import Predictor
# import allennlp_models.tagging
'''
run these pieces of code first:
pip install wheel setuptools --upgrade
pip install h5py matplotlib
create a virtual environment:
python -m venv nlp_env
nlp_env\Scripts\activate  # On Windows

'''

import spacy
from contextlib import redirect_stdout
def SRL_demo():
    with open("./output_results/SRL_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("SRL demo......")
            nlp = spacy.load("en_core_web_sm")
            sentence = "John gave Mary a book on her birthday."
            doc = nlp(sentence)

            print("\nSyntactic Dependencies:")
            for token in doc:
                print(f"{token.text:10} {token.dep_:10} {token.head.text}")
