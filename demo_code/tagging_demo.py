import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy
from contextlib import redirect_stdout
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def rule_based_pos_tagging():
    sentence = "The quick brown fox jumps over the lazy dog"
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    print("Rule-based POS tagging (NLTK):")
    for word, tag in tagged:
        print(f"{word:<10} ➝ {tag}")

def neural_pos_tagging_spacy():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The quick brown fox jumps over the lazy dog")
    print("\nNeural POS tagging (spaCy):")
    for token in doc:
        print(f"{token.text:<10} ➝ {token.pos_}")

def tagging_demo():
    with open("./output_results/tag_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            rule_based_pos_tagging()
            neural_pos_tagging_spacy()