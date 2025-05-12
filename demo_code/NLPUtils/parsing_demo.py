import benepar
import spacy
from contextlib import redirect_stdout
def dependency_parsing():
    '''
    This part shows grammatical relationships between words using arrows and dependency labels.
    
    '''
    print("Dependency Parsing...")
    # Load English model
    nlp = spacy.load("en_core_web_sm")

    # Parse a sentence
    doc = nlp("The cat sat on the mat.")

    # Print dependencies
    for token in doc:
        print(f"{token.text:<10} → Head: {token.head.text:<10} | Dep: {token.dep_}")

def constituency_parsing():
    '''
    # Install required packages first:
    # pip install spacy benepar
    # python -m spacy download en_core_web_sm
    # python -m benepar.download_en3
    '''
    print("Constituency Parsing...")
    

    # First-time setup (only run once)
    benepar.download('benepar_en3')

    # Load spacy and benepar
    nlp = spacy.load("en_core_web_sm")
    if not nlp.has_pipe("benepar"):
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    # Parse sentence
    doc = nlp("The cat sat on the mat.")
    for sent in doc.sents:
        print(sent._.parse_string)

def top_down_bottom_up_parsing():
    print("Top Down Bottom Up Parsing...")
    import nltk

    # Define grammar
    grammar = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> DT N
    VP -> V PP
    PP -> P NP
    DT -> 'the'
    N -> 'cat' | 'mat'
    V -> 'sat'
    P -> 'on'
    """)

    # Sentence to parse
    sentence = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    print(f"Sentences: {' '.join(sentence)}")
    # Top-down parser
    print("Top-Down Parsing:")
    parser_td = nltk.RecursiveDescentParser(grammar)
    for tree in parser_td.parse(sentence):
        print(tree)

    # Bottom-up parser
    print("\nBottom-Up Parsing:")
    parser_bu = nltk.ShiftReduceParser(grammar)
    for tree in parser_bu.parse(sentence):
        print(tree)

    
def parsing_demo():
    with open("./output_results/parsing_result.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            dependency_parsing()
            constituency_parsing()
            top_down_bottom_up_parsing()