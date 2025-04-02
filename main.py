
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from demo_code.nlp_challenges import nlp_challenges
from demo_code.nlp_tasks import nlp_tasks
from demo_code.preprocessing import preprocessing_demo
def tokenization_demo():
    # Download required NLTK data files (only need to do this once)
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    # Sample sentence
    text = "Natural Language Processing enables machines to understand human language."

    # Tokenize the sentence into words
    tokens = word_tokenize(text)

    # Get Part-of-Speech tags
    pos_tags = pos_tag(tokens)

    print("Tokens:", tokens)
    print("POS Tags:", pos_tags)

def main():
    # tokenization_demo()
    # nlp_challenges()
    # nlp_tasks()
    preprocessing_demo()
if __name__=='__main__':
    main()
