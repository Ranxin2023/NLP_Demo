
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from demo_code.nlp_challenges import nlp_challenges
from demo_code.nlp_tasks import nlp_tasks
from demo_code.preprocessing import preprocessing_demo
from demo_code.text_normalization import text_normalizaion_demo
from demo_code.three_approaches import three_processing_approach
from demo_code.nltk_demo import nltk_demo
from demo_code.cos_similarity_examples import cos_sililarity_demo
from demo_code.parsing_demo import parsing_demo
from demo_code.bow_demo import bow_demo
from demo_code.tf_idf_demo import tf_idf
from demo_code.ML_algorithms import ML_algorithms
from demo_code.Word_Embeddings.word_embeddings import word_embedding_demo
from demo_code.Sequence_Labelling.tagging_demo import tagging_demo
from demo_code.Sequence_Labelling.semantic_role_labeling import SRL_demo
from demo_code.ML_in_NLP.transformer.Transfomer import transfomer_datacamp
from demo_code.Word_Embeddings.word_embedding_algorithms import word_embeddings_algorithms
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
    # preprocessing_demo()
    # text_normalizaion_demo()
    # three_processing_approach()
    # cos_sililarity_demo()
    # nltk_demo()
    # parsing_demo()
    # bow_demo()
    # tf_idf()
    # ML_algorithms()
    # word_embedding_demo()
    # tagging_demo()
    # SRL_demo()
    # transfomer_datacamp()
    word_embeddings_algorithms()
    
if __name__=='__main__':
    main()
