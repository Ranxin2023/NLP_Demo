
import nltk
from nltk.tokenize import word_tokenize
from contextlib import redirect_stdout
from transformers import BertTokenizer
def tokenization_demo():
    with open("./output_results/tokenization_result.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # Download tokenizer data for nltk
            nltk.download('punkt')

            # Sample text
            text = "Tokenization is crucial for NLP tasks like translation, summarization, and classification."

            # 1. Word Tokenization (NLTK)
            word_tokens = word_tokenize(text)
            print("Word Tokenization:", word_tokens)

            # 2. Character Tokenization
            char_tokens = list(text)
            print("\nCharacter Tokenization:", char_tokens)

            # 3. Subword Tokenization (BERT Tokenizer)
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            subword_tokens = bert_tokenizer.tokenize(text)
            print("\nSubword Tokenization (BERT):", subword_tokens)