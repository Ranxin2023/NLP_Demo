
import nltk
from nltk.tokenize import word_tokenize
from contextlib import redirect_stdout
from transformers import BertTokenizer
def tokenization_demo():
    with open("./output_results/tokenization_result.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # Download tokenizer data for nltk
            nltk.download('punkt')

            examples = {
                "Simple sentence": "Tokenization is crucial for NLP.",
                "Contractions": "You're going to love it, isn't it?",
                "Punctuation": "Hello!!! How are you?? -- I'm fine...",
                "Numbers & symbols": "The price is $9.99, not $10!",
                "Hyphenated words": "State-of-the-art NLP models are powerful.",
                "Rare compound": "Xenobot is a synthetic, programmable organism.",
                "Mixed language": "我喜欢 using NLP tools like spaCy or BERT."
            }

            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            for title, text in examples.items():
                print(f"\n--- {title} ---")
                print("Original Text:", text)

                # 1. Word Tokenization (NLTK)
                word_tokens = word_tokenize(text)
                print("Word Tokens:", word_tokens)

                # 2. Character Tokenization
                char_tokens = list(text)
                print("Character Tokens:", char_tokens)

                # 3. Subword Tokenization (BERT)
                subword_tokens = bert_tokenizer.tokenize(text)
                print("Subword Tokens (BERT):", subword_tokens)