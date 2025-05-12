import spacy
from textblob import TextBlob
from contextlib import redirect_stdout
def nlp_challenges():
    with open("./output_results/nlp_challenges.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # Load spaCy English model
            nlp = spacy.load("en_core_web_sm")

            # Challenge 1: Semantics and Meaning
            text_semantics = "He saw a bat in the dark."
            doc = nlp(text_semantics)
            print("Semantics Challenge Example:")
            for token in doc:
                print(token.text, "-", token.pos_, "-", token.lemma_)

            # Challenge 2: Ambiguity
            text_ambiguity = "She couldn't bear the pain."
            blob = TextBlob(text_ambiguity)
            print("\nAmbiguity Challenge (Sentiment):", blob.sentiment)

            # Challenge 3: Contextual Understanding
            text_context = "Alex told Jordan that he would win."
            doc_context = nlp(text_context)
            print("\nContext Challenge (Coreference not handled by spaCy by default):")
            for token in doc_context:
                print(token.text, "-", token.dep_, "-", token.head.text)

            # Challenge 4: Language Diversity
            text_hindi = "मैं घर जा रहा हूँ।"
            doc_hindi = nlp(text_hindi)
            print("\nLanguage Diversity (English model struggles with Hindi):")
            for token in doc_hindi:
                print(token.text, "-", token.pos_)

            # Challenge 5: Data Limitations and Bias
            example_biased = "The nurse helped the doctor because she was tired."
            doc_biased = nlp(example_biased)
            print("\nBias Challenge (Gender bias in pronoun resolution):")
            for token in doc_biased:
                print(token.text, "-", token.dep_, "-", token.head.text)