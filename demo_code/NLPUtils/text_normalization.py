import re
import nltk
from contextlib import redirect_stdout
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import spacy
def text_normalizaion_demo():
    with open("./output_results/text_normalization.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # Download necessary data
            nltk.download('wordnet')
            nltk.download('punkt')

            # Load spaCy model
            nlp = spacy.load("en_core_web_sm")

            text = "The cats are playing at 5:00PM on 2024-04-02. NLP is fun, and 100 people agree!"

            print("Original Text:\n", text)

            # 1. Lowercasing
            lowercased = text.lower()
            print("\n1. Lowercased:\n", lowercased)

            # 2. Lemmatization (spaCy)
            doc = nlp(lowercased)
            lemmatized = " ".join([token.lemma_ for token in doc])
            print("\n2. Lemmatized:\n", lemmatized)

            # 3. Stemming (Porter Stemmer)
            stemmer = PorterStemmer()
            tokens = nltk.word_tokenize(lowercased)
            stemmed = " ".join([stemmer.stem(token) for token in tokens])
            print("\n3. Stemmed:\n", stemmed)

            # 4. Abbreviation Expansion (simple example using a dictionary)
            abbreviations = {"nlp": "Natural Language Processing"}
            expanded = " ".join([abbreviations.get(word, word) for word in lowercased.split()])
            print("\n4. Abbreviation Expanded:\n", expanded)

            # 5. Numerical Normalization (convert numbers to words using TextBlob or manually)
            blob = TextBlob("100")
            try:
                number_to_words = blob.words[0].replace("100", "one hundred")
            except:
                number_to_words = "one hundred"
            print("\n5. Numerical Normalization:\n", lowercased.replace("100", number_to_words))

            # 6. Date and Time Normalization (extract and standardize with spaCy)
            print("\n6. Detected Date/Time entities:")
            for ent in doc.ents:
                if ent.label_ in ["DATE", "TIME"]:
                    print(ent.text, "->", ent.label_)