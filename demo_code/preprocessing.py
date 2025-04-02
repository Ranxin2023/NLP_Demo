import re
import spacy
from bs4 import BeautifulSoup
from contextlib import redirect_stdout
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
def preprocessing_demo():
    with open("./output_results/text_preprocessing.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # Read from HTML file
            html_content=None
            with open("datasets/text_preprocessing.html", "r", encoding="utf-8") as f:
                html_content = f.read()

            # 1. Remove HTML tags
            text_no_html = BeautifulSoup(html_content, "html.parser").get_text()
            #  This is the full visible text extracted from index.html using BeautifulSoup.
            print("Text from HTML:\n", text_no_html)

            # 2. Lowercasing
            text_lower = text_no_html.lower()
            # Converted to lowercase for uniformity. This helps eliminate case sensitivity in tokens like "The" vs "the".
            print("\nLowercase:", text_lower)

            # 3. Tokenization
            tokens = word_tokenize(text_lower)
            # The text is split into words and punctuation using nltk.word_tokenize.
            print("\nTokens:", tokens)

            # 4. Stop Word Removal
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [word for word in tokens if word not in stop_words]
            #  Common stopwords (like "the", "is", "in", "and", "it", "for", "with", "a") were removed.
            print("\nNo Stopwords:", filtered_tokens)

            # 5. Stemming
            stemmer = PorterStemmer()
            stemmed = [stemmer.stem(word) for word in filtered_tokens]
            '''
            Words are reduced to their base root forms:

            learning → learn

            techniques → techniqu

            waiting → wait

            someone → someon
            '''
            print("\nStemmed:", stemmed)

            # 6. Lemmatization
            lemmatizer = WordNetLemmatizer()
            lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
            '''
             Words are normalized to their dictionary (lemma) forms:

            techniques → technique

            waiting stays waiting (already base form)
            '''
            print("\nLemmatized:", lemmatized)

            # 7. Remove Punctuation & Special Characters
            text_no_punct = re.sub(r'[^\w\s]', '', text_lower)
            print("\nNo Punctuation:", text_no_punct)

            # 8. Spell Correction
            corrected = TextBlob("I am lerning NLP and it is awsome").correct()
            print("\nSpell Corrected:", corrected)

            # 9. Sentence Segmentation
            sentences = sent_tokenize(text_no_html)
            print("\nSentences:", sentences)

            # 10. Date and Time Normalization using spaCy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text_no_html)
            print("\nDetected Date/Time entities:")
            for ent in doc.ents:
                if ent.label_ in ["DATE", "TIME"]:
                    print(ent.text, "->", ent.label_)