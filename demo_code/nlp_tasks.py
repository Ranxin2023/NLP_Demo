from contextlib import redirect_stdout
def nlp_tasks():
    with open("./output_results/nlp_tasks.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            from transformers import pipeline

            # Explicit model for classification
            classifier = pipeline(
                "text-classification", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            '''
            label: 'POSITIVE' → The model thinks the sentence has a positive sentiment.
            score: 0.99986 → This is the model’s confidence, which is 99.99% sure the sentence is positive.
            '''
            
            result = classifier("I really enjoyed the new Batman movie!")
            print("Text Classification:", result)
            import spacy

            nlp = spacy.load("en_core_web_sm")
            doc = nlp("Barack Obama was born in Hawaii and became the president of the United States.")


            '''
            Barack Obama → PERSON: Recognized as a person.

            Hawaii → GPE: GPE stands for "Geo-Political Entity" — like a city, state, or country.

            The United States → GPE: Also correctly labeled as a geo-political location.
            '''
            print("Named Entities:")
            for ent in doc.ents:
                print(ent.text, "->", ent.label_)