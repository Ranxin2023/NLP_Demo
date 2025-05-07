def NER_demo():
    import spacy
    nlp = spacy.load("en_core_web_sm")

    doc = nlp("Barack Obama was born in Hawaii and served as President of the United States.")
    print("\nNamed Entity Recognition:")
    for ent in doc.ents:
        print(f"{ent.text:<25} ➝ {ent.label_}")