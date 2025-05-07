import nltk
def chunking():
    from nltk.chunk import RegexpParser

    sentence = "The quick brown fox jumps over the lazy dog"
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    # Define a chunk grammar - noun phrases (NP): determiner + adjectives + noun
    chunk_grammar = r"NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = RegexpParser(chunk_grammar)
    tree = chunk_parser.parse(pos_tags)

    print("\nChunked Phrases (Noun Phrases):")
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            print(" ".join(word for word, tag in subtree.leaves()))
