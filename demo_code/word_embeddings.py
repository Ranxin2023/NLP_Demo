import gensim.downloader as api
from contextlib import redirect_stdout, redirect_stderr
def word_embedding_demo():
    with open("./output_results/word_embeddings.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f), redirect_stderr(f):
            # Load pre-trained Word2Vec (Google News vectors)
            model = api.load("word2vec-google-news-300")

            # Example 1: Semantic similarity
            similarity = model.similarity("king", "queen")
            print(f"Similarity between 'king' and 'queen': {similarity:.4f}")
            '''
            This is the cosine similarity between the vectors for “king” and “queen”.

            The value 0.6511 shows they are fairly similar (on a scale of -1 to 1).

            This makes sense — they’re both royal nouns with similar usage.

            '''
            # Example 2: Syntactic analogy
            result = model.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
            print(f"'king' - 'man' + 'woman' = '{result[0][0]}' (score: {result[0][1]:.4f})")
            '''
            This demonstrates a classic word embedding analogy:

            king
            −
            man
            +
            woman
            ≈
            queen
            king−man+woman≈queen
            A high score (0.7118) means the result vector is very close to the vector for “queen.”

            This shows that the model captures gender analogies based on context.

            '''
            # Example 3: Nearest neighbors
            neighbors = model.most_similar("apple", topn=5)
            print("Words most similar to 'apple':")
            for word, score in neighbors:
                print(f"  {word}: {score:.4f}")
            '''
            These are the top 5 most similar words to “apple” by cosine similarity.

            The results show:

            apples, pears, fruit, berry → all semantically related.

            This proves that Word2Vec successfully groups words based on meaning and context in text.


            '''
            # Example 4: Out-of-vocabulary check
            word = "datascience"
            if word in model:
                print(f"{word} is in vocabulary")
            else:
                print(f"{word} is OOV (Out-of-Vocabulary)")
            '''
            “datascience” does not exist in the model’s vocabulary.

            Word2Vec is not character-based — it can't handle new or unseen words unless they existed in the training data.

            This shows the model’s vocabulary is fixed and limited to the 3 million words it was trained on.


            '''