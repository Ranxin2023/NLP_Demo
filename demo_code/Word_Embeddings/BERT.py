from transformers import BertTokenizer, BertModel
import torch
'''

🔍 What This Means:
[1, 7, 768] refers to the embedding tensor dimensions:

1 = batch size (1 sentence)

7 = number of tokens (after tokenization, including [CLS] and [SEP])

768 = the dimensionality of each token's embedding (standard for BERT base)
🧠 Token Embeddings:
tensor([[[-0.2810, -0.0482, -0.1013, ..., -0.4972,  0.1912,  0.9041],
         ...
         [ 0.6660, -0.1010, -0.4946, ...,  0.4617, -0.8314, -0.1667]]],
       grad_fn=<NativeLayerNormBackward0>)
Each row inside this tensor represents the 768-dimensional vector for one token in the sentence.

These embeddings are contextual: they depend on the surrounding words, thanks to the transformer mechanism.

You can use the [CLS] token's embedding (i.e., tensor[0][0]) to represent the whole sentence, or use individual token embeddings for word-level tasks (e.g., NER, parsing).

'''
def BERT_demo():
    print("BERT demo......")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    text = "This is a sample sentence"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # shape: [batch_size, sequence_length, hidden_size]
    print(f"The shape of embeddings are: {embeddings.shape}")
    print(f"word embeddings are:\n{embeddings}")