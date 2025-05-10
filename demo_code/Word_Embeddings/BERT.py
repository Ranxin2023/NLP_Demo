from transformers import BertTokenizer, BertModel
import torch
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