from contextlib import redirect_stdout
from demo_code.ML_in_NLP.transformer.self_implementation import TransformerSentimentClassifier
from demo_code.ML_in_NLP.transformer.self_implementation import tokenizer
from demo_code.ML_in_NLP.transformer.self_implementation import vocab
from demo_code.ML_in_NLP.transformer.datacamp_tutorial import Transformer
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
def transfomer_demo():
    # Input sentence
    print("Transfomer demo....")
    sentence = "I love using transformers for NLP"
    print(f"Input Sentense is{sentence}")

    # Tokenize sentence to IDs using the toy tokenizer
    tokens = tokenizer(sentence)  # e.g., [0, 1, 2, 3, 4, 5]

    # Convert to PyTorch tensor with batch dimension: shape (1, seq_len)
    input_tensor = torch.tensor([tokens])

    # Instantiate the classifier model with random weights
    model = TransformerSentimentClassifier(
        vocab_size=len(vocab), embed_size=32, heads=4,
        ff_hidden=64, dropout=0.1, num_classes=2
    )

    # Run forward pass without computing gradients
    with torch.no_grad():
        logits = model(input_tensor)                # Raw output before softmax
        probs = F.softmax(logits, dim=-1)           # Convert to probabilities
        sentiment = "POSITIVE" if torch.argmax(probs) == 1 else "NEGATIVE"
        print(f"Sentiment: {sentiment}, Confidence: {probs.max().item():.4f}")

def transfomer_datacamp():
    with open("./output_results/transformer_datacamp.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # data preparation
            src_vocab_size = 5000
            tgt_vocab_size = 5000
            d_model = 512
            num_heads = 8
            num_layers = 6
            d_ff = 2048
            max_seq_length = 100
            dropout = 0.1
            transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

            # Generate random sample data
            src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  
            tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

            transformer.train()

            for epoch in range(100):
                optimizer.zero_grad()
                output = transformer(src_data, tgt_data[:, :-1])
                loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch+1}, Loss: {loss.item()}")