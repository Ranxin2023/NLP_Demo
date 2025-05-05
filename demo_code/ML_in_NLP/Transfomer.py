import torch
import torch.nn.functional as F
from demo_code.ML_in_NLP.self_transfomer import TransformerSentimentClassifier
from demo_code.ML_in_NLP.self_transfomer import tokenizer
from demo_code.ML_in_NLP.self_transfomer import vocab

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
