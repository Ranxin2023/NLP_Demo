import torch
import torch.nn.functional as F
from demo_code.ML_in_NLP.transformer.self_implementation import TransformerSentimentClassifier
from demo_code.ML_in_NLP.transformer.self_implementation import tokenizer
from demo_code.ML_in_NLP.transformer.self_implementation import vocab
from demo_code.ML_in_NLP.transformer.datacamp_tutorial import Transformer_Datacamp
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
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1