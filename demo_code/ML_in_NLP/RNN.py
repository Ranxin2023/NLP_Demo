import torch 
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import re
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

class RNNClassifier:
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Last time step
        return torch.sigmoid(self.fc(out))
    
# 2. Dataset & DataLoader
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def RNN_demo():
    # 1. Preprocessing: simple tokenizer
    sentences = ["I love NLP", "NLP is amazing", "I hate spam", "Spam is bad"]
    labels = [1, 1, 0, 0]
    # Build vocabulary
    tokenized = [tokenize(sent) for sent in sentences]
    vocab = Counter(token for sent in tokenized for token in sent)
    vocab = {word: idx+1 for idx, (word, _) in enumerate(vocab.items())}  # 0 for padding
    vocab = Counter(token for sent in tokenized for token in sent)
    model = RNNClassifier(vocab_size=len(vocab), embed_dim=8, hidden_size=8)
