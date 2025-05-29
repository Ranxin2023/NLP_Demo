import torch
import torch.nn as nn

import math

vocab = {"i": 0, "love": 1, "using": 2, "transformers": 3, "for": 4, "nlp": 5, "<pad>": 6}
tokenizer = lambda sentence: [vocab.get(word.lower(), 6) for word in sentence.split()]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe=torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                # Even index: sine
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/d_model)))
                # Odd index: cosine
                if i + 1 < d_model:
                    pe[pos, i+1] = math.cos(pos / (10000 ** ((2*(i+1))/d_model)))
        # Add a batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # Register as a buffer so it's saved with the model but not trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input embeddings
        return x + self.pe[:, :x.size(1)]
    
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads=heads
        self.head_dim=embed_size//heads
        assert embed_size%heads==0
         # Linear transformations for queries, keys, values
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        # Final output projection
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        N, query_len, _ = query.shape

        # Apply linear projections and reshape to (batch, seq_len, heads, head_dim)
        values = self.values(values).view(N, -1, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, -1, self.heads, self.head_dim)
        queries = self.queries(query).view(N, -1, self.heads, self.head_dim)

        # Compute scaled dot-product attention: shape (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        # Compute attention-weighted sum of values
        out = torch.einsum("nhql,nlhd->nqhd", attention, values)

        # Concatenate heads and project
        out = out.reshape(N, query_len, -1)
        return self.fc_out(out)
    
class TransfomerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden, dropout):
        super().__init__()
        # Multi-head self-attention layer
        self.attention = SelfAttention(embed_size, heads)
        # LayerNorm after attention and FFN
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Feedforward neural network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply attention + residual + normalization
        attention = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attention))
        # Apply FFN + residual + normalization
        forward = self.feed_forward(x)
        return self.norm2(x + self.dropout(forward))

    
class TransformerSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, ff_hidden, dropout, num_classes):
        super().__init__()
        # Embedding layer for tokens
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Add positional information
        self.pos_encoding = PositionalEncoding(embed_size)
        # Apply transformer block (1 encoder layer)
        self.transformer = TransfomerBlock(embed_size, heads, ff_hidden, dropout)
        # Classifier layer (e.g., 2 classes: positive/negative)
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # Shape: (batch_size, seq_len, embed_size)
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        # Aggregate via mean pooling over sequence length
        x = x.mean(dim=1)
        return self.classifier(x)
