import torch
import math
import torch.nn as nn

def sinusoidal_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape: [1, seq_len, d_model]

class LearnablePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.pos_embedding(pos_ids)

# Example
x = torch.randn(2, 6, 16)  # batch, seq_len, embed_dim
pos_encoder = LearnablePositionEncoding(max_len=50, d_model=16)
print("Learnable Positional Embedding:\n", pos_encoder(x))

def relative_position(seq_len):
    return torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)

rel_pos = relative_position(4)
print("Relative Position Matrix:\n", rel_pos)

def apply_rope(x, seq_len):
    d = x.shape[-1]
    half_d = d // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half_d) / half_d))
    pos = torch.arange(seq_len)
    angles = torch.einsum('i,j->ij', pos, freq)

    sin = torch.sin(angles).unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0)

    x1, x2 = x[..., :half_d], x[..., half_d:]
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated

# Example: [batch, seq_len, dim]
x = torch.randn(1, 6, 16)
x_rope = apply_rope(x, seq_len=6)
print("RoPE Applied:\n", x_rope)
