import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

class MultiHead_Attention(nn.Module):
    def __init__(self, d_model, num_of_heads):
        super(MultiHead_Attention, self).__init__()
        assert d_model%num_of_heads==0
        self.d_model=d_model
        self.num_of_heads=num_of_heads
        self.d_k=d_model
        self.W_q=nn.Linear(d_model, d_model)
        self.W_k=nn.Linear(d_model, d_model)
        self.W_v=nn.Linear(d_model, d_model)
        self.W_o=nn.Linear(d_model, d_model)
    def scale_dot_product_attention(self, Q, K, V):
        attention_score=torch.matmul(Q, K.transponse(-2, -1))/math.sqrt(self.d_k)
        
