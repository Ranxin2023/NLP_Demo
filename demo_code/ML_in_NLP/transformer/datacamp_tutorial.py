import math
import torch
import torch.nn as nn
import torch.optim as optim
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1=nn.Linear(d_model, d_ff)
        self.fc2=nn.Linear(d_ff, d_model)
        self.relu=nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
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

    def scale_dot_product_attention(self, Q, K, V, mask=None):
        attention_score=torch.matmul(Q, K.transponse(-2, -1))/math.sqrt(self.d_k)
        if mask is not None: 
            attention_score=attention_score.masked_fill(mask == 0, -1e9)
    '''
    🔍 Why Split into Multiple Heads?
    1. Learn Diverse Representations (Multiple Attention Subspaces)
    Each head learns different patterns or relationships in the input data.

    One head might focus on short-term dependencies, while another tracks long-range dependencies.

    This is more expressive than a single attention mechanism.

    Example:
    In language:
    One head could focus on subject-verb agreement, another on pronoun resolution, and another on sentence structure.

    2. Improved Performance Without Increasing Depth
    Splitting allows parallel attention operations on smaller subspaces (head_dim = embed_size // heads), keeping computational cost manageable.

    Instead of one huge dot-product (which is expensive), you do many small ones and combine the results.

    3. Efficiency with Same Embedding Size
    Multi-head attention keeps the total dimension fixed (e.g., embed_size=512), while each head only processes 512 / heads dimensions.

    After all heads finish, you concatenate the outputs and project back to embed_size.


    '''
    def split_heads(self, x):
        batch_size, seq_length, d_model=x.size()
        return x.view(batch_size, seq_length, self.num_of_heads, self.d_k).transpose(1, 2)
'''
Processing steps:

1. Self-attention: The input x is passed through the multi-head self-attention mechanism.
2. Add and normalize (after attention): The attention output is added to the original input (residual connection), followed by dropout and normalization using norm1.
3. Feed-forward network: The output from the previous step is passed through the position-wise feed-forward network.
4. Add and normalize (after feed-forward): Similar to step 2, the feed-forward output is added to the input of this stage (residual connection), followed by dropout and normalization using norm2.
5. Output: The processed tensor is returned as the output of the encoder layer.
'''
class EncodeLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncodeLayer, self).__init__()
        self.self_attention=MultiHead_Attention(d_model=d_model, num_of_heads=num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        '''
        In a Transformer’s encoder or decoder, dropout is a regularization technique used to 
        prevent overfitting and improve generalization. It randomly "drops" units (sets them to zero) during training, 
        which forces the model to be more robust by not relying too much on any one part of its architecture.
        '''
        self.dropout=nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output=self.self_attention(x, x, x, mask)
        x=self.norm1(x+self.dropout(attention_output))
        ff_output=self.feed_forward(x)
        x=self.norm2(x+self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.attention=MultiHead_Attention(d_model=d_model, num_of_heads=num_heads)

class Transformer_Datacamp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)