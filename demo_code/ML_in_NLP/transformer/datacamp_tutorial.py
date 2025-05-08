import math
import torch
import torch.nn as nn
import torch.optim as optim
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe=torch.zeros(max_seq_length, d_model)
        position=torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        '''
        In Transformer models, positional encoding is added to input embeddings to give the model a sense of word order. This is done using sine and cosine functions of different frequencies.

        oscillate across dimensions.
        '''
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2]=torch.sin(position*div_term)
        pe[:, 1::2]=torch.cos(position*div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x+self.pe[:, :x.size(1)]
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
        self.d_k=d_model//num_of_heads
        self.W_q=nn.Linear(d_model, d_model)
        self.W_k=nn.Linear(d_model, d_model)
        self.W_v=nn.Linear(d_model, d_model)
        self.W_o=nn.Linear(d_model, d_model)

    def scale_dot_product_attention(self, Q, K, V, mask=None):
        attention_score=torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(self.d_k)
        if mask is not None: 
            '''
            It's a numerical trick:

            Softmax turns large negative values into values very close to zero.

            -1e9 is so small that:

            
            softmax([-1e9, 0]) ≈ [0.0, 1.0]
            This ensures the masked positions get nearly zero attention weight.
            '''
            attention_score=attention_score.masked_fill(mask == 0, -1e9)
        attention_probability=torch.softmax(attention_score, dim=-1)
        output=torch.matmul(attention_probability, V)
        return output

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
    # Explain in tutorial
    # This method reshapes the input x into the shape (batch_size, num_heads, seq_length, d_k). 
    # It enables the model to process multiple attention heads concurrently, allowing for parallel computation.
    def split_heads(self, x):
        batch_size, seq_length, d_model=x.size()
        return x.view(batch_size, seq_length, self.num_of_heads, self.d_k).transpose(1, 2)
    '''
    🔄 Why Combine Heads?
    🧠 Each head captures different types of relationships (syntax, semantics, etc.)

    📦 But downstream layers (like FeedForward) expect a single vector per token — not multiple heads.

    🧩 So we merge all attention heads into a single sequence of vectors with shape (batch_size, seq_len, d_model).
    '''
    #This function is used after attention has been computed separately for each head. Its job is to:
    #🔁 combine multiple attention heads back into a single tensor so it can be processed by the rest of the transformer block.
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attention_output = self.scale_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention_output))
        return output
'''
Processing steps:

1. Self-attention: The input x is passed through the multi-head self-attention mechanism.
2. Add and normalize (after attention): The attention output is added to the original input (residual connection), followed by dropout and normalization using norm1.
3. Feed-forward network: The output from the previous step is passed through the position-wise feed-forward network.
4. Add and normalize (after feed-forward): Similar to step 2, the feed-forward output is added to the input of this stage (residual connection), followed by dropout and normalization using norm2.
5. Output: The processed tensor is returned as the output of the encoder layer.
'''
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
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
'''
Processing steps:

1. Self-attention on target sequence: The input x is processed through a self-attention mechanism.
2. Add and normalize (after self-attention): The output from self-attention is added to the original x, followed by dropout and normalization using norm1.
3. Cross-attention with encoder output: The normalized output from the previous step is processed through a cross-attention mechanism that attends to the encoder's output enc_output.
4. Add and normalize (after cross-attention): The output from cross-attention is added to the input of this stage, followed by dropout and normalization using norm2.
5. Feed-forward network: The output from the previous step is passed through the feed-forward network.
6. Add and normalize (after feed-forward): The feed-forward output is added to the input of this stage, followed by dropout and normalization using norm3.
7. Output: The processed tensor is returned as the output of the decoder layer.
'''
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention=MultiHead_Attention(d_model=d_model, num_of_heads=num_heads)
        self.cross_attention=MultiHead_Attention(d_model=d_model, num_of_heads=num_heads)
        self.feed_forward=PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x, enc_output,src_mask, tgt_mask):
        attention_output=self.self_attention(x, x, x, tgt_mask)
        x=self.norm1(x + self.dropout(attention_output))
        attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
class Transformer(nn.Module):
    '''
    The constructor takes the following parameters:

    1. src_vocab_size: Source vocabulary size.
    2. tgt_vocab_size: Target vocabulary size.
    3. d_model: The dimensionality of the model's embeddings.
    4. num_heads: Number of attention heads in the multi-head attention mechanism.
    5. num_layers: Number of layers for both the encoder and the decoder.
    6. d_ff: Dimensionality of the inner layer in the feed-forward network.
    7. max_seq_length: Maximum sequence length for positional encoding.
    8. dropout: Dropout rate for regularization.
    '''
    def __init__(self,src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding=nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding=nn.Embedding(tgt_vocab_size, d_model)
        self.position_encoding=PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers=nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.position_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.position_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
