import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import redirect_stdout
from transformers import BertTokenizer, BertModel
def attention_demo():
    # Dummy input: batch_size=1, seq_len=4, embed_dim=8
    batch_size, seq_len, embed_dim = 1, 4, 8
    num_heads = 2
    head_dim = embed_dim // num_heads
    x = torch.rand(batch_size, seq_len, embed_dim)  # shape: [1, 4, 8]
    # Linear layers for projecting Q, K, V
    W_q = nn.Linear(embed_dim, embed_dim)
    W_k = nn.Linear(embed_dim, embed_dim)
    W_v = nn.Linear(embed_dim, embed_dim)
    W_o = nn.Linear(embed_dim, embed_dim)  # Output projection for multi-head
    def single_head_self_attention(x):
        Q = W_q(x)  # [B, T, D]
        K = W_k(x)
        V = W_v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (embed_dim ** 0.5)  # [B, T, T]
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)  # [B, T, D]
        return output, attn_weights

    def multi_head_attention(x, num_heads):
        B, T, D = x.shape
        print(f"B, T, D for x shape is{B}, {T}, {D}")
        H = num_heads
        D_h = D // H

        # Project Q, K, V and reshape for multi-head
        Q = W_q(x).reshape(B, T, H, D_h).transpose(1, 2)  # [B, H, T, D_h]
        K = W_k(x).reshape(B, T, H, D_h).transpose(1, 2)
        V = W_v(x).reshape(B, T, H, D_h).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D_h ** 0.5)  # [B, H, T, T]
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)  # [B, H, T, D_h]

        # Concatenate heads and apply final linear projection
        output = output.transpose(1, 2).reshape(B, T, D)  # [B, T, D]
        return W_o(output), attn_weights
    
    with open("./output_results/attention_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # Run both
            print("attention of transformer......")
            single_output, single_weights = single_head_self_attention(x)
            multi_output, multi_weights = multi_head_attention(x, num_heads=num_heads)
            # Display results
            print("== Single-Head Self-Attention Output ==")
            print(single_output)
            print("\n== Multi-Head Attention Output ==")
            print(multi_output)
            print("\n== Multi-Head Attention Weights Shape ==")
            print(multi_weights.shape)  # Should be [1, 2, 4, 4] for [B, H, T, T]
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

            # Example sentence (length = 4)
            inputs = tokenizer("The cat sat down", return_tensors="pt")
            outputs = model(**inputs)

            # attention: List of [layer] tensors of shape [B, H, T, T]
            attentions = outputs.attentions  # 12 layers, each [1, 12, T, T]

            # Let's inspect attention from layer 0, head 0
            attn_layer0_head0 = attentions[0][0, 0]  # [T, T]
            print("BERT Layer 0 Head 0 Attention Weights:\n", attn_layer0_head0)

    def dot_product_attention(Q, K):
        return torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))

    def additive_attention(Q, K):
        # suppose Q, K: [batch, seq_len, d_model]
        Q_exp = Q.unsqueeze(2)  # [batch, seq_len_q, 1, d_model]
        K_exp = K.unsqueeze(1)  # [batch, 1, seq_len_k, d_model]
        return torch.tanh(Q_exp + K_exp).sum(dim=-1)
    
    Q = K = torch.rand(1, 4, 8)
    with open("./output_results/matrix_multi.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("dot product attention:", dot_product_attention(Q, K).shape)
            print("additive attention:", additive_attention(Q, K).shape)