import torch
import torch.nn as nn
from contextlib import redirect_stdout
def norm_demo():
    with open("./output_results/lb_norm_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # Input: batch_size=2, seq_len=3, feature_dim=4
            x = torch.tensor([
                [[1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0]],

                [[4.0, 5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0, 9.0]]
            ], dtype=torch.float32)

            print("Original x shape:", x.shape)  # [2, 3, 4]

            # BatchNorm expects [N, C, *] shape, so we reshape to [N*seq, C]
            x_bn_input = x.view(-1, x.size(-1))  # [6, 4]
            batchnorm = nn.BatchNorm1d(num_features=4, affine=False)
            batchnorm_output = batchnorm(x_bn_input).view(2, 3, 4)

            # LayerNorm works directly on [*, D]
            layernorm = nn.LayerNorm(normalized_shape=4, elementwise_affine=False)
            layernorm_output = layernorm(x)

            print("\n=== BatchNorm Output ===")
            print(f"batch norm result: {batchnorm_output}")

            print("\n=== LayerNorm Output ===")
            print(f"layer norm output: {layernorm_output}")
