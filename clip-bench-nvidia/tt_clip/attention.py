import ttnn 
import math


class MultiHeadAttention:
    def __init__(self, state_dict, num_heads, attention_mask=None, prefix=""):
        self.attention_mask = attention_mask
        self.prefix = prefix
        self.num_heads = num_heads

        self.q_proj_weight = state_dict[f"{prefix}.q_proj.weight"]
        self.q_proj_bias = state_dict[f"{prefix}.q_proj.bias"]
        self.k_proj_weight = state_dict[f"{prefix}.k_proj.weight"]
        self.k_proj_bias = state_dict[f"{prefix}.k_proj.bias"]
        self.v_proj_weight = state_dict[f"{prefix}.v_proj.weight"]
        self.v_proj_bias = state_dict[f"{prefix}.v_proj.bias"]
        self.out_proj_weight = state_dict[f"{prefix}.out_proj.weight"]
        self.out_proj_bias = state_dict[f"{prefix}.out_proj.bias"]

        # Scale factor for attention scores: 1/sqrt(head_dim) for numerical stability.
        # q_proj.weight is [hidden_size, hidden_size], and hidden_size is split evenly
        # across the heads, so head_dim = hidden_size // num_heads.
        head_size = self.q_proj_weight.shape[0] // num_heads
        self.scale = 1.0 / math.sqrt(head_size)


    def forward(self, hidden_states):
        sequence_size, batch_size, hidden_size = hidden_states.shape

        head_size = hidden_size // self.num_heads

        # Compute Q, K, V projections from input hidden states
        # Each projection: [sequence_size, batch_size, hidden_size] -> [sequence_size, batch_size, hidden_size]
        q = ttnn.linear(hidden_states, self.q_proj_weight, bias=self.q_proj_bias, transpose_b=True)
        k = ttnn.linear(hidden_states, self.k_proj_weight, bias=self.k_proj_bias, transpose_b=True)
        v = ttnn.linear(hidden_states, self.v_proj_weight, bias=self.v_proj_bias, transpose_b=True)

        # Reshape for multi-head attention: split hidden_size into (num_heads * head_dim)
        # [sequence_size, batch_size, hidden_size] -> [sequence_size, batch_size * num_heads, head_dim]
        q = ttnn.reshape(q, (sequence_size, batch_size * self.num_heads, head_size))
        k = ttnn.reshape(k, (sequence_size, batch_size * self.num_heads, head_size))
        v = ttnn.reshape(v, (sequence_size, batch_size * self.num_heads, head_size))

        # Transpose to bring batch_size * num_heads dimension first for parallel attention computation
        # [sequence_size, batch_size * num_heads, head_dim] -> [batch_size * num_heads, sequence_size, head_size]
        q = ttnn.transpose(q, 0, 1)
        k = ttnn.transpose(k, 0, 1)
        v = ttnn.transpose(v, 0, 1)

        # Compute attention scores: Q @ K^T
        # [batch_size * num_heads, sequence_size, head_size] @ [batch_size * num_heads, head_size, sequence_size]
        #   -> [batch_size * num_heads, sequence_size, sequence_size]
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        # Scale scores by 1 / sqrt(head_size) to prevent softmax saturation
        scores = scores * self.scale

        # Apply attention mask if provided (for causal attention in text encoder)
        if self.attention_mask is not None:
            # Add mask to scores (mask contains -inf for positions that should be ignored)
            # Mask is broadcastable to [batch_size * num_heads, sequence_size, sequence_size]
            scores = scores + self.attention_mask

        # Apply softmax to get attention weights
        # numeric_stable=True uses the numerically stable softmax: softmax(x) = softmax(x - max(x))
        # This prevents overflow when computing exp(x) for large values
        attn_weights = ttnn.softmax(
            scores, dim=-1, numeric_stable=True,
        )

        # [batch_size * num_heads, sequence_size, sequence_size] @ [batch_size*heads, sequence_size, head_size]
        #   -> [batch_size * num_heads, sequence_size, head_size]
        attn_output = ttnn.matmul(attn_weights, v)

        # Transpose back to sequence-first format
        # [batch_size * num_heads, sequence_size, head_size] -> [sequence_size, batch_size * num_heads, head_size]
        attn_output = ttnn.transpose(attn_output, 0, 1)

        # Merge heads back into hidden dimension
        # [sequence_size, batch_size * num_heads, head_size] -> [sequence_size, batch_size, hidden_size]
        attn_output = ttnn.reshape(attn_output, (sequence_size, batch_size, hidden_size))

        # Apply output projection
        dense_out = ttnn.linear(
            attn_output,
            self.out_proj_weight,
            bias=self.out_proj_bias,
            transpose_b=True,
        )

        return dense_out

class MultilayerPerceptron:
    def __init__(self, state_dict, attention_mask=None, prefix=""):
        self.prefix = prefix

        self.mlp_c_fc_weight = state_dict[f"{prefix}.fc1.weight"]
        self.mlp_c_fc_bias = state_dict[f"{prefix}.fc1.bias"]
        self.mlp_c_proj_weight = state_dict[f"{prefix}.fc2.weight"]
        self.mlp_c_proj_bias = state_dict[f"{prefix}.fc2.bias"]

    def forward(self, x):
        x = ttnn.linear(x, self.mlp_c_fc_weight, bias=self.mlp_c_fc_bias, transpose_b=True)
        
        # x = ttnn.gelu(x)
        x = x * ttnn.sigmoid(x*1.702)   # GELU --> QuickGELU, OpenAI trained on QuickGELU (https://huggingface.co/spaces/HtSimple/ImageSearch/blame/c859bb5db8d96d446afa2b9dd00b236f36f98f3f/CLIPTool/CLIP/model.py)

        x = ttnn.linear(x, self.mlp_c_proj_weight, bias=self.mlp_c_proj_bias, transpose_b=True)
        return x


class ResidualAttentionBlock:
    def __init__(self, state_dict, num_heads, attention_mask=None, prefix=""):
        self.prefix = prefix
        self.num_heads = num_heads

        self.attention = MultiHeadAttention(state_dict, num_heads=num_heads, attention_mask=attention_mask, prefix=f"{prefix}.self_attn")
        self.mlp = MultilayerPerceptron(state_dict, prefix=f"{prefix}.mlp")

        self.layer_norm_1_weight = state_dict[f"{prefix}.layer_norm1.weight"]
        self.layer_norm_1_bias = state_dict[f"{prefix}.layer_norm1.bias"]
        self.layer_norm_2_weight = state_dict[f"{prefix}.layer_norm2.weight"]
        self.layer_norm_2_bias = state_dict[f"{prefix}.layer_norm2.bias"]


    def forward(self, x):
        # LayerNorm
        residual = x
        x = ttnn.layer_norm(x, weight=self.layer_norm_1_weight, bias=self.layer_norm_1_bias)

        # Multihead attention / Self-Attention
        # This must be equal to nn.MultiheadAttention(d_model, n_head)(x, x, x, need_weights=False, attn_mask=self.attn_mask)
        x = residual + self.attention.forward(x)

        # LayerNorm
        x_post_layer_norm = ttnn.layer_norm(x, weight=self.layer_norm_2_weight, bias=self.layer_norm_2_bias)

        # Multi-Layer Perceptron
        x = x + self.mlp.forward(x_post_layer_norm)

        return x

class Transformer:
    def __init__(self, state_dict, num_layers, num_heads, attention_mask=None, prefix=""):
        """
        Initialize a generic Transformer that can be used for both text and vision encoding.

        Args:
            state_dict: Model weights dictionary
            num_heads: Number of attention heads
            attention_mask: Attention mask for causal attention (used for text, None for vision)
            prefix: Prefix for layer names in state_dict (e.g., "text_model.encoder" or "vision_model.encoder")
        """
        self.prefix = prefix
        self.layers = []

        # Initialize each transformer layer with converted weights
        self.layers = [
            ResidualAttentionBlock(state_dict, attention_mask=attention_mask, num_heads=num_heads, prefix=f"{prefix}.layers.{i}")
            for i in range(0, num_layers)
        ]


    def forward(self, x):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer.forward(x)

        return x