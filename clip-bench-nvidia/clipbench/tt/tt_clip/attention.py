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
        # Batch-first layout [batch, seq, hidden]. Do NOT use sequence-first [seq, batch, hidden]:
        # in tile layout the last two dims are padded to 32x32, so (seq, 1, hidden) pads every
        # row to a 32-row tile and runs `seq` tiny matmuls instead of one big one.
        batch_size, sequence_size, hidden_size = hidden_states.shape

        head_size = hidden_size // self.num_heads

        # Q, K, V projections: [batch, seq, hidden] -> [batch, seq, hidden]
        q = ttnn.linear(hidden_states, self.q_proj_weight, bias=self.q_proj_bias, transpose_b=True)
        k = ttnn.linear(hidden_states, self.k_proj_weight, bias=self.k_proj_bias, transpose_b=True)
        v = ttnn.linear(hidden_states, self.v_proj_weight, bias=self.v_proj_bias, transpose_b=True)

        # Split heads: [batch, seq, hidden] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        q = ttnn.permute(ttnn.reshape(q, (batch_size, sequence_size, self.num_heads, head_size)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (batch_size, sequence_size, self.num_heads, head_size)), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, (batch_size, sequence_size, self.num_heads, head_size)), (0, 2, 1, 3))

        # Scores: [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq] -> [batch, heads, seq, seq]
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        scores = scores * self.scale

        # Optional mask (causal attention in a text encoder); broadcastable to the scores shape
        if self.attention_mask is not None:
            scores = scores + self.attention_mask

        # numeric_stable=True computes softmax(x - max(x)) to avoid overflow in exp
        attn_weights = ttnn.softmax(
            scores, dim=-1, numeric_stable=True,
        )

        # [batch, heads, seq, seq] @ [batch, heads, seq, head_dim] -> [batch, heads, seq, head_dim]
        attn_output = ttnn.matmul(attn_weights, v)

        # Merge heads: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim] -> [batch, seq, hidden]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, sequence_size, hidden_size))

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


    def forward(self, x, output_hidden_states=False):
        ## UPDATED
        ## Same order as HF: the input (after pre_layrnorm), then after each layer
        hidden_states = [x] if output_hidden_states else None

        for layer in self.layers:
            x = layer.forward(x)
            if output_hidden_states:
                hidden_states.append(x)

        if output_hidden_states:
            return x, hidden_states
        return x