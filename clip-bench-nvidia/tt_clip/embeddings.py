import ttnn
import torch 
import math

from device import get_device
from vision import VisionTransformer
from attention import Transformer


class CLIP:
    def __init__(self, state_dict):
        self.token_embedding = state_dict["text_model.embeddings.token_embedding.weight"]
        self.positional_embedding = state_dict["text_model.embeddings.position_embedding.weight"]

        self.text_projection = state_dict["text_projection.weight"]
        self.context_length = self.positional_embedding.shape[0]
        self.vocab_size = self.token_embedding.shape[0]
        self.transformer_width = state_dict["text_model.final_layer_norm.weight"].shape[0]
        transformer_heads = self.transformer_width // 64

        self.ln_final_weights = state_dict["text_model.final_layer_norm.weight"]
        self.ln_final_bias = state_dict["text_model.final_layer_norm.bias"]

        self.logit_scale = state_dict["logit_scale"].item()

        num_vision_layers = 12 # Hardcoded value for CLIP-ViT-base-patch32
        self.visual = VisionTransformer(state_dict, num_vision_layers=num_vision_layers)

        num_text_layers = 12 # Hardcoded value for CLIP-ViT-base-patch32
        self.transformer = Transformer(
            state_dict, num_layers=num_text_layers, num_heads=transformer_heads, attention_mask=self.build_attention_mask(), prefix="text_model.encoder"
        )

    def build_attention_mask(self):
        """
        Build causal attention mask for text transformer.

        Causal masking ensures each token can only attend to itself and previous tokens,
        preventing the model from "cheating" by looking at future tokens. This is essential
        for autoregressive language modeling.

        Returns:
            Upper triangular mask [context_length, context_length] with -inf above diagonal
        """
        # Create a square mask filled with -inf (tokens cannot attend to masked positions)
        # Shape: [context_length, context_length]
        mask = ttnn.full(
            shape=[self.context_length, self.context_length],
            fill_value=float("-inf"),
            dtype=ttnn.bfloat16,
            device=get_device(),
            layout=ttnn.TILE_LAYOUT
        )
        # Keep only upper triangle (excluding diagonal): prevents attending to future tokens
        # diagonal=1 means the diagonal itself is not masked (tokens can attend to themselves)
        mask = ttnn.triu(mask, diagonal=1)
        return mask


    def encode_image(self, image):
        return self.visual.forward(image)

    def encode_text(self, tokens):
        """
        Encode text tokens into feature embeddings.

        Args:
            tokens: Tokenized text input [batch_size, context_length]

        Returns:
            Text embeddings in shared vision-language space [batch_size, embed_dim]
        """
        # Convert token IDs to uint32 for embedding lookup
        tokens = ttnn.typecast(tokens, dtype=ttnn.uint32)

        # Token embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
        x = ttnn.embedding(tokens, weight=self.token_embedding, dtype=ttnn.bfloat16)

        # Add learned positional embeddings
        # Positional embeddings help the model understand token order
        x = x + self.positional_embedding

        # Permute to sequence-first format for transformer
        # [batch, seq_len, embed] -> [seq_len, batch, embed]
        x = ttnn.permute(x, (1, 0, 2))

        # Pass through text transformer with causal masking
        # Causal masking prevents tokens from attending to future tokens
        x = self.transformer.forward(x)

        # Permute back to batch-first format
        # [seq_len, batch, embed] -> [batch, seq_len, embed]
        x = ttnn.permute(x, (1, 0, 2))

        # Final layer normalization
        x = ttnn.layer_norm(x, weight=self.ln_final_weights, bias=self.ln_final_bias)


        # Text Transformer is auto-regressive. This means that the last token has access to all the information in the sequence.
        # We can thus extract text features from the end-of-text (EOT) token position
        # Note: Using PyTorch for argmax since TT-NN doesn't support advanced indexing yet
        torch_tokens = ttnn.to_torch(tokens).to(torch.int64)
        torch_x = ttnn.to_torch(x)

        eot_indices = torch_tokens.argmax(dim=-1)  # [batch_size]
        torch_selected_features = torch_x[torch.arange(torch_x.shape[0]), eot_indices]  # [batch_size, embed_dim]

        # Move back to TT device and apply text projection
        # Projects from transformer hidden size to shared embedding space
        x = ttnn.from_torch(torch_selected_features, device=get_device(), layout=ttnn.TILE_LAYOUT)
        x = ttnn.matmul(x, self.text_projection, transpose_b=True)

        return x

    def forward(self, image, tokens):
        """
        Compute similarity scores between images and text descriptions.

        Args:
            image: Preprocessed image tensor [batch_size, channels, height, width]
            tokens: Tokenized text tensor [batch_size, context_length]

        Returns:
            logits_per_image: Image-to-text similarity scores [batch_size_image, batch_size_text]
            logits_per_text: Text-to-image similarity scores [batch_size_text, batch_size_image]
        """
        # Encode both modalities into the shared embedding space
        text_features = self.encode_text(tokens)  # [batch_text, embed_dim]
        image_features = self.encode_image(image)  # [batch_image, embed_dim]

        # Normalize features to unit vectors for cosine similarity
        # L2 norm: ||x||_2 = sqrt(sum(x_i^2))
        norm_image_features = ttnn.operations.moreh.norm(image_features, p=2.0, dim=1, keepdim=True)
        norm_text_features = ttnn.operations.moreh.norm(text_features, p=2.0, dim=1, keepdim=True)

        # Normalize: x / ||x|| -> unit vector
        image_features = ttnn.divide(image_features, norm_image_features)
        text_features = ttnn.divide(text_features, norm_text_features)

        # Compute cosine similarity scaled by learned temperature parameter
        # logit_scale is learned during training to control the sharpness of the distribution
        logit_scale = math.exp(self.logit_scale)

        # Compute similarity matrix: scaled dot product of normalized features
        # Result: [batch_image, embed] @ [embed, batch_text] = [batch_image, batch_text]
        logits_per_image = ttnn.matmul(logit_scale * image_features, text_features, transpose_b=True)
        # Transpose for text-to-image direction
        logits_per_text = ttnn.transpose(logits_per_image, 0, 1)

        return logits_per_image, logits_per_text