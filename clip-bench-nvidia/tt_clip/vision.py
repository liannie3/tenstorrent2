import ttnn 
from device import get_device
from attention import Transformer


class VisionTransformer:
    def __init__(self, state_dict, num_vision_layers):
        self.output_dim = 0

        conv2_state_dict_name = "vision_model.embeddings.patch_embedding.weight"
        self.vision_width = state_dict[conv2_state_dict_name].shape[0]
        self.patch_size = state_dict[conv2_state_dict_name].shape[-1]
        self.vision_heads = self.vision_width // 64

        self.class_embedding = state_dict["vision_model.embeddings.class_embedding"]
        self.positional_embedding = state_dict["vision_model.embeddings.position_embedding.weight"]
        self.proj = state_dict["visual_projection.weight"]

        # Weights preparation for convolution (ttnn.conv2d) must be done on host (CPU)
        # To that end, we move convolution weights from device to host and perform its
        # layout to Row-Major, which is the preferred layout for TT-NN convolution kernels.
        conv1_weights = ttnn.from_device(state_dict[conv2_state_dict_name])
        conv1_weights = ttnn.to_dtype(conv1_weights, dtype=ttnn.bfloat16)
        conv1_weights = ttnn.to_layout(conv1_weights, layout=ttnn.ROW_MAJOR_LAYOUT)

        # UPDATED:
        # Originally, conv1_weights was a local variables and would be used in forward and then was overwritten with the prepared result. 
        # The next run then occurred and used the prepared weights and not the original weights. Therefore an error was thrown.
        # This method stops this. The conv1_weights are computed a single time and stored in the dictionary to be used across multiple runs.
        
        self.conv1_weights_host = conv1_weights   
        self._conv1_prepared = {}

        # Layer normalization applied before transformer layers
        self.ln_pre_weights = state_dict["vision_model.pre_layrnorm.weight"]
        self.ln_pre_bias = state_dict["vision_model.pre_layrnorm.bias"]

        # Layer normalization applied after transformer layers (to class token)
        self.ln_post_weights = state_dict["vision_model.post_layernorm.weight"]
        self.ln_post_bias = state_dict["vision_model.post_layernorm.bias"]

        self.transformer = Transformer(
            state_dict, num_layers=num_vision_layers, num_heads=self.vision_heads, attention_mask=None, prefix="vision_model.encoder"
        )

    def forward(self, x):
        (batch_size, in_channels, height, width) = x.shape

        # === Important: TT-NN conv2d differs from PyTorch in tensor layout ===
        #
        # PyTorch Conv2d expects: (N, C_in, H, W) - channels-first "Struct of Arrays"
        # TT-NN conv2d expects:   (N, H, W, C_in) - channels-last "Array of Structs"
        #
        # PyTorch Conv2d output:  (N, C_out, H_out, W_out) - 4D tensor
        # TT-NN conv2d output:    (1, 1, N*H_out*W_out, C_out) - flattened 4D tensor
        #
        # This is why we need to permute and reshape before and after convolution.

        # Step 1: Rearrange from channels-first to channels-last layout
        # [batch_size, in_channels, height, width] -> [batch_size, height, width, in_channels]
        x = ttnn.permute(x, [0, 2, 3, 1])

        # Step 2: Convert to row-major layout (required by ttnn.conv2d)
        # TT-NN convolution kernels are optimized for row-major data access
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Output channels for patch embedding (standard ViT uses 768)
        #out_channels = 768

        # UPDATED: Set out_channels to vision_width as this is 1024 for the CLIP-ViT-L-14/336px Model
        out_channels = self.vision_width  # Should Return 1024
        
        key = (batch_size, height, width)
        if key not in self._conv1_prepared:
            self._conv_prepared[key] = ttnn.prepare_conv_weights(
                weight_tensor=self.conv1_weights_host,
                input_memory_config=x.memory_config(),
                input_layout=x.layout,
                weights_format="OIHW",
                in_channels=in_channels,
                out_channels=out_channels,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                kernel_size=[self.patch_size, self.patch_size],
                stride=[self.patch_size, self.patch_size],
                padding=[0, 0],
                dilation=[1, 1],
                has_bias=False,
                groups=1,
                device=get_device(),
                input_dtype=x.dtype
            )

        # Step 3: Apply patch embedding convolution
        # This converts the 2D image into a sequence of patch embeddings
        # For patch_size=14 and image 336x336: creates (336/14)^2 = 576 patches
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self._conv1_prepared[key],
            in_channels=in_channels,  # Input channels (3 for RGB)
            out_channels=out_channels,  # Embedding dimension (1024)
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            kernel_size=(self.patch_size, self.patch_size),  # Patch size (e.g., 14x14)
            stride=(self.patch_size, self.patch_size),  # Non-overlapping patches: stride = kernel_size
            padding=(0, 0),  # No padding needed
            dilation=(1, 1),  # Standard convolution (no dilation)
            groups=0,  # Standard convolution (not grouped/depthwise)
            device=get_device(),
            return_weights_and_bias=False,  # We already have weights, don't return them
            return_output_dim=False,  # We know the output dimensions
        )

        # Step 4: Reshape convolution output from flattened to sequence format
        # Convert to tile layout for subsequent operations (TT-NN's optimized 2D tiled format)
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        # Unflatten: [1, 1, N*H_out*W_out, C_out] -> [N, num_patches, embed_dim]
        # where num_patches = H_out * W_out = (336/14)^2 = 576

        x = ttnn.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))

        # Step 5: Prepare the [CLS] token (class embedding)
        # ViT prepends a learnable class token to the sequence of patch embeddings
        # Reshape class token: [embed_dim] -> [batch_size, 1, embed_dim]
        class_embedding = ttnn.reshape(self.class_embedding, (x.shape[0], 1, x.shape[-1]))


        # Step 6: Prepare tensors for concatenation
        # Move to DRAM memory (slower but more capacity than L1) for concatenation operation
        # Note: Concatenation currently requires DRAM memory; future optimizations may use L1 sharded memory
        x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Ensure class_embedding has matching memory configuration
        class_embedding = ttnn.to_memory_config(class_embedding, memory_config=x.memory_config())

        # Step 7: Prepend class token to patch sequence
        # [batch, 1, embed] + [batch, num_patches, embed] -> [batch, num_patches+1, embed]
        # This creates: [[CLS], patch_1, patch_2, ..., patch_576]
        x = ttnn.concat([class_embedding, x], dim=1, memory_config=None)

        # Step 8: Add positional embeddings
        # Positional embeddings encode the position of each token in the sequence
        # [1, num_patches+1, embed] -> broadcast and add to [batch, num_patches+1, embed]
        positional_embedding = ttnn.reshape(self.positional_embedding, (1, x.shape[1], x.shape[2]))
        x = x + positional_embedding

        # Step 9: Pre-transformer layer normalization
        x = ttnn.layer_norm(x, weight=self.ln_pre_weights, bias=self.ln_pre_bias)

        # Step 10: Permute to sequence-first format for transformer
        # Transformers typically process in sequence-first format
        # [batch_size, sequence_size, hidden_size] -> [sequence_size, batch_size, hidden_size]
        x = ttnn.permute(x, (1, 0, 2))

        # Step 11: Pass through transformer encoder layers
        x = self.transformer.forward(x)

        # Step 12: Permute back to batch-first format
        # [sequence_size, batch_size, hidden_size] -> [batch_size, sequence_size, hidden_size]
        x = ttnn.permute(x, (1, 0, 2)) 

        return x