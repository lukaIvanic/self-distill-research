import torch
import torch.nn as nn
from torch.nn import functional as F

import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, ctx_size, batch_size, device):
        positions = torch.arange(0, ctx_size, dtype=torch.long, device=device).unsqueeze(0)
        positions_embedded = self.pos_embedding(positions)
        return positions_embedded.repeat(batch_size, 1, 1)


# class Head(nn.Module):
#     def __init__(self, d_model, head_size, p_dropout, ctx_size, tril):
#         super().__init__()
#         self.key = nn.Linear(d_model, head_size, bias=False)
#         self.query = nn.Linear(d_model, head_size, bias=False)
#         self.value = nn.Linear(d_model, head_size, bias=False)
#         self.tril = tril
#
#
#
#         self.dropout = nn.Dropout(p_dropout)
#
#     def forward(self, x):
#         B, T, C = x.shape
#         k = self.key(x)
#         q = self.query(x)
#         wei = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
#
#         wei = F.softmax(wei, dim=-1)
#
#         # attn_scores_for_distill = wei.clone()
#
#         wei = self.dropout(wei)
#         v = self.value(x)
#         out = wei @ v
#         return out, None
#
#     def inference(self, x, kv_cache):
#
#         past_k, past_v = (None, None) if kv_cache is None else kv_cache
#
#
#         B, T_query, C = x.shape
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
#
#         if past_k is not None:
#             k = torch.cat((past_k, k), dim=-2)
#             v = torch.cat((past_v, v), dim=-2)
#
#         new_head_cache = (k, v)
#         T_key = k.size(-2)
#
#         wei = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5
#
#         if T_query > 1:
#             wei = wei.masked_fill(self.tril[:T_query, :T_key] == 0, float('-inf'))
#
#
#
#         wei = F.softmax(wei, dim=-1)
#         wei = self.dropout(wei)
#
#         out = wei @ v
#         return out, new_head_cache


# TODO: Keeping this in comments because will later need inference function in OptimizedMHA
# class MultiHeadAttention(nn.Module):
#     def __init__(self, n_heads, head_size, d_model, p_dropout, ctx_size, tril):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(d_model=d_model,
#                                          head_size=head_size,
#                                          p_dropout=p_dropout,
#                                          ctx_size=ctx_size,
#                                          tril=tril) for _ in range(n_heads)])
#         self.output_proj_mha = nn.Linear(head_size * n_heads, d_model)
#         self.dropout = nn.Dropout(p_dropout)  # Use global dropout
#
#     def forward(self, x):
#         head_outputs = [h(x) for h in self.heads]
#         head_individual_outputs = [data[0] for data in head_outputs]
#         # head_individual_attn_scores = [data[1] for data in head_outputs]
#
#         out = torch.cat(head_individual_outputs, dim=-1)
#         out = self.dropout(self.output_proj_mha(out))
#
#         # stacked_attn_scores = torch.stack(head_individual_attn_scores, dim=1)
#
#         return out, None
#
#     def inference(self, x, kv_cache_list):
#         if kv_cache_list is None:
#             kv_cache_list = [None] * len(self.heads)
#
#         head_outputs = []
#         new_kv_cache_list = []
#         for i, h in enumerate(self.heads):
#             out, new_head_cache = h.inference(x, kv_cache_list[i])
#             head_outputs.append(out)
#             new_kv_cache_list.append(new_head_cache)
#
#         out = torch.cat(head_outputs, dim=-1)
#         out = self.dropout(self.output_proj_mha(out))
#         return out, new_kv_cache_list

class OptimizedMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, d_model, p_dropout, ctx_size, tril):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_size = head_size
        self.ctx_size = ctx_size
        self.tril = tril

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)  # 3 * d_model for Q, K, V
        self.output_proj_mha = nn.Linear(d_model, d_model)  # head_size * n_heads is just d_model

        self.attn_dropout = nn.Dropout(p_dropout)
        self.resid_dropout = nn.Dropout(p_dropout)


    def forward(self, x):


        B, T, C = x.shape

        qkv = self.qkv_proj(x)

        q, k, v = qkv.view(B, T, 3, self.n_heads, self.head_size).permute(2, 0, 3, 1, 4)

        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.output_proj_mha(out))
        return out, None


class FeedForward(nn.Module):
    def __init__(self, d_model, p_dropout):
        super().__init__()

        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.geluActivation = nn.GELU()
        self.feed_forward_lay_second = nn.Linear(4 * d_model, d_model)  # This is the one we need to scale
        self.dropout_layer = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.geluActivation(x)
        x = self.feed_forward_lay_second(x)
        x = self.dropout_layer(x)
        return x


class TransBlock(nn.Module):
    def __init__(self, d_model, n_heads, p_dropout, ctx_size, tril):
        super().__init__()

        head_size = d_model // n_heads

        self.sa = OptimizedMultiHeadAttention(n_heads=n_heads,
                                              head_size=head_size,
                                              d_model=d_model,
                                              p_dropout=p_dropout,
                                              ctx_size=ctx_size,
                                              tril=tril)
        # print(f"Using regular MHA")
        # self.sa = MultiHeadAttention(n_heads=n_heads,
        #                              head_size=head_size,
        #                              d_model=d_model,
        #                              p_dropout=p_dropout,
        #                              ctx_size=ctx_size,
        #                              tril=tril)

        self.ffwd = FeedForward(d_model=d_model,
                                p_dropout=p_dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        mha_output, _ = self.sa(self.ln1(x))
        x = x + mha_output
        x = x + self.ffwd(self.ln2(x))
        return x, None


    def inference(self, x, kv_cache):
        mha_output, new_kv_cache = self.sa.inference(self.ln1(x), kv_cache)
        x = x + mha_output
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv_cache


class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size, token_embd_weights):
        super().__init__()
        self.final_norm = nn.LayerNorm(d_model)  # Common to have a final LayerNorm
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = token_embd_weights

    def forward(self, x):
        x = self.final_norm(x)
        return self.lm_head(x)


class MyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ctx_size, p_dropout):
        super().__init__()

        self.register_buffer('tril', torch.tril(torch.ones(ctx_size, ctx_size)))

        self.initial_std = d_model ** -0.5

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(ctx_size, d_model)
        self.dropout = nn.Dropout(p_dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransBlock(d_model=d_model,
                        n_heads=n_heads,
                        p_dropout=p_dropout,
                        ctx_size=ctx_size,
                        tril=self.tril) for _ in range(n_layers)]
        )

        self.lm_head = LMHead(d_model, vocab_size, self.token_embedding.embedding.weight)

        self.apply(self._init_default_weights)
        self._apply_scaled_residual_initialization(n_layers)




    def _init_default_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.initial_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.initial_std)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _apply_scaled_residual_initialization(self, n_layers):
        # Scaling sub-layer outputs before adding to residual, so residuals
        # hold more power at the earlier stages of training

        scale_factor = math.sqrt(2.0 * n_layers)
        scaled_std = self.initial_std / scale_factor

        for block in self.transformer_blocks:
            if hasattr(block.sa, 'output_proj_mha') and isinstance(block.sa.output_proj_mha, nn.Linear):
                torch.nn.init.normal_(block.sa.output_proj_mha.weight, mean=0.0, std=scaled_std)
                if block.sa.output_proj_mha.bias is not None:
                    torch.nn.init.zeros_(block.sa.output_proj_mha.bias)
            else:
                print(f"Warning: Could not find 'output_projection' in TransBlock's SelfAttention.")

            if hasattr(block.ffwd, 'feed_forward_lay_second') and isinstance(block.ffwd.feed_forward_lay_second,
                                                                             nn.Linear):
                torch.nn.init.normal_(block.ffwd.feed_forward_lay_second.weight, mean=0.0, std=scaled_std)
                if block.ffwd.feed_forward_lay_second.bias is not None:
                    torch.nn.init.zeros_(block.ffwd.feed_forward_lay_second.bias)
            else:
                print(f"Warning: Could not find 'feed_forward_lay_second' in TransBlock's FeedForward.")

    def forward_embd_layer(self, input_ids):
        batch_size, ctx_len = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        pos_emb = self.positional_embedding(ctx_len, batch_size, device)  # [batch_size, seq_len, d_model]

        x = tok_emb + pos_emb
        return self.dropout(x)

    def forward_lm_head_layer(self, x):
        logits = self.lm_head(x)
        return logits

    def forward_with_attn_for_distill(self, input_ids):

        x = self.forward_embd_layer(input_ids)
        attns_per_block = []

        for i, block in enumerate(self.transformer_blocks):
            x, attn_scores = block(x)
            attns_per_block.append(attn_scores)

        return self.forward_lm_head_layer(x), attns_per_block

    def forward_with_out_hidd_for_distill(self, input_ids):
        x = self.forward_embd_layer(input_ids)

        hidd_states_per_block = []

        for i, block in enumerate(self.transformer_blocks):
            x, _ = block(x)
            hidd_states_per_block.append(x)

        return self.forward_lm_head_layer(x), hidd_states_per_block

    def forward(self, input_ids):

        x = self.forward_embd_layer(input_ids)

        for block in self.transformer_blocks:
            x, _ = block(x)

        return self.forward_lm_head_layer(x)


    def inference_step(self, input_ids, kv_caches=None):
        x = self.forward_embd_layer(input_ids)

        if kv_caches is None:
            kv_caches = [None] * len(self.transformer_blocks)

        new_kv_caches = []
        for i, block in enumerate(self.transformer_blocks):
            x, new_layer_cache = block.inference(x, kv_caches[i])
            new_kv_caches.append(new_layer_cache)

        logits = self.forward_lm_head_layer(x)
        return logits, new_kv_caches



import os
if __name__ == '__main__':
    # Define model parameters
    vocab_size = 5000
    d_model = 1024
    n_heads = 8
    n_layers = 9
    ctx_size = 1024  # Using a larger context size to see the impact
    p_dropout = 0.1

    # Instantiate the model
    model = MyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ctx_size=ctx_size,
        p_dropout=p_dropout,
    )

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- Memory analysis of 'tril' buffers ---
    total_tril_memory = 0
    print("--- Analyzing 'tril' buffer memory ---")
    for name, buffer in model.named_buffers():
        if 'tril' in name:
            tril_memory = buffer.element_size() * buffer.nelement()
            total_tril_memory += tril_memory
            print(f"Buffer '{name}':")
            print(f"  - Shape: {buffer.shape}")
            print(f"  - Memory: {tril_memory / (1024**2):.4f} MB")

    print(f"\nTotal memory occupied by 'tril' buffers: {total_tril_memory / (1024**2):.4f} MB")

    # --- Checkpoint size on disk analysis ---
    checkpoint_path = "temp_checkpoint.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    # Get the size of the saved checkpoint file
    checkpoint_size_bytes = os.path.getsize(checkpoint_path)
    checkpoint_size_mb = checkpoint_size_bytes / (1024**2)

    print("\n--- Analyzing checkpoint size on disk ---")
    print(f"Checkpoint (model + optimizer state) saved to '{checkpoint_path}'")
    print(f"Total checkpoint size on disk: {checkpoint_size_mb:.4f} MB")

    # Clean up the saved model file
    os.remove(checkpoint_path)

    # --- Insights ---
    print("\n--- Insights ---")
    if total_tril_memory > 0:
        tril_percentage = (total_tril_memory / checkpoint_size_bytes) * 100
        print(f"The 'tril' buffers make up approximately {tril_percentage:.2f}% of the total checkpoint size.")
        print("\nBy saving the optimizer state, the total file size has increased significantly.")
        print("This is because optimizers like Adam store additional information (like momentum and variance estimates) for each model parameter, effectively doubling the storage required for parameters.")
        print("\nConclusion:")
        print(" - For deploying a model for INFERENCE: Save only `model.state_dict()` to keep the file small.")
        print(" - For saving a checkpoint to RESUME TRAINING: You MUST save the `optimizer.state_dict()` as well, despite the larger file size.")
        print("\nYour suspicion about 'tril' buffers is still correct; they contribute to the model's state size. However, the optimizer state is often an even larger contributor to the overall checkpoint size.")

    else:
        print("No 'tril' buffers were found in the model. If you were expecting them,")
        print("check if the `new_system` flag is set correctly and if the model architecture is as intended.")