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


class Head(nn.Module):
    def __init__(self, d_model, head_size, p_dropout, ctx_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(ctx_size, ctx_size)))
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # TODO: Is F for softmax the best here?
        wei = F.softmax(wei, dim=-1)

        attn_scores_for_distill = wei.clone()

        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out, attn_scores_for_distill


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, d_model, p_dropout, ctx_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model=d_model,
                                         head_size=head_size,
                                         p_dropout=p_dropout,
                                         ctx_size=ctx_size) for _ in range(n_heads)])
        self.output_proj_mha = nn.Linear(head_size * n_heads, d_model)
        self.dropout = nn.Dropout(p_dropout)  # Use global dropout

    def forward(self, x):

        head_outputs = [h(x) for h in self.heads]
        head_individual_outputs = [data[0] for data in head_outputs]
        head_individual_attn_scores = [data[1] for data in head_outputs]

        out = torch.cat(head_individual_outputs, dim=-1)
        out = self.dropout(self.output_proj_mha(out))

        stacked_attn_scores = torch.stack(head_individual_attn_scores, dim=1)

        return out, stacked_attn_scores


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
    def __init__(self, d_model, n_heads, p_dropout, ctx_size):
        super().__init__()
        head_size = d_model // n_heads
        self.sa = MultiHeadAttention(n_heads=n_heads,
                                     head_size=head_size,
                                     d_model=d_model,
                                     p_dropout=p_dropout,
                                     ctx_size=ctx_size)
        self.ffwd = FeedForward(d_model=d_model,
                                p_dropout=p_dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        mha_output, attn_scores = self.sa(self.ln1(x))
        x = x + mha_output
        x = x + self.ffwd(self.ln2(x))
        return x, attn_scores

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

        self.initial_std = d_model**-0.5

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(ctx_size, d_model)
        self.dropout = nn.Dropout(p_dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransBlock(d_model=d_model,
                        n_heads=n_heads,
                        p_dropout=p_dropout,
                        ctx_size=ctx_size) for _ in range(n_layers)]
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

            if hasattr(block.ffwd, 'feed_forward_lay_second') and isinstance(block.ffwd.feed_forward_lay_second, nn.Linear):
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



# if __name__ == '__main__':
#     # Quick test of the model structure
#     vocab_size_test = 100
#     d_model_test = 32
#     num_heads_test = 4
#     num_layers_test = 2
#     max_seq_len_test = 16
#     batch_size_test = 2
#     dropout_test = 0.1
#
#     model = MyTransformerLM(
#         vocab_size=vocab_size_test,
#         d_model=d_model_test,
#         n_heads=num_heads_test,
#         n_layers=num_layers_test,
#         ctx_size=max_seq_len_test,
#         p_dropout=dropout_test
#     )
#
#     print(model)
#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Model has {num_params:,} trainable parameters.")
#
#     dummy_input_ids = torch.randint(0, vocab_size_test, (batch_size_test, max_seq_len_test))
#     print(f"\nInput IDs shape: {dummy_input_ids.shape}")
#
#     seq_len = dummy_input_ids.size(1)
#     output_logits = model(dummy_input_ids)
#     print(f"Output logits shape: {output_logits.shape}")
#     assert output_logits.shape == (batch_size_test, max_seq_len_test, vocab_size_test)
#     print("\nModel structure seems callable.")
