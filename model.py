import torch
import torch.nn as nn
from torch.nn import functional as F

import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, token_ids):
        """
        Args:
            token_ids: Tensor of shape [batch_size, seq_len]
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        # TODO: investigate scaling necessity
        return self.embedding(token_ids) * math.sqrt(self.d_model) # Scaling by sqrt(d_model) is common

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, ctx_size, batch_size, device):
        """
        Args:
            seq_len: Length of the input sequences
            batch_size: Number of sequences in the batch
            device: The device tensors are on
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        # TODO: investigate if this is optimal
        positions = torch.arange(0, ctx_size, dtype=torch.long, device=device).unsqueeze(0) # [1, seq_len]
        positions_embedded = self.pos_embedding(positions) # [1, seq_len, d_model]
        return positions_embedded.repeat(batch_size, 1, 1) # [batch_size, seq_len, d_model]


class Head(nn.Module):
    def __init__(self, d_model, head_size, p_dropout, ctx_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(ctx_size, ctx_size)))
        self.dropout = nn.Dropout(p_dropout)
        # TODO: efficacy and correctness of this Head forward mechanism

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # TODO: Is F for softmax the best here?
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, d_model, p_dropout, ctx_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model = d_model,
                                         head_size = head_size,
                                         p_dropout = p_dropout,
                                         ctx_size = ctx_size) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, d_model)
        # TODO investigate where to put in dropout
        self.dropout = nn.Dropout(p_dropout)  # Use global dropout

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, p_dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            # TODO: probably use GELU
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(p_dropout)  # Use global dropout
        )

    def forward(self, x):
        return self.net(x)


class TransBlock(nn.Module):
    def __init__(self, d_model, n_heads, p_dropout, ctx_size):
        super().__init__()
        head_size = d_model // n_heads
        self.sa = MultiHeadAttention(n_heads = n_heads,
                                     head_size = head_size,
                                     d_model = d_model,
                                     p_dropout = p_dropout,
                                     ctx_size = ctx_size)
        self.ffwd = FeedForward(d_model = d_model,
                                p_dropout = p_dropout)
        # TODO: residuals and layer norms, their ordering
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ctx_size, p_dropout):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(ctx_size, d_model)
        self.dropout = nn.Dropout(p_dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransBlock(d_model = d_model,
                        n_heads = n_heads,
                        p_dropout = p_dropout,
                        ctx_size = ctx_size) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model) # Common to have a final LayerNorm
        self.lm_head = nn.Linear(d_model, vocab_size)

        # TODO: make embedding and de-embedding layers the same
        # TODO: apply custom initialization of weights (prob N(0, 0.02)

    def forward(self, input_ids):
        batch_size, ctx_len = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding(input_ids)     # [batch_size, seq_len, d_model]

        # TODO: In original script, arange was done here, but now it's in the PosEmbd module, investigate
        pos_emb = self.positional_embedding(ctx_len, batch_size, device) # [batch_size, seq_len, d_model]

        x = tok_emb + pos_emb # Add token and positional embeddings
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x) # Pass the mask to each block

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

if __name__ == '__main__':
    # Quick test of the model structure
    vocab_size_test = 100
    d_model_test = 32
    num_heads_test = 4
    num_layers_test = 2
    max_seq_len_test = 16
    batch_size_test = 2
    dropout_test = 0.1

    model = MyTransformerLM(
        vocab_size=vocab_size_test,
        d_model=d_model_test,
        n_heads=num_heads_test,
        n_layers=num_layers_test,
        ctx_size=max_seq_len_test,
        p_dropout=dropout_test
    )

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")

    dummy_input_ids = torch.randint(0, vocab_size_test, (batch_size_test, max_seq_len_test))
    print(f"\nInput IDs shape: {dummy_input_ids.shape}")

    # Create a dummy causal mask for testing
    # For a decoder, the mask ensures that a position can only attend to previous positions.
    # PyTorch's MultiheadAttention module expects a mask where `True` indicates a position *should not* be attended to.
    # Or it can accept additive masks. We'll create an additive mask.
    seq_len = dummy_input_ids.size(1)
    # For our dummy MHA, the mask argument is present but not used.
    # For a real MHA, it might look like this:
    # causal_mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    # causal_mask = causal_mask.float().masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))
    # causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size_test, num_heads_test, -1, -1)

    # Since our MHA is a stub, we can pass None or a simplified mask for now.
    # If you use nn.TransformerEncoderLayer, it can generate causal masks internally.
    # For now, we will pass None and let the model generate its basic causal mask (which is also not fully used by the dummy MHA)
    output_logits = model(dummy_input_ids)
    print(f"Output logits shape: {output_logits.shape}")
    assert output_logits.shape == (batch_size_test, max_seq_len_test, vocab_size_test)
    print("\nModel structure seems callable.")