import torch
import torch.nn as nn
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
        return self.embedding(token_ids) * math.sqrt(self.d_model) # Scaling by sqrt(d_model) is common

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        # Alternative: Sinusoidal positional encoding (more complex to implement initially)
        # For simplicity, using a learned positional embedding here.

    def forward(self, seq_len, batch_size, device):
        """
        Args:
            seq_len: Length of the input sequences
            batch_size: Number of sequences in the batch
            device: The device tensors are on
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0) # [1, seq_len]
        positions_embedded = self.pos_embedding(positions) # [1, seq_len, d_model]
        return positions_embedded.repeat(batch_size, 1, 1) # [batch_size, seq_len, d_model]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of each head's key/query/value

        # For simplicity in this initial stub, we'll just use a single linear layer
        # to "process" the input, rather than full Q,K,V projections and attention.
        # This allows the overall architecture to be connected.
        self.dummy_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Tensor of shape [batch_size, seq_len_q, d_model]
            key: Tensor of shape [batch_size, seq_len_k, d_model]
            value: Tensor of shape [batch_size, seq_len_v, d_model] (seq_len_k == seq_len_v)
            mask: Optional attention mask. For a decoder, this would be a causal mask.
                  Shape [batch_size, 1, seq_len_q, seq_len_k] or [batch_size, num_heads, seq_len_q, seq_len_k]
        Returns:
            Tensor of shape [batch_size, seq_len_q, d_model]
        """
        # In a real implementation:
        # 1. Project Q, K, V to multiple heads
        # 2. Compute scaled dot-product attention for each head
        # 3. Concatenate head outputs
        # 4. Final linear projection
        # For now, just pass query through a linear layer to keep it simple.
        # We assume query, key, and value are the same for self-attention.
        batch_size = query.size(0)
        seq_len_q = query.size(1)

        # This is a placeholder. A real MHA is much more complex.
        out = self.dummy_linear(query)
        out = self.dropout(out)
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU() # Or GeLU, etc.
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        # Missing: Residual connections for full functionality.
        # We'll add them when fleshing out. For now, just sequential flow.

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask for self-attention.
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Simplified: No residuals or proper pre/post-norm yet.
        # x_norm1 = self.norm1(x) # Pre-norm or Post-norm
        attn_output = self.self_attn(x, x, x, mask) # Query, Key, Value are the same for self-attn
        # x = x + self.dropout1(attn_output) # Post-norm residual
        # x_norm2 = self.norm2(x)
        ff_output = self.feed_forward(attn_output) # Should operate on x_norm2 if pre-norm, or x if post-norm with residual
        # x = x + self.dropout2(ff_output) # Post-norm residual
        return ff_output # Returning ff_output directly for extreme simplicity now.

class LanguageModelHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model] (final hidden states)
        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size]
        """
        return self.linear(x)

class MyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len, d_ff_factor=4, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        d_ff = d_model * d_ff_factor # Dimension of the feed-forward hidden layer

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model) # Common to have a final LayerNorm
        self.lm_head = LanguageModelHead(d_model, vocab_size)

        # Optional: Weight tying between token embedding and LM head
        # self.lm_head.linear.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Token IDs, tensor of shape [batch_size, seq_len]
            attention_mask: Optional mask for attention (e.g., for padding or causal).
                            For a decoder, this would be a causal mask.
                            Shape: [batch_size, 1, seq_len, seq_len] for PyTorch MHA.
        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding(input_ids)     # [batch_size, seq_len, d_model]
        pos_emb = self.positional_embedding(seq_len, batch_size, device) # [batch_size, seq_len, d_model]

        x = tok_emb + pos_emb # Add token and positional embeddings
        x = self.dropout(x)

        # For a decoder-only model, we need a causal mask.
        # Let's create a simple one if not provided.
        if attention_mask is None:
            # Create a causal mask: [seq_len, seq_len]
            # For a target sequence of length T, the (i, j) entry of the mask is -inf if j > i and 0 otherwise.
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
            # PyTorch MHA expects mask: (N, L, S) where N is batch_size, L is target sequence length, S is source sequence length.
            # Or (N, num_heads, L, S). For self-attention L=S.
            # It adds the mask to the attention scores.
            # For our simplified MHA, we are not using it yet, but it's good to have the structure.
            # The dummy MHA doesn't use the mask, but a real one would.
            # The mask for MHA typically needs to be [batch_size, num_heads, seq_len, seq_len] or broadcastable.
            # For now, we'll pass it as [batch_size, 1, seq_len, seq_len] if we were to use it.
            # For the sake of this example running, we'll ensure our dummy MHA can accept it.
            # If attention_mask is not None, it should be correctly shaped.

        for block in self.transformer_blocks:
            x = block(x, mask=attention_mask) # Pass the mask to each block

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

    model = MyTransformerLM(
        vocab_size=vocab_size_test,
        d_model=d_model_test,
        num_heads=num_heads_test,
        num_layers=num_layers_test,
        max_seq_len=max_seq_len_test
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
    output_logits = model(dummy_input_ids, attention_mask=None)
    print(f"Output logits shape: {output_logits.shape}")
    assert output_logits.shape == (batch_size_test, max_seq_len_test, vocab_size_test)
    print("\nModel structure seems callable.")