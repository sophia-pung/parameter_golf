import torch                        # core PyTorch library for tensors and autograd
import torch.nn as nn                # neural network modules (Linear, Embedding, etc.)
import torch.nn.functional as F      # stateless functions like cross_entropy, normalize


# ── RoPE helpers ──────────────────────────────────────────────────────────────
# RoPE (Rotary Position Embedding) encodes position by rotating Q and K vectors
# inside each attention layer. It adds zero learned parameters, generalizes to
# sequence lengths longer than those seen during training, and is compatible with
# grouped-query / KV-cache setups used in modern efficient transformers.

def build_rope_cache(seq_len, dim, device):
    # dim here is the per-head dimension (d_model // n_heads)
    inv_freq = 1.0 / (                                 # classic RoPE formula: θ_i = 1 / 10000^(2i/d)
        10000 ** (torch.arange(0, dim, 2, device=device) / dim)
    )
    t = torch.arange(seq_len, device=device)           # integer position indices 0..T-1
    freqs = torch.outer(t, inv_freq)                   # (T, dim/2): each position × each frequency
    freqs = torch.cat([freqs, freqs], dim=-1)          # (T, dim): duplicate so even/odd slots alternate
    return freqs                                       # returned and reused every forward pass


def apply_rope(x, freqs):
    # x shape: (B, n_heads, T, head_dim) — query or key tensor
    cos = freqs.cos()[None, None, :, :]                # (1, 1, T, D) broadcast over batch and heads
    sin = freqs.sin()[None, None, :, :]                # (1, 1, T, D)
    x1 = x[..., : x.shape[-1] // 2]                   # first half of head dim
    x2 = x[..., x.shape[-1] // 2 :]                   # second half of head dim
    rotated = torch.cat([-x2, x1], dim=-1)             # 90-degree rotation in each 2D subspace
    return x * cos + rotated * sin                     # combine original and rotated components


# ── Stream B: Symbol Generator ─────────────────────────────────────────────────
# Stream B is a small auxiliary module that learns a separate, lower-dimensional
# embedding for each token and then projects it up to d_model. The intent is to
# capture structural / symbolic signals (morphological, syntactic) that the main
# embedding in Stream A may not naturally emphasize.

class SymbolGenerator(nn.Module):
    def __init__(self, vocab_size, d_model, d_sym=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_sym)   # small embedding table: V × d_sym
        self.proj = nn.Sequential(
            nn.Linear(d_sym, d_model, bias=False),     # lift d_sym → d_model
            nn.GELU(),                                 # smooth nonlinearity (standard in modern LLMs)
            nn.Linear(d_model, d_model, bias=False),   # mix features within d_model space
        )

    def forward(self, token_ids):
        sym = self.embed(token_ids)                    # (B, T, d_sym): low-dim symbolic features
        return self.proj(sym)                          # (B, T, d_model): projected to match Stream A


# ── Dual-stream merge ──────────────────────────────────────────────────────────
# Combines Stream A (standard dense embedding) and Stream B (symbolic features)
# into a single representation before the transformer. Concatenation followed by
# a linear projection is used so the model can learn any mixture of the two signals.

class DualStreamEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, d_sym=64):
        super().__init__()
        self.stream_a = nn.Embedding(vocab_size, d_model)      # Stream A: main soft embedding table
        self.stream_b = SymbolGenerator(vocab_size, d_model, d_sym)  # Stream B: symbolic auxiliary
        self.merge = nn.Linear(2 * d_model, d_model, bias=False)     # project concat → d_model

    def forward(self, ids):
        a = self.stream_a(ids)                         # (B, T, d_model): standard token embeddings
        b = self.stream_b(ids)                         # (B, T, d_model): symbolic features
        combined = torch.cat([a, b], dim=-1)           # (B, T, 2*d_model): concatenate both streams
        return self.merge(combined)                    # (B, T, d_model): learnable blend of A and B


# ── Transformer block ──────────────────────────────────────────────────────────
# Standard pre-norm transformer block with causal self-attention and an MLP.
# RoPE is applied to Q and K inside the attention step so the model understands
# token order without a dedicated positional embedding in the residual stream.

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads                         # number of attention heads
        self.head_dim = d_model // n_heads             # dimension per head
        self.norm1 = nn.RMSNorm(d_model)              # pre-norm before attention (RMSNorm = lighter LayerNorm)
        self.norm2 = nn.RMSNorm(d_model)              # pre-norm before MLP
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # project input → queries
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # project input → keys
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # project input → values
        self.o_proj = nn.Linear(d_model, d_model, bias=False)  # project attention output back to d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),       # expand to 4× for feed-forward width
            nn.GELU(),                                          # smooth activation
            nn.Linear(4 * d_model, d_model, bias=False),       # contract back to d_model
        )

    def forward(self, x, rope_freqs):
        B, T, D = x.shape                              # batch size, sequence length, model dim
        normed = self.norm1(x)                         # apply RMSNorm before attention (pre-norm)

        # Project to Q, K, V and split into heads
        q = self.q_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        v = self.v_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)

        q = apply_rope(q, rope_freqs)                  # rotate queries with positional frequencies
        k = apply_rope(k, rope_freqs)                  # rotate keys with positional frequencies

        # Scaled dot-product attention with causal mask (no future tokens visible)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, Dh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)      # merge heads → (B, T, D)
        x = x + self.o_proj(attn_out)                 # residual connection after attention

        x = x + self.mlp(self.norm2(x))               # residual connection after MLP (pre-norm)
        return x                                       # (B, T, D)


# ── Full model ─────────────────────────────────────────────────────────────────
# Ties together the dual-stream embedding, stacked transformer blocks, and the
# two output heads. The LM head shares weights with Stream A's embedding table
# (weight tying) to halve the vocabulary parameter cost. The JEPA head is a
# small projection used only during training and discarded before serialization.

class DualJEPAModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_sym=64):
        super().__init__()
        self.d_model = d_model                         # model hidden dimension
        self.n_heads = n_heads                         # number of attention heads
        self.head_dim = d_model // n_heads             # dimension per attention head

        self.embed = DualStreamEmbedding(vocab_size, d_model, d_sym)  # dual-stream input embedding
        self.blocks = nn.ModuleList(                   # stack of transformer blocks
            [Block(d_model, n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(d_model)               # final layer norm before output heads

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)     # next-token logits
        self.lm_head.weight = self.embed.stream_a.weight               # tie weights to Stream A embedding

        self.jepa_head = nn.Sequential(               # auxiliary head for JEPA training objective
            nn.Linear(d_model, d_model // 2, bias=False),             # bottleneck down to half dim
            nn.GELU(),                                                  # nonlinearity
            nn.Linear(d_model // 2, d_model, bias=False),             # project back to d_model space
        )

    def forward(self, ids, return_jepa=True):
        B, T = ids.shape                               # batch size and sequence length
        device = ids.device                            # keep everything on the same device

        x = self.embed(ids)                            # (B, T, d_model): merged dual-stream embeddings

        rope_freqs = build_rope_cache(T, self.head_dim, device)  # precompute RoPE for this sequence

        for block in self.blocks:                      # pass through each transformer layer
            x = block(x, rope_freqs)

        h = self.norm(x)                               # (B, T, d_model): normalized hidden states
        logits = self.lm_head(h)                       # (B, T, vocab_size): next-token prediction scores
        jepa_pred = self.jepa_head(h) if return_jepa else None  # only compute JEPA head when training

        return logits, h, jepa_pred                    # return all three for flexible use in loss fn
