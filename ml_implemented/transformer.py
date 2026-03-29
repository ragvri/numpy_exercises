import torch
from einops import einsum, rearrange
from torch import nn, Tensor
from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int = 1000):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, token_ids: Int[Tensor, "b seq"]) -> Float[Tensor, "b seq d"]:
        b, seq = token_ids.shape
        positions = torch.arange(seq, device=token_ids.device).unsqueeze(0)  # (1, seq)
        # b,s,d  + 1,s,d
        return self.token_embed(token_ids) + self.pos_embed(positions)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_in: int, d_output: int, is_causal: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_in = d_in
        self.d_output = d_output
        self.head_dim = d_output // self.n_heads
        self.is_causal = is_causal
        self.W_q = nn.Linear(in_features=d_in, out_features=d_output)
        self.W_k = nn.Linear(in_features=d_in, out_features=d_output)
        self.W_v = nn.Linear(in_features=d_in, out_features=d_output)
        self.linear = nn.Linear(in_features=d_output, out_features=d_output)

    def forward(self, X: Float[Tensor, "b seq d"]):
        b, seq, d = X.shape
        q = self.W_q(X)
        k = self.W_k(X)
        v = self.W_v(X)

        q = rearrange(q, "b s (n k) -> b n s k", n=self.n_heads)
        k = rearrange(k, "b s (n k) -> b n s k", n=self.n_heads)
        v = rearrange(v, "b s (n k) -> b n s k", n=self.n_heads)

        q_k_t = einsum(q, k, "b n i d, b n j d -> b n i j") / (
            (self.head_dim * 1.0) ** 0.5
        )

        if self.is_causal:
            # each q is only allowed to attend previous tokens so for all other scores make them -inf
            rows = torch.arange(seq)[:, None]  # (d, 1)
            cols = torch.arange(seq)[None, :]  # (1,d)
            mask = rows >= cols
            q_k_t[~mask] = -torch.inf

        scores = torch.softmax(q_k_t, dim=-1)

        final = einsum(scores, v, "b n i j, b n j d -> b n i d")
        final = rearrange(final, "b n i d -> b i (n d)")

        return self.linear(final)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        d_ff = int(8 / 3 * hidden_dim)

        self.W_g = nn.Linear(hidden_dim, d_ff)
        self.W_u = nn.Linear(hidden_dim, d_ff)
        self.W_d = nn.Linear(d_ff, hidden_dim)

    def silu(self, X):
        return X * torch.sigmoid(X)

    def forward(self, x: Float[Tensor, "b ... d"]):
        # (X sig(X))
        gate = self.silu(self.W_g(x))
        inner = gate * self.W_u(x)
        return self.W_d(inner)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(
            n_heads=num_heads, d_in=embed_dim, d_output=hidden_dim, is_causal=True
        )

        self.ln = nn.RMSNorm(normalized_shape=embed_dim)
        self.mlp = SwiGLU(hidden_dim)

    def forward(self, x: Float[Tensor, "b s d"]):
        """
        x: (Batch, Seq_Len, Embed_Dim)
        mask: (Seq_Len, Seq_Len) usually
        """
        residual = x
        norm1 = self.ln(x)
        x = self.attention(norm1) + residual

        residual = x
        norm2 = self.ln(x)
        x = self.mlp(norm2) + residual
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(embed_dim, num_heads, hidden_dim)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.RMSNorm(embed_dim)

    def forward(self, token_ids: Int[Tensor, "b seq"]) -> Float[Tensor, "b seq d"]:
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.ln_final(x)


if __name__ == "__main__":
    # Build a simple vocab from the input
    text = input().strip()
    words = text.split()
    vocab = {w: i for i, w in enumerate(sorted(set(words)))}
    vocab_size = len(vocab)

    token_ids = torch.tensor([[vocab[w] for w in words]])  # (1, seq_len)

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=64,
        num_heads=4,
        hidden_dim=64,
        num_layers=2,
    )
    model.eval()

    with torch.no_grad():
        output = model(token_ids)  # (1, seq_len, embed_dim)

    # Print space-separated representation for each token
    for i, word in enumerate(words):
        vec = output[0, i].numpy()
        formatted = " ".join(f"{v:.4f}" for v in vec)
        print(f"{word}: {formatted}")
