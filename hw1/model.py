from torch import nn
import torch
import torch.nn.functional as F

torch.manual_seed(0)


class RMSN(nn.Module):
    __eps = 1e-6

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * self.gamma / torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim=True) + self.__eps)
        return output


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim

        theta = 1. / (1e5 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)

        pos_theta = torch.outer(t, theta)
        pos_theta = torch.cat([pos_theta, pos_theta], dim=1)

        self.cos = pos_theta.cos()[None, None, :, :]
        self.sin = pos_theta.sin()[None, None, :, :]


    def forward(self, x):
        neg = torch.cat([-x[:, :, :, self.dim // 2:], x[:, :, :, :self.dim // 2]], dim=-1)
        cos = self.cos.to(x.device)
        sin = self.sin.to(x.device)
        return x * cos + neg * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.groups = n_heads // n_kv_heads
        self.head_dim = hidden_dim // n_heads
        self.dim_sqrt = self.head_dim ** 0.5
        self.wq = nn.Linear(hidden_dim, n_heads * self.head_dim)
        self.wk = nn.Linear(hidden_dim, n_kv_heads * self.head_dim)
        self.wv = nn.Linear(hidden_dim, n_kv_heads * self.head_dim)
        self.wo = nn.Linear(n_heads * self.head_dim, hidden_dim)
        self.rope = RoPE(self.head_dim, max_seq_len)

    def forward(self, x, mask):
        batch_size, seq_size = x.shape[0], x.shape[1]

        Q = self.wq(x).view(batch_size, seq_size, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(batch_size, seq_size, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(batch_size, seq_size, self.n_kv_heads, self.head_dim).transpose(1, 2)
        Q, K = self.rope(Q), self.rope(K)

        K = torch.repeat_interleave(K, repeats=self.groups, dim=1)
        V = torch.repeat_interleave(V, repeats=self.groups, dim=1)

        attn_scores = Q @ K.transpose(-2, -1) / self.dim_sqrt
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)
        context_vec = attn_weights @ V

        combined = context_vec.transpose(1, 2).contiguous().view(batch_size, seq_size, -1)
        return self.wo(combined)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim

        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Decoder(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention_norm = RMSN(dim)
        self.attention = GroupedQueryAttention(
            dim, 
            n_heads,
            n_kv_heads,
            max_seq_len
            )
        self.ffn = FeedForward(
            dim,
            hidden_dim=4 * dim
        )
        self.ffn_norm = RMSN(dim)

    def forward(self, x, mask):
        h = x + self.attention(self.attention_norm(x), mask)
        return h + self.ffn(self.ffn_norm(h))


class OutputBlock(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.dim = dim
        self.norm = RMSN(dim)
        
        self.w = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        return self.w(self.norm(x))


class LLamaModel(nn.Module):
    def __init__(self, dim, attn_params, decoder_blocks_num, vocab_size):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.attn_params = attn_params
        for i in range(decoder_blocks_num):
            decoder = Decoder(dim, 
                              attn_params.n_heads, 
                              attn_params.n_kv_heads,
                              attn_params.max_seq_len
            )
            self.layers.append(decoder)

        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.layers.append(OutputBlock(dim, vocab_size))

    def forward(self, x, targets):
        h = self.embeddings(x)
        seq_len = self.attn_params.max_seq_len
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        for layer in self.layers[:-1]:
            h = layer(h, mask)

        h = self.layers[-1](h)

        logits = h.view(-1, self.vocab_size)
        loss = F.cross_entropy(logits, targets.view(-1))
        return logits, loss
