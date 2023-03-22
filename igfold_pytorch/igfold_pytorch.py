import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return self.fn(x) + x

class GatedResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.proj = nn.Sequential(
            nn.LazyLinear(1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, **kwargs):
        x_hat = self.fn(x, **kwargs)
        b = self.proj(torch.cat([x_hat, x, x_hat - x], dim=-1))
        return (1 - b) * x + b * x_hat
    
class FeedForward(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, x):
        return self.main(x)

class Attention(nn.Module):
    def __init__(self, d_orig=64, d_head=32, n_head=8):
        super().__init__()
        self.d_head = d_head
        self.n_head = n_head
        d_emb = d_head * n_head

        self.to_q = nn.Linear(d_orig, d_emb, bias=False)
        self.to_k = nn.Linear(d_orig, d_emb, bias=False)
        self.to_v = nn.Linear(d_orig, d_emb, bias=False)
        self.to_e = nn.Linear(d_orig, d_emb, bias=False)

        self.to_out = nn.Linear(d_emb, d_orig)

    def forward(self, x, e):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        e = self.to_e(e)

        q, k, v = map(lambda t: rearrange(t, 'b l (n d) -> b n l d', n=self.n_head), (q, k, v))
        e = rearrange(e, 'b i j (n d) -> b n i j d', n=self.n_head)

        k = rearrange(k, 'b n j d -> b n () j d') + e
        logit = einsum('b n i d, b n i j d -> b n i j', q, k) * (self.d_head ** -0.5)
        attn = logit.softmax(dim=-1)

        v = rearrange(v, 'b n j d -> b n () j d') + e
        out = einsum('b n i j, b n i j d -> b n i d', attn, v)
        out = rearrange(out, 'b n l d -> b l (n d)')
        out = self.to_out(out)
        return out
    
class GraphTransformer(nn.Module):
    def __init__(self, d_orig=64, d_head=32, n_head=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_orig)
        self.attn = GatedResidual(Attention(d_orig, d_head, n_head))
        self.ln2 = nn.LayerNorm(d_orig)
        self.ff = GatedResidual(FeedForward(d_orig))

    def forward(self, x, e):
        x = self.ln1(x)
        x = self.attn(x, e=e)
        x = self.ln2(x)
        x = self.ff(x)

        return x
    
class TriMult(nn.Module):
    def __init__(self, d, c, type='out'):
        super().__init__()
        self.ln_in = nn.LayerNorm(d)
        self.ln_out = nn.LayerNorm(c)
        
        self.left_proj = nn.Linear(d, c)
        self.right_proj = nn.Linear(d, c)
        self.out_proj = nn.Linear(c, d)

        self.left_gate = nn.Linear(d, c)
        self.right_gate = nn.Linear(d, c)
        self.out_gate = nn.Linear(d, d)

        # Initialization.
        # See https://github.com/deepmind/alphafold/blob/2e6a78f08b3a31945ad0fad77f7cf4344ae73d13/alphafold/model/modules.py#L1342
        for gate in [self.left_gate, self.right_gate, self.out_gate]:
            gate.weight.data.fill_(0.)
            gate.bias.data.fill_(1.)

        self.type = type

    def forward(self, x):
        x = self.ln_in(x)
        left = self.left_gate(x).sigmoid() * self.left_proj(x)
        right = self.right_gate(x).sigmoid() * self.right_proj(x)

        eq = 'bikc, bjkc -> bijc' if self.type == 'in' else 'bkic, bkjc -> bijc'
        out = einsum(eq, left, right)
        out = self.out_gate(x).sigmoid() * self.out_proj(self.ln_out(out))

        return out

class TriMultUpdate(nn.Module):
    def __init__(self, d_orig):
        super().__init__()

        self.main = nn.Sequential(
            Residual(TriMult(d_orig, d_orig * 2, type='out')),
            Residual(TriMult(d_orig, d_orig * 2, type='in')),
        )
    
    def forward(self, e):
        return self.main(e)

class IGFoldLayer(nn.Module):
    def __init__(self, d_orig=64, d_head=32, n_head=8):
        super().__init__()
        self.gn = GraphTransformer(d_orig, d_head, n_head)
        self.tm = TriMultUpdate(d_orig)
    
    def forward(self, x, e):
        x = self.gn(x, e)
        e = self.tm(e)
        return x, e

class IGFold(nn.Module):
    def __init__(self, d_orig=64, d_head=32, n_head=8, n_layers=4):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.LazyLinear(d_orig), nn.ReLU(), nn.LayerNorm(d_orig)
        )
        self.edge_proj = nn.Sequential(
            nn.LazyLinear(d_orig), nn.ReLU(), nn.LayerNorm(d_orig)
        )

        self.layers = nn.ModuleList([
            IGFoldLayer(d_orig, d_head, n_head) for _ in range(n_layers)
        ])

    def forward(self, x, e):
        x = self.node_proj(x)
        e = self.edge_proj(e)

        for layer in self.layers:
            x, e = layer(x, e)
        
        return x, e

if __name__ == '__main__':
    x = torch.randn([1, 128, 512])
    e = torch.randn([1, 128, 128, 512])

    model = IGFold()
    x, e = model(x, e)
    print(x.shape)
    print(e.shape)