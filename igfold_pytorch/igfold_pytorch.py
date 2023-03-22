import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

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

def euclidean_transform(x, r, t):
    """r: rotation matrix of size (b, l, 3, 3)
    t: translation vector of size (b, l, 3)
    """
    # infer number of heads
    n = x.size(1)
    r = repeat(r, 'b l x y -> b n l x y', n=n)
    t = repeat(t, 'b l x -> b n l () x', n=n)

    return einsum('b n l d k, b n l k c -> b n l d c', x, r) + t

def inverse_euclidean_transform(x, r, t):
    """r: rotation matrix of size (b, l, 3, 3)
    t: translation vector of size (b, l, 3)
    """
    # infer number of heads
    n = x.size(1)
    r_inv = repeat(r, 'b l x y  -> b n l y x', n=n)  # note that R^-1 = R^T
    t = repeat(t, 'b l x -> b n l () x', n=n)

    return einsum('b n l d k, b n l k c -> b n l d c', x - t, r_inv)
    
class InvariantPointAttention(nn.Module):
    def __init__(
            self,
            d_orig,
            d_head_scalar=16,
            d_head_point=4,
            n_head=8,
        ):
        super().__init__()
        self.n_head = n_head

        # standard self-attention (scalar attention)
        d_scalar = d_head_scalar * n_head
        self.to_q_scalar = nn.Linear(d_orig, d_scalar, bias=False)
        self.to_k_scalar = nn.Linear(d_orig, d_scalar, bias=False)
        self.to_v_scalar = nn.Linear(d_orig, d_scalar, bias=False)
        self.scale_scalar = d_head_scalar ** -0.5

        # modulation by pair representation
        self.to_pair_bias = nn.Linear(d_orig, n_head, bias=False)

        # point attention
        d_point = (d_head_point * 3) * n_head 
        self.to_q_point = nn.Linear(d_orig, d_point, bias=False)
        self.to_k_point = nn.Linear(d_orig, d_point, bias=False)
        self.to_v_point = nn.Linear(d_orig, d_point, bias=False)
        self.scale_point = (4.5 * d_head_point) ** -0.5
        self.gamma = nn.Parameter(torch.log(torch.exp(torch.ones(n_head)) - 1.0))

        self.to_out = nn.Linear(d_scalar + d_orig * n_head + d_point + (d_head_point * n_head), d_orig)

    def forward(self, x, e, r, t):
        q_scalar = self.to_q_scalar(x)
        k_scalar = self.to_k_scalar(x)
        v_scalar = self.to_v_scalar(x)

        q_scalar, k_scalar, v_scalar = map(
            lambda t: rearrange(t, 'b l (n d) -> b n l d', n=self.n_head),
            (q_scalar, k_scalar, v_scalar)
        )

        q_point = self.to_q_point(x)
        k_point = self.to_k_point(x)
        v_point = self.to_v_point(x)

        q_point, k_point, v_point = map(
            lambda t: rearrange(t, 'b l (n p c) -> b n l p c', n=self.n_head, c=3),
            (q_point, k_point, v_point)
        )

        q_point, k_point, v_point = map(
            lambda v: euclidean_transform(v, r, t),
            (q_point, k_point, v_point),
        )

        # standard self-attention (scalar attention)
        logit_scalar = einsum('b n i d, b n j d -> b n i j', q_scalar, k_scalar) * self.scale_scalar

        # modulation by pair representation
        bias_pair = rearrange(self.to_pair_bias(e), 'b i j n -> b n i j')

        # point attention
        all_pairwise_diff = rearrange(q_point, 'b n i p c -> b n i () p c') - rearrange(k_point, 'b n j p c -> b n () j p c')
        gamma = rearrange(self.gamma, 'n -> () n () ()')

        # note 18**-0.5 for w_c
        logit_point = -0.5 * 0.25 * gamma * (all_pairwise_diff**2).sum(dim=-1).sum(dim=-1)

        logit = (3**-0.5) * (logit_scalar + bias_pair + logit_point)
        attn = logit.softmax(dim=-1)

        out_scalar = einsum('b n i j, b n j d -> b n i d', attn, v_scalar)
        out_scalar = rearrange(out_scalar, 'b n i d -> b i (n d)')

        out_pair = einsum('b n i j, b i j d -> b n i d', attn, e)
        out_pair = rearrange(out_pair, 'b n i d -> b i (n d)')

        out_point = einsum('b n i j, b n j p c -> b n i p c', attn, v_point)
        out_point = inverse_euclidean_transform(out_point, r, t)
        out_point_norm = out_point.norm(dim=-1, keepdim=True)

        out_point = rearrange(out_point, 'b n i p c -> b i (n p c)')
        out_point_norm = rearrange(out_point_norm, 'b n i p c -> b i (n p c)')

        out = torch.cat([out_scalar, out_pair, out_point, out_point_norm], dim=-1)
        x = self.to_out(out)

        return x
    
class InvariantPointAttentionBlock(nn.Module):
    def __init__(
            self,
            d_orig=64,
            d_head_scalar=16,
            d_head_point=4,
            n_head=8,
        ):
        super().__init__()

        self.ipa = Residual(InvariantPointAttention(d_orig, d_head_scalar, d_head_point, n_head))
        self.ln1 = nn.LayerNorm(d_orig)
        self.ff = Residual(nn.Sequential(
            nn.Linear(d_orig, d_orig),
            nn.ReLU(),
            nn.Linear(d_orig, d_orig),
            nn.ReLU(),
            nn.Linear(d_orig, d_orig),
        ))
        self.ln2 = nn.LayerNorm(d_orig)

    def forward(self, x, e, r, t):
        x = self.ln1(self.ipa(x, e=e, r=r, t=t))
        x = self.ln2(self.ff(x))
        return x

class IGFold(nn.Module):
    def __init__(
            self,
            d_orig=64,
            d_head=32,
            n_head=8,
            n_graph_transformer_layers=4,
            n_ipa_temp_layers=2,
        ):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.LazyLinear(d_orig), nn.ReLU(), nn.LayerNorm(d_orig)
        )
        self.edge_proj = nn.Sequential(
            nn.LazyLinear(d_orig), nn.ReLU(), nn.LayerNorm(d_orig)
        )

        self.graph_transformer_layers = nn.ModuleList([
            IGFoldLayer(d_orig, d_head, n_head) for _ in range(n_graph_transformer_layers)
        ])

        self.ipa_temp_layers = nn.ModuleList([
            InvariantPointAttentionBlock(d_orig) for _ in range(n_ipa_temp_layers)
        ])

    def forward(self, x, e, r, t):
        x = self.node_proj(x)
        e = self.edge_proj(e)

        for layer in self.graph_transformer_layers:
            x, e = layer(x, e)
        
        for layer in self.ipa_temp_layers:
            x = layer(x, e, r, t)
        
        return x, e

if __name__ == '__main__':
    x = torch.randn([1, 128, 512])
    e = torch.randn([1, 128, 128, 512])
    r = torch.randn([1, 128, 3, 3])
    t = torch.randn([1, 128, 3])

    model = IGFold()
    x, e = model(x, e, r, t)
    print(x.shape)
    print(e.shape)

    # q_point = torch.randn([1, 8, 128, 4, 3])
    # k_point = torch.randn([1, 8, 128, 4, 3])

    # gamma = torch.randn(8)
    # g = repeat(F.softplus(gamma), 'n -> b n () ()', b=1)

    # all_pairwise_diff = rearrange(q_point, 'b n i p c -> b n i () p c') - rearrange(k_point, 'b n j p c -> b n () j p c')

    # dist = g * (all_pairwise_diff**2).sum(dim=-1).sum(dim=-2)
    # print(dist.sum())

    # g = rearrange(F.softplus(gamma), 'n -> () n () () ()')
    # dist = (all_pairwise_diff**2).sum(dim=-2)
    # dist = (g * dist).sum(dim=-1)
    # print(dist.sum())