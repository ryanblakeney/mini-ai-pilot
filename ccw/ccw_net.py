"""CCW-Net components: concept encoder, bias-free attention head, mediator flow,
alignment losses (Algorithm 1 pieces), zero-ablation attribution and Prop-1
self-checks.

Centerpiece paper: Rountree, Verma, So, Fan, Shah. "Causal Concept Explanations
for Deep Neural Models." NeurIPS 2026. arXiv 2603.21369.

Follows the paper's aircraft main-table config (Apps C.1, E): D=16 concept
slots, single-head bias-free attention wrapper; mediator = conditional affine
coupling flow (E.3). Zero-ablation attribution per Sec 5.4 / App D.3.

Drone adaptation: the wrapper output dim R is the SAC continuous action dim
(3 velocity setpoints, already tanh-squashed into the +/-1 m/s action box), so
the wrapper head output Y_r = tanh(...) matches the expert action range exactly.
Prop.1 semantics implemented: a zeroed slot is absent from the attention
context and contributes exactly nothing (zero contribution).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 16          # concept slot dimension (paper aircraft main-table config)
R = 3           # wrapper outputs: parameterized module constant.
                # Drone domain: R = 3 continuous action dims
                # [v_north, v_east, v_down]; head output tanh already
                # matches the +/-1 action box. (StS harness had R=32.)


def squash(v):
    """v: (..., D) -> slot with length r^2/(1+r^2) in [0,1); exactly 0 iff v == 0."""
    r2 = (v * v).sum(-1, keepdim=True)
    return v * (r2 / ((1.0 + r2) * (r2.sqrt() + 1e-12)))


class Encoder(nn.Module):
    """g_theta: 3-layer MLP ReLU hidden 64: Z(64) -> J slots of D dims, squashed."""

    def __init__(self, J, z_dim=64, hidden=64):
        super().__init__()
        self.J = J
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, J * D))

    def forward(self, z):
        return squash(self.net(z).view(-1, self.J, D))


class Head(nn.Module):
    """h_phi: per output r a learned query q_r; bias-free W_Q,W_K,W_V,W_O;
    s_rj = (W_Q q_r)^T (W_K M_j) / sqrt(D); softmax over present context slots;
    c_r = sum_j alpha_rj W_O W_V M_j; Y_r = tanh(w_r^T c_r + b_r).
    Default context = slots with nonzero norm (zero slot == absent concept)."""

    def __init__(self, J, R_=R):
        super().__init__()
        self.J = J
        self.R = R_
        self.q = nn.Parameter(torch.randn(R_, D) * 0.1)
        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        self.W_O = nn.Linear(D, D, bias=False)
        self.w = nn.Parameter(torch.randn(R_, D) * 0.1)
        self.b = nn.Parameter(torch.zeros(R_))

    def forward(self, M, ctx_mask=None):
        """M: (B,J,D); ctx_mask: (B,J) bool, True = slot in context."""
        if ctx_mask is None:
            ctx_mask = M.abs().sum(-1) > 0
        Q = self.W_Q(self.q)                       # (R,D)
        K = self.W_K(M)                            # (B,J,D)
        s = torch.einsum('rd,bjd->brj', Q, K) / math.sqrt(D)
        s = s.masked_fill(~ctx_mask.unsqueeze(1), -1e9)
        alpha = torch.softmax(s, dim=-1) * ctx_mask.unsqueeze(1)
        V = self.W_O(self.W_V(M))                  # (B,J,D)
        c = torch.einsum('brj,bjd->brd', alpha, V)
        return torch.tanh(torch.einsum('brd,rd->br', c, self.w) + self.b)


def W_OV(head):
    """matrix with W_O(W_V(M)) = M @ W_OV.T"""
    return head.W_O.weight @ head.W_V.weight


def singleton_deltas(head, M):
    """Delta0_j = h(0_{j<-M_j}) - h(0) in closed form (singleton ctx => alpha=1).
    Returns (B,R,J) and h0 = tanh(b) of shape (R,)."""
    WOV = W_OV(head)                               # W_O(W_V(M)) = M @ WOV.T
    MV = torch.einsum('bjd,ed->bje', M, WOV)      # (B,J,D)
    pre = torch.einsum('rd,bjd->brj', head.w, MV)  # (B,R,J)
    y_s = torch.tanh(pre + head.b[None, :, None])
    h0 = torch.tanh(head.b)
    return y_s - h0[None, :, None], h0


class CouplingLayer(nn.Module):
    """Affine coupling layer on half of D dims, conditioned on `cond`."""

    def __init__(self, cond_dim, dim=D, hidden=256, flip=False):
        super().__init__()
        self.flip = flip
        self.h = dim // 2
        self.net = nn.Sequential(
            nn.Linear(cond_dim + self.h, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * self.h))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _split(self, x):
        a, b = x[..., :self.h], x[..., self.h:]
        return (b, a) if self.flip else (a, b)

    def _merge(self, kept, moved):
        return torch.cat([moved, kept], -1) if self.flip else torch.cat([kept, moved], -1)

    def forward(self, x, cond):
        """data -> noise direction; returns (y, logdet_term).
        s is clamped for numerical stability (standard flow hygiene)."""
        a, b = self._split(x)
        p = self.net(torch.cat([a, cond], -1))
        s, t = (c.clamp(-8, 8) for c in p.chunk(2, -1))
        return self._merge(a, b * torch.exp(s) + t), s

    def inverse(self, y, cond):
        """noise -> data direction."""
        a, y_b = self._split(y)
        p = self.net(torch.cat([a, cond], -1))
        s, t = (c.clamp(-8, 8) for c in p.chunk(2, -1))
        return self._merge(a, (y_b - t) * torch.exp(-s))


class MediatorFlow(nn.Module):
    """F_psi (paper E.3): conditional normalizing flow over one concept slot.
    Conditioning = [Z, full label vector with c_j flipped, one-hot concept id].
    One flow conditioned on the concept id serves all concepts (functionally
    equivalent to J per-concept flows since the one-hot gates sharing; recorded
    as a documented deviation)."""

    def __init__(self, J, z_dim=64, n_layers=8, hidden=256):
        super().__init__()
        self.J = J
        self.layers = nn.ModuleList(
            [CouplingLayer(z_dim + 2 * J, D, hidden, flip=(i % 2 == 1))
             for i in range(n_layers)])

    def forward(self, x, cond):
        logdet = torch.zeros(x.shape[0], device=x.device)
        for l in self.layers:
            x, s = l(x, cond)
            logdet = logdet + s.sum(-1)
        return x, logdet

    def sample(self, n_per_row, cond_base, gen=None):
        """cond_base: (B, cond_dim); returns (B*n, D) samples, row-major
        (b*n + s) grouped, from p(x | cond)."""
        cond = cond_base.repeat_interleave(n_per_row, dim=0)
        z = torch.randn(cond.shape[0], D, generator=gen, device=cond.device)
        x = z
        for l in reversed(self.layers):
            x = l.inverse(x, cond)
        return torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)


def make_cond(z, c, j):
    """mediator conditioning for concept j with its label FLIPPED."""
    c2 = c.clone()
    c2[:, j] = 1.0 - c2[:, j]
    oh = torch.zeros_like(c2)
    oh[:, j] = 1.0
    return torch.cat([z, c2, oh], dim=-1)


def flow_nll(flow, x, cond):
    """mean NLL under standard normal base (summed dims)."""
    z, logdet = flow(x, cond)
    base = 0.5 * (z * z + math.log(2 * math.pi))
    nll = base.sum(-1) - logdet
    return nll.mean()


def flow_fit(flow, M_all, Z_all, C_all, J, epochs, batch, device, lr=1e-3, log_fn=None):
    """Fit mediator flow on frozen encoder representations M (N,J,D).
    Gradient clipping + best-epoch rollback guard against divergence on refits."""
    import copy
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    N = M_all.shape[0]
    g = torch.Generator(device=device)
    g.manual_seed(12345)
    flow.train()
    hist = []

    def _probe_nll():
        flow.eval()
        with torch.no_grad():
            part = []
            step = max(N // 4, 1)
            for st in range(0, N, step):
                idx = torch.arange(st, min(st + 512, N), device=device)
                tot = 0.0
                for j in range(J):
                    cond = make_cond(Z_all[idx], C_all[idx], j)
                    tot = tot + flow_nll(flow, M_all[idx, j], cond)
                part.append(float(tot) / J)
        flow.train()
        return sum(part) / len(part)

    best_nll = _probe_nll()  # guard: refits must not worsen the kept flow
    best_state = copy.deepcopy(flow.state_dict())
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g, device=device)
        tot = 0.0
        nb = 0
        for st in range(0, N, batch):
            idx = perm[st:st + batch]
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for j in range(J):
                cond = make_cond(Z_all[idx], C_all[idx], j)
                loss = loss + flow_nll(flow, M_all[idx, j], cond)
            (loss / J).backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
            nb += 1
        epnll = tot / max(nb, 1) / J
        hist.append(epnll)
        if epnll < best_nll:
            best_nll = epnll
            best_state = copy.deepcopy(flow.state_dict())
        if log_fn:
            log_fn(ep, epnll)
    flow.load_state_dict(best_state)
    if hist and hist[-1] > best_nll:
        hist[-1] = best_nll  # report the state we keep
    return hist


def zero_ablation_deltas(head, M, chunk=2048):
    """CUB-style zero-ablation (Sec 5.4 / App D.3):
    delta_j = ||h(M) - h(M with slot j zeroed/removed)||_2 for each concept j."""
    J = M.shape[1]
    was = head.training
    head.eval()
    out = []
    with torch.no_grad():
        for st in range(0, M.shape[0], chunk):
            Mc = M[st:st + chunk]
            y = head(Mc)
            d = torch.zeros(Mc.shape[0], J, device=M.device)
            for j in range(J):
                Mz = Mc.clone()
                Mz[:, j] = 0.0
                d[:, j] = (head(Mz) - y).norm(dim=-1)
            out.append(d)
    head.train(was)
    return torch.cat(out)


def prop1_checks(device='cpu'):
    """Mandatory Prop.1 unit checks. Raises AssertionError on failure."""
    torch.manual_seed(0)
    J = 14
    z_dim = 64  # must match MediatorFlow default conditioning Z width
    head = Head(J)  # R_ defaults to the module constant R
    # (i) all-zero M fed as two different batches -> constant output = tanh(b)
    o1, o2 = head(torch.zeros(5, J, D, device=device)), head(torch.zeros(11, J, D, device=device))
    const = torch.tanh(head.b)
    assert torch.allclose(o1[:1], o2[:1], atol=1e-6), "Prop1: all-zero batches differ"
    assert torch.allclose(o1[1:], o1[:1], atol=1e-6), "Prop1: all-zero batch not constant"
    assert torch.allclose(o1, const.unsqueeze(0).expand(5, R), atol=1e-5), \
        "Prop1: all-zero output != tanh(b)"
    # (ii) exactly-one-zero-slot == that slot deleted from context
    M = squash(torch.randn(7, J, D, device=device))
    for j in (0, 3, J - 1):
        Mz = M.clone()
        Mz[:, j] = 0.0
        ctx = torch.ones(7, J, device=device, dtype=torch.bool)
        ctx[:, j] = False
        assert torch.allclose(head(Mz), head(M, ctx_mask=ctx), atol=1e-5), \
            f"Prop1: zeroed slot {j} != deleted slot"
    # (iii) singleton closed form matches head forward on singleton context
    dj, h0 = singleton_deltas(head, M)
    j = 5
    Mone = torch.zeros(7, J, D, device=device)
    Mone[:, j] = M[:, j]
    assert torch.allclose(head(Mone), h0[None, :] + dj[:, :, j], atol=1e-5), \
        "Prop1: singleton delta closed form mismatch"
    # (iv) mediator flow: identity at init, finite sample
    flow = MediatorFlow(J)
    x = torch.randn(8, D)
    cond = torch.randn(8, z_dim + 2 * J)
    z, ld = flow(x, cond)
    assert torch.isfinite(z).all() and torch.isfinite(ld).all(), "flow forward NaN"
    assert torch.allclose(z, x, atol=1e-5), "flow init not identity"
    s = flow.sample(2, cond)
    assert torch.isfinite(s).all() and s.shape == (16, D), "flow sample bad"
    return float(const.mean().item())
