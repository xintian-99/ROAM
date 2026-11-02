# ROAM: Region-Graph Optimal Transport Routing for Mixture-of-Experts WSI Classification
# Xin Tian et al., IEEE BIBM 2026 | https://arxiv.org/abs/2604.07298 | https://github.com/xintian-99/ROAM
"""ROAM: Region-graph OptimAl-transport Mixture-of-experts for WSI classification.

Pipeline (paper Sec. II):
    1. Region tokenisation   N patches -> M <= n_tokens region tokens by grid binning
                             (16x16 grid for n_tokens=256) after scaling each
                             coordinate axis to [0, 1]; empty bins are dropped.
    2. Region graph          kNN graph over region centroids.
    3. Routing GNN           GATv2 on the region graph -> routing embeddings Z
                             (used for routing only; experts pool the region features H0).
    4. Transport cost        C[m, e] = 1 - cos(z_m, mu_e).
    5. Capacity-constrained  supply r_m ∝ A_m (1 + lam_rare * u_m), capacity q_e = 1/E,
       Sinkhorn OT           graph regularisation of log Pi between Sinkhorn blocks (Eq. 2).
    6. Sparse dispatch       top-k experts per region, rescaled to keep the row mass (Eq. 3).
    7. Expert pooling        per-expert gated attention with + log(gamma + delta) (Eq. 4).
    8. Fusion + classifier   gated attention over experts, concatenated with the
                             routing-profile embedding eta(gamma), then an MLP head.

Interface:
    forward(features, coords) -> (logits, patch_saliency)

    patch_saliency is routing-derived (sum_e gamma[region(i), e] * gate[e]), not
    ABMIL-style attention.

Diagnostics stored in self._diag after each forward pass:
    Pi           (M, E) dense transport plan (column sums ≈ 1/E)
    Alpha        (M, E) sparse dispatch gamma (top-k per row)
    region_id    (N,)   patch -> region index
    expert_gate  (E, 1) fusion weights
    rarity       (M,)   u_m
    expert_z     (E, d) expert outputs
    neighbors    (M, k) region kNN indices
    centroids    (M, 2) region centroids
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════
#  Step 1: Region tokenisation (grid binning)
# ═════════════════════════════════════════════════════════════════════


def _build_regions(coords, features, target_M):
    """Bin N patches into at most target_M spatial regions on a square grid.

    Args:
        coords:   (N, 2) patch coordinates, or None.
        features: (N, d) projected patch features.
        target_M: maximum number of regions (grid side = ceil(sqrt(target_M))).

    Returns:
        region_id: (N,)   patch -> region index (contiguous, 0..M-1).
        H0:        (M, d) mean region features.
        A:         (M,)   region size (patch count).
        C_cent:    (M, 2) region centroids.
    """
    N, d = features.shape
    dev = features.device

    if N <= target_M:
        # Small bag: every patch is its own region.
        c = coords.float() if coords is not None else torch.zeros(N, 2, device=dev)
        return torch.arange(N, device=dev), features.clone(), torch.ones(N, device=dev), c

    if coords is not None:
        coords_f = coords.float().to(dev)
        grid_res = max(int(math.ceil(math.sqrt(target_M))), 2)
        cmin = coords_f.min(dim=0).values
        span = (coords_f.max(dim=0).values - cmin).clamp(min=1.0)
        norm = (coords_f - cmin) / span                       # [0, 1] per axis
        gc = (norm * (grid_res - 1e-6)).long().clamp(0, grid_res - 1)
        region_id = gc[:, 0] * grid_res + gc[:, 1]
    else:
        # No coordinates: contiguous chunks in file order.
        coords_f = torch.zeros(N, 2, device=dev)
        region_id = (torch.arange(N, device=dev) * target_M) // N

    # Drop empty bins: compact to contiguous IDs
    _, region_id = region_id.unique(return_inverse=True)
    M = int(region_id.max().item()) + 1

    H0 = torch.zeros(M, d, device=dev)
    A = torch.zeros(M, device=dev)
    C_cent = torch.zeros(M, 2, device=dev)

    H0.scatter_add_(0, region_id.unsqueeze(1).expand_as(features), features)
    A.scatter_add_(0, region_id, torch.ones(N, device=dev))
    C_cent.scatter_add_(0, region_id.unsqueeze(1).expand(N, 2), coords_f)

    A_safe = A.clamp(min=1.0)
    H0 = H0 / A_safe.unsqueeze(1)
    C_cent = C_cent / A_safe.unsqueeze(1)

    return region_id, H0, A, C_cent


# ═════════════════════════════════════════════════════════════════════
#  Step 2: Region graph (kNN on centroids)
# ═════════════════════════════════════════════════════════════════════


def _knn_graph(C, k):
    """kNN graph on region centroids. Returns (M, k) neighbour indices."""
    M = C.shape[0]
    k = min(k, M - 1)
    if k <= 0:
        return torch.zeros(M, 1, dtype=torch.long, device=C.device)
    dist = torch.cdist(C.unsqueeze(0), C.unsqueeze(0)).squeeze(0)
    dist.fill_diagonal_(float("inf"))
    _, neighbors = dist.topk(k, dim=1, largest=False)
    return neighbors


# ═════════════════════════════════════════════════════════════════════
#  Step 3: Routing GNN (GATv2, routing only)
# ═════════════════════════════════════════════════════════════════════


class GATLayer(nn.Module):
    """GATv2 multi-head graph attention layer over a fixed kNN neighbourhood.

    GATv2: Brody, Alon and Yahav, "How attentive are graph attention networks?",
    ICLR 2022 (https://arxiv.org/abs/2105.14491). Written here in plain PyTorch.

        e_ij^h = a_h^T · LeakyReLU(W_q·h_i + W_k·h_j)
        α_ij^h = softmax_j(e_ij^h)                       (over the k neighbours)
        h'_i   = concat_h [ Σ_j α_ij^h · W_v^h · h_j ]

    Each head works on a d/n_heads subspace; the concatenated output goes through
    LayerNorm + ReLU + Dropout.
    """

    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.W_q = nn.Linear(d, d, bias=False)       # query (self)
        self.W_k = nn.Linear(d, d, bias=False)       # key   (neighbour)
        self.W_v = nn.Linear(d, d, bias=False)       # value (neighbour)
        self.attn_vec = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, neighbors):
        """h: (M, d) region embeddings; neighbors: (M, k). Returns (M, d)."""
        M, d = h.shape
        H, dh = self.n_heads, self.d_head
        k = neighbors.shape[1]

        q = self.W_q(h).view(M, H, dh)
        kk = self.W_k(h).view(M, H, dh)
        v = self.W_v(h).view(M, H, dh)

        kk_j = kk[neighbors]                                  # (M, k, H, dh)
        v_j = v[neighbors]                                    # (M, k, H, dh)

        q_exp = q.unsqueeze(1).expand(-1, k, -1, -1)          # (M, k, H, dh)
        e = F.leaky_relu(q_exp + kk_j, negative_slope=0.2)
        e = (e * self.attn_vec).sum(-1)                       # (M, k, H)
        attn = F.softmax(e, dim=1)                            # over neighbours

        out = (attn.unsqueeze(-1) * v_j).sum(dim=1).reshape(M, d)
        return self.drop(F.relu(self.ln(out)))


class RoutingGAT(nn.Module):
    """Stack of GATv2 layers with residual connections -> routing embeddings Z."""

    def __init__(self, d, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GATLayer(d, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, h, neighbors):
        z = h
        for layer in self.layers:
            z = layer(z, neighbors) + z
        return z


# ═════════════════════════════════════════════════════════════════════
#  Step 4: Rarity score u_m (enters the supply marginal only)
# ═════════════════════════════════════════════════════════════════════


def _compute_rarity(Z, neighbors):
    """Mean Euclidean distance to kNN routing embeddings, min-max scaled to [0, 1]."""
    dists = (Z.unsqueeze(1) - Z[neighbors]).norm(dim=-1).mean(dim=1)
    dmin, dmax = dists.min(), dists.max()
    if dmax - dmin < 1e-8:
        return torch.zeros_like(dists)
    return (dists - dmin) / (dmax - dmin)


# ═════════════════════════════════════════════════════════════════════
#  Step 5: Capacity-constrained Sinkhorn with graph regularisation
# ═════════════════════════════════════════════════════════════════════


def _sinkhorn_basic(C, mu, nu, eps=0.1, iters=20):
    """Standard log-domain Sinkhorn (used only when the region graph is empty).

    Cuturi, "Sinkhorn distances: Lightspeed computation of optimal transport",
    NeurIPS 2013.
    """
    logK = -C / eps
    lmu, lnu = torch.log(mu + 1e-8), torch.log(nu + 1e-8)
    u, v = torch.zeros_like(lmu), torch.zeros_like(lnu)
    for _ in range(iters):
        v = lnu - torch.logsumexp(logK + u.unsqueeze(1), dim=0)
        u = lmu - torch.logsumexp(logK + v.unsqueeze(0), dim=1)
    return torch.exp(u.unsqueeze(1) + logK + v.unsqueeze(0))


def _graph_sinkhorn(C_cost, r, q, neighbors, eps=0.1, iters=20,
                    smooth_iters=3, lam_smooth=0.1):
    """Sinkhorn OT with graph regularisation of the log-plan (paper Eq. 1-2).

    Runs max(iters - 3*smooth_iters, iters // 2) plain Sinkhorn updates, then
    smooth_iters blocks of [log Pi <- (1-lam) log Pi + lam * mean_kNN(log Pi),
    followed by 3 Sinkhorn re-projections]. With iters=20, smooth_iters=3 this is
    11 + 3*3 = 20 updates. smooth_iters=0 (or lam_smooth=0) is plain Sinkhorn.
    """
    logK = -C_cost / eps
    lr, lq = torch.log(r + 1e-8), torch.log(q + 1e-8)
    u, v = torch.zeros_like(lr), torch.zeros_like(lq)

    base_iters = max(iters - smooth_iters * 3, iters // 2)
    for _ in range(base_iters):
        v = lq - torch.logsumexp(logK + u.unsqueeze(1), dim=0)
        u = lr - torch.logsumexp(logK + v.unsqueeze(0), dim=1)

    for _ in range(smooth_iters):
        log_Pi = u.unsqueeze(1) + logK + v.unsqueeze(0)
        smooth_log_Pi = log_Pi[neighbors].mean(dim=1)      # uniform 1/k weights
        log_Pi = (1 - lam_smooth) * log_Pi + lam_smooth * smooth_log_Pi

        logK_s = log_Pi - u.unsqueeze(1) - v.unsqueeze(0)
        for _ in range(3):
            v = lq - torch.logsumexp(logK_s + u.unsqueeze(1), dim=0)
            u = lr - torch.logsumexp(logK_s + v.unsqueeze(0), dim=1)
        logK = logK_s

    return torch.exp(u.unsqueeze(1) + logK + v.unsqueeze(0))


# ═════════════════════════════════════════════════════════════════════
#  Step 6: Sparse dispatch
# ═════════════════════════════════════════════════════════════════════


def _topk_dispatch(Pi, k=2):
    """Keep the top-k experts per region and rescale them to the row mass (Eq. 3)."""
    M, E = Pi.shape
    k = min(k, E)
    vals, idx = Pi.topk(k, dim=1)
    row_sums = Pi.sum(dim=1, keepdim=True)
    vals = vals * (row_sums / (vals.sum(dim=1, keepdim=True) + 1e-8))
    Alpha = torch.zeros_like(Pi)
    Alpha.scatter_(1, idx, vals)
    return Alpha


# ═════════════════════════════════════════════════════════════════════
#  Step 7: Per-expert OT-guided gated attention pooling
# ═════════════════════════════════════════════════════════════════════


class ExpertAggregator(nn.Module):
    """Multi-head gated-attention pooling for one expert (paper Eq. 4).

    The gated attention (V, U, W) is the form of Ilse, Tomczak and Welling,
    "Attention-based deep multiple instance learning", ICML 2018, and of their code
    https://github.com/AMLab-Amsterdam/AttentionDeepMIL. ROAM adds the multi-head
    subspace pooling and the log(gamma) routing bias.

        score_h = W_h · (tanh(V h0) ⊙ sigmoid(U h0))
        beta_h  = softmax_m(score_h + log(gamma_e + 1e-8))
        o_e     = concat_h [ Σ_m beta_h(m) · h0_m[subspace h] ]

    Each head scores the full region features but pools its own d/heads subspace.
    Regions outside the expert's support have gamma = 0, i.e. a -18.4 logit offset.
    """

    def __init__(self, d, heads=2, d_att=64, dropout=0.25, use_ot_modulation=True,
                 ot_mod_detach=False):
        super().__init__()
        self.heads = heads
        self.hd = d // heads
        self.use_ot_modulation = use_ot_modulation
        self.ot_mod_detach = ot_mod_detach

        self.V = nn.Sequential(nn.Linear(d, d_att), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(d, d_att), nn.Sigmoid())
        self.W = nn.Linear(d_att, heads)
        self.drop = nn.Dropout(dropout)

    def forward(self, H0, weights):
        """H0: (M, d) region features; weights: (M,) dispatch gamma[:, e]. Returns (d,)."""
        a = self.W(self.V(H0) * self.U(H0))                  # (M, H)
        if self.use_ot_modulation:
            w = weights.detach() if self.ot_mod_detach else weights
            a = a + torch.log(w + 1e-8).unsqueeze(-1)
        a = self.drop(F.softmax(a, dim=0))                    # (M, H)

        t = H0.view(-1, self.heads, self.hd)                  # (M, H, d/H)
        return (a.unsqueeze(-1) * t).sum(0).reshape(-1)       # (d,)


# ═════════════════════════════════════════════════════════════════════
#  Step 8: Expert fusion
# ═════════════════════════════════════════════════════════════════════


class ExpertFusion(nn.Module):
    """Gated attention over the E expert outputs (same form as Ilse et al., ICML 2018,
    https://github.com/AMLab-Amsterdam/AttentionDeepMIL)."""

    def __init__(self, d, d_att=128):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(d, d_att), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(d, d_att), nn.Sigmoid())
        self.W = nn.Linear(d_att, 1)

    def forward(self, Y):
        g = self.W(self.V(Y) * self.U(Y))
        g = F.softmax(g, dim=0)
        return (g * Y).sum(dim=0), g


# ═════════════════════════════════════════════════════════════════════
#  ROAM
# ═════════════════════════════════════════════════════════════════════


class ROAM(nn.Module):
    """Region-graph OT-routed mixture of attention-pooling experts.

    Args:
        d_in:          input patch-feature dim (1536 for UNI2-h and GigaPath).
        d_proj:        working dim d.
        d_att:         hidden dim of the expert-fusion gate and classifier; the
                       per-expert attention dim is max(d_att // 4, 32) (= 64).
        n_classes:     1 for binary (BCE), >1 for multiclass (CE).
        dropout:       dropout in projection, expert attention and classifier.
        use_gnn:       routing GNN on/off. Off -> Z = H0 (ablation "w/o routing GNN";
                       the region graph is still used for graph regularisation).
        gnn_layers:    number of GATv2 layers.
        gnn_heads:     GATv2 attention heads.
        gnn_dropout:   GATv2 dropout.
        k_neighbors:   k of the region kNN graph.
        n_tokens:      maximum number of region tokens M (grid side = sqrt(n_tokens)).
        n_experts:     number of experts E.
        ot_eps:        entropic regularisation eps_ot.
        sinkhorn_iters: total Sinkhorn updates T.
        expert_heads:  attention heads per expert.
        d_comp:        routing-profile embedding dim (0 disables eta).
        topk:          experts kept per region.
        lam_size:      weight of a row-constant lam_size * log(A_m) term in the cost.
                       Sinkhorn row scaling absorbs row constants, so it does not
                       change the converged plan; kept so released checkpoints
                       reproduce bit-for-bit.
        lam_rare:      rarity boost in the supply marginal, r_m ∝ A_m (1 + lam_rare u_m).
        lam_smooth:    graph-regularisation strength lam_s (Eq. 2).
        smooth_iters:  number of graph-regularisation blocks (0 = plain Sinkhorn).
        routing:       "ot" (ROAM) or "softmax" (ablation: row-softmax gate over
                       cosine scores / eps; no capacity constraint, no graph term,
                       and dispatch rows sum to 1 instead of the supply r_m).
        use_ot_modulation: add log(gamma + delta) to expert attention (Eq. 4). If
                       False, each expert attends over all M regions (its dispatch
                       support is no longer enforced).
        ot_mod_detach: stop-gradient on gamma inside expert attention (ablation).
    """

    def __init__(
        self,
        d_in=1536, d_proj=512, d_att=256, n_classes=1, dropout=0.25,
        use_gnn=True, gnn_layers=2, gnn_heads=4, gnn_dropout=0.1,
        k_neighbors=8,
        n_tokens=256, n_experts=8, ot_eps=0.1, sinkhorn_iters=20,
        expert_heads=2,
        d_comp=64,
        topk=2, lam_size=0.1, lam_rare=0.2, lam_smooth=0.1,
        smooth_iters=3,
        routing="ot", use_ot_modulation=True, ot_mod_detach=False,
    ):
        super().__init__()
        if routing not in ("ot", "softmax"):
            raise ValueError(f"Unknown routing: {routing}")
        self.n_experts = n_experts
        self.use_gnn = use_gnn
        self.target_M = n_tokens
        self.k_neighbors = k_neighbors
        self.ot_eps = ot_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.d_comp = d_comp
        self.topk = topk
        self.lam_size = lam_size
        self.lam_rare = lam_rare
        self.lam_smooth = lam_smooth
        self.smooth_iters = smooth_iters
        self.routing = routing

        # 1. Feature projection
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_proj), nn.ReLU(), nn.Dropout(dropout),
        )

        # 3. Routing GNN
        if use_gnn:
            self.gnn = RoutingGAT(d_proj, n_layers=gnn_layers,
                                  n_heads=gnn_heads, dropout=gnn_dropout)

        # 4. Expert prototypes mu_e
        self.protos = nn.Parameter(torch.randn(n_experts, d_proj) * 0.02)

        # 7. Per-expert aggregators
        expert_att_dim = max(d_att // 4, 32)
        self.experts = nn.ModuleList([
            ExpertAggregator(d_proj, expert_heads, expert_att_dim, dropout,
                             use_ot_modulation=use_ot_modulation,
                             ot_mod_detach=ot_mod_detach)
            for _ in range(n_experts)
        ])

        # 8. Fusion
        self.fusion = ExpertFusion(d_proj, d_att)

        # Routing-profile embedding eta(gamma)
        if d_comp > 0:
            self.profile_net = nn.Sequential(
                nn.Linear(n_experts * 2, d_comp), nn.ReLU(),
                nn.Linear(d_comp, d_comp),
            )

        # 9. Classifier
        cls_in = d_proj + (d_comp if d_comp > 0 else 0)
        self.head = nn.Sequential(
            nn.Linear(cls_in, d_att), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_att, n_classes),
        )

        self._diag = {}

    def _expert_profile(self, Alpha):
        """eta(gamma): per-expert dispatched load L_e and allocation entropy.

        Uses the sparse dispatch Alpha rather than Pi: Pi's column sums are fixed
        at 1/E, whereas the sparse loads vary across slides.
        """
        E_act = Alpha.shape[1]
        mass = Alpha.sum(dim=0)                                    # (E,) L_e
        An = Alpha / (Alpha.sum(dim=0, keepdim=True) + 1e-8)      # column-normalise
        ent = -(An * torch.log(An + 1e-8)).sum(dim=0)             # (E,)
        if E_act < self.n_experts:
            pad = self.n_experts - E_act
            mass, ent = F.pad(mass, (0, pad)), F.pad(ent, (0, pad))
        return self.profile_net(torch.cat([mass, ent]))

    def forward(self, features, coords=None):
        """
        Args:
            features: (N, d_in) patch features.
            coords:   (N, 2) patch coordinates.
        Returns:
            logits:         (n_classes,)
            patch_saliency: (N,) routing saliency, sum_e Alpha[region(i), e] * gate[e].
        """
        N = features.shape[0]
        E = min(self.n_experts, N)
        dev = features.device

        if coords is not None:
            coords = coords.to(dev)

        # Step 1: project + region tokens
        h = self.proj(features)                               # (N, d)
        target_M = min(self.target_M, N)
        region_id, H0, A, C_cent = _build_regions(coords, h, target_M)
        M = H0.shape[0]

        # Step 2: region graph
        k = min(self.k_neighbors, M - 1)
        has_graph = M > 1 and k >= 1
        if has_graph:
            neighbors = _knn_graph(C_cent, k)
        else:
            neighbors = torch.zeros(M, 1, dtype=torch.long, device=dev)

        # Step 3: routing embeddings
        if self.use_gnn and has_graph:
            Z = self.gnn(H0, neighbors)
        else:
            Z = H0

        # Step 4: cost + marginals
        protos = self.protos[:E]
        D = 1.0 - F.normalize(Z, dim=-1) @ F.normalize(protos, dim=-1).T

        if has_graph:
            g_rare = _compute_rarity(Z, neighbors)
        else:
            g_rare = torch.zeros(M, device=dev)

        C_cost = D + self.lam_size * torch.log(A + 1e-8).unsqueeze(1)

        r_raw = A * (1.0 + self.lam_rare * g_rare)
        r = r_raw / (r_raw.sum() + 1e-8)
        q = h.new_ones(E) / E

        # Step 5-6: routing plan + sparse dispatch
        if self.routing == "softmax":
            scores = F.normalize(Z, dim=-1) @ F.normalize(protos, dim=-1).T
            Pi = F.softmax(scores / self.ot_eps, dim=-1)
        elif has_graph:
            Pi = _graph_sinkhorn(
                C_cost, r, q, neighbors,
                eps=self.ot_eps, iters=self.sinkhorn_iters,
                smooth_iters=self.smooth_iters, lam_smooth=self.lam_smooth,
            )
        else:
            Pi = _sinkhorn_basic(C_cost, r, q, self.ot_eps, self.sinkhorn_iters)
        Alpha = _topk_dispatch(Pi, k=self.topk)

        # Step 7: per-expert pooling
        ez = []
        for e in range(E):
            w_e = Alpha[:, e]
            if w_e.sum() < 1e-8:
                ez.append(torch.zeros(H0.shape[1], device=dev))
            else:
                ez.append(self.experts[e](H0, w_e))
        for _ in range(self.n_experts - E):
            ez.append(torch.zeros_like(ez[0]))
        ez = torch.stack(ez)

        # Step 8: fusion (+ routing profile)
        z, gate = self.fusion(ez)
        if self.d_comp > 0:
            z = torch.cat([z, self._expert_profile(Alpha)])

        # Step 9: classify
        logits = self.head(z)

        with torch.no_grad():
            ea = gate.squeeze(-1)[:E]
            region_saliency = (Alpha[:, :E] * ea.unsqueeze(0)).sum(dim=1)
            patch_saliency = region_saliency[region_id]

            self._diag = dict(
                Pi=Pi, Alpha=Alpha, region_id=region_id,
                expert_gate=gate, rarity=g_rare, expert_z=ez,
                neighbors=neighbors, centroids=C_cent,
            )

        return logits, patch_saliency
