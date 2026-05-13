"""
Point Transformer decoder for CryoEM structural heterogeneity analysis.

Re-implementation in JAX / Flax NNX of the architecture introduced in:

    Chen, M., Li, M., Liao, R. (2026).
    "Point transformer for protein structural heterogeneity analysis using CryoEM."
    arXiv:2601.18713

The decoder maps a low-dimensional latent vector z (output of the existing
MLP encoder that embeds the particle image) to a Gaussian Mixture Model
(GMM) representation of the protein:

    z : (B, d_latent)  ->  GMM : (B, N, 5)

where each of the N Gaussians is described by 5 channels
(x, y, z, amplitude, sigma).

Architecture (Figure 1 of the paper)
------------------------------------

    z  --[2-layer MLP]-->  (B, 64, 256)
       --[PT block 1] -->  (B,  64, 256)   ── res1 ─┐
       --[TransitionUp]->  (B, 256, 256)            │
       --[PT block 2] -->  (B, 256, 256)   ── res2 ─┤
       --[TransitionUp]->  (B,1024, 256)            │
       --[PT block 3] -->  (B,1024, 256)            │
       --[TransitionUp]->  (B,  N , 256) main ──────┤
                                                    │
              concat(main, up(res1), up(res2))  ──> MLP ── (B, N, 5)

The Point Transformer (PT) layer is the *vector self-attention* of
Zhao et al. (arXiv:2012.09164):

    y_i = Σ_{j ∈ N(i)}  softmax_j( γ( φ(x_i) − ψ(x_j) + δ_ij ) ) ⊙ ( α(x_j) + δ_ij )

with relative-position encoding δ_ij = θ(p_i − p_j).

Because the point-cloud *geometry* (the GMM Gaussian centres) is fixed
for a given dataset, all k-NN neighbour indices and inverse-distance
interpolation weights are computed once (``build_geometry``) and held
constant throughout training, which is the speed-up trick mentioned in
the paper.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct


default_kernel_init = nnx.initializers.lecun_normal()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class MLP(nnx.Module):
    """Two-layer MLP with a configurable hidden size and ReLU activation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 last_kernel_init: nnx.Initializer = default_kernel_init, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, out_dim, kernel_init=last_kernel_init, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(nnx.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Point Transformer (vector self-attention) layer
# ---------------------------------------------------------------------------

class PointTransformerLayer(nnx.Module):
    """Vector self-attention restricted to a local k-NN neighbourhood.

    Implements the operator of Zhao et al. (2020, eq. 3):

        y_i = Σ_{j ∈ N(i)} softmax_j(γ(φ(x_i) − ψ(x_j) + δ_ij)) ⊙ (α(x_j) + δ_ij)

    Notes
    -----
    *   φ, ψ, α are point-wise linear projections (Q, K, V).
    *   γ is a small MLP that maps the *difference* vector to the same
        dimension as V — vector attention, not scalar attention.
    *   δ_ij = θ(p_i − p_j) is a learned positional encoding of the
        *relative* position of neighbour j with respect to point i.
    *   The neighbourhood N(i) (k-NN indices) and the relative positions
        are pre-computed and passed in at call time.
    """

    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.dim = dim
        self.phi   = nnx.Linear(dim, dim, rngs=rngs)
        self.psi   = nnx.Linear(dim, dim, rngs=rngs)
        self.alpha = nnx.Linear(dim, dim, rngs=rngs)
        self.gamma = MLP(dim, dim, dim, rngs=rngs)       # attention weights
        self.theta = MLP(3,   dim, dim, rngs=rngs)       # δ(p_i − p_j)

    def __call__(
        self,
        x: jax.Array,             # (B, N, C)
        neighbour_idx: jax.Array, # (N, K)
        rel_pos: jax.Array,       # (N, K, 3)
    ) -> jax.Array:                # (B, N, C)
        q = self.phi(x)            # (B, N, C)
        k = self.psi(x)            # (B, N, C)
        v = self.alpha(x)          # (B, N, C)

        # Gather neighbour features:  result shape (B, N, K, C)
        # JAX fancy indexing over the second axis works as we want.
        k_nb = k[:, neighbour_idx, :]
        v_nb = v[:, neighbour_idx, :]

        # Relative-position encoding δ, shape (N, K, C) -> (1, N, K, C)
        delta = self.theta(rel_pos)[None, ...]

        # γ( φ(x_i) − ψ(x_j) + δ_ij )  →  attention logits of shape (B, N, K, C)
        attn = self.gamma(q[:, :, None, :] - k_nb + delta)
        attn = jax.nn.softmax(attn, axis=2)            # softmax over neighbours K

        # Weighted sum with the *value + position* tensor; reduction over K.
        return jnp.sum(attn * (v_nb + delta), axis=2)  # (B, N, C)


class PTBlock(nnx.Module):
    """Full Point Transformer *block* with residual connection.

    Following the original PT paper:
        x -> Linear -> PointTransformerLayer -> Linear -> + x
    With a LayerNorm at the end for training stability.
    """

    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.lin_in  = nnx.Linear(dim, dim, rngs=rngs)
        self.attn    = PointTransformerLayer(dim, rngs=rngs)
        self.lin_out = nnx.Linear(dim, dim, rngs=rngs)
        # self.norm    = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(self, x, neighbour_idx, rel_pos):
        h = self.lin_in(x)
        h = self.attn(h, neighbour_idx, rel_pos)
        h = self.lin_out(h)
        # return self.norm(x + h)
        return x + h


# ---------------------------------------------------------------------------
# Transition Up — interpolation from coarse to fine point set
# ---------------------------------------------------------------------------

class TransitionUp(nnx.Module):
    """Upsample features from a coarse point set to a fine one.

    For every fine point we keep the indices of its k nearest neighbours
    in the *coarse* set together with normalised inverse-distance
    weights.  The feature of the fine point is then::

        y_i  =  Σ_{j ∈ NN_k(i)} w_ij · (Linear · x_j)

    Both ``upsample_idx`` and ``upsample_w`` are pre-computed once and
    re-used for the whole dataset.
    """

    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.proj = nnx.Linear(in_dim, out_dim, rngs=rngs)

    def __call__(
        self,
        x_coarse: jax.Array,       # (B, N_coarse, C_in)
        upsample_idx: jax.Array,   # (N_fine, K)
        upsample_w: jax.Array,     # (N_fine, K)
    ) -> jax.Array:                 # (B, N_fine, C_out)
        x = self.proj(x_coarse)               # (B, N_coarse, C_out)
        gathered = x[:, upsample_idx, :]      # (B, N_fine, K, C_out)
        return jnp.sum(gathered * upsample_w[None, :, :, None], axis=2)


# ---------------------------------------------------------------------------
# Point Transformer decoder (the full Figure-1 architecture)
# ---------------------------------------------------------------------------

@struct.dataclass
class Geometry:
    """All pre-computed point positions and neighbour tables.

    Registered as a JAX pytree via ``flax.struct.dataclass`` so it can be
    passed transparently through ``nnx.jit`` / ``nnx.grad``.

    For 3 PT layers operating on (n0, n1, n2) = (64, 256, 1024) points,
    plus a final GMM layer of N points:

    *   ``positions`` :  list of 4 arrays, shape (n_i, 3)
    *   ``neighbours``:  list of 3 arrays, shape (n_i, K_attn) — k-NN within each PT layer
    *   ``rel_pos``   :  list of 3 arrays, shape (n_i, K_attn, 3) — relative positions
    *   ``up_idx`` /
        ``up_w``      :  list of 3 arrays each — sequential 64→256→1024→N upsampling
    *   ``res_idx`` /
        ``res_w``     :  list of 2 arrays each — direct residual upsampling
                         from the n0-point and n1-point layers to the final N points
    """
    positions:  list
    neighbours: list
    rel_pos:    list
    up_idx:     list
    up_w:       list
    res_idx:    list
    res_w:      list


class PointTransformerDecoder(nnx.Module):
    """Point-Transformer based decoder of GMM parameters.

    Parameters
    ----------
    latent_dim
        Size of the latent code produced by the (frozen) MLP encoder.
    feat_dim
        Feature width carried through the PT pipeline (256 in the paper).
    hierarchical_sizes
        Number of points at each PT layer.  The paper uses (64, 256, 1024).
    """

    def __init__(
        self,
        latent_dim: int,
        feat_dim: int = 256,
        hierarchical_sizes: Sequence[int] = (64, 256, 1024),
        *,
        rngs: nnx.Rngs,
    ):
        assert len(hierarchical_sizes) == 3, \
            "Paper architecture has exactly 3 PT layers"
        self.feat_dim = feat_dim
        self.n0, self.n1, self.n2 = hierarchical_sizes

        # z (B, d_latent)  ->  (B, n0 * feat_dim)  ->  reshape to (B, n0, feat_dim)
        self.input_mlp = MLP(latent_dim, feat_dim * 2,
                             self.n0 * feat_dim, rngs=rngs)

        # Three PT blocks, all 256-dim
        self.pt1 = PTBlock(feat_dim, rngs=rngs)
        self.pt2 = PTBlock(feat_dim, rngs=rngs)
        self.pt3 = PTBlock(feat_dim, rngs=rngs)

        # Sequential upsamplers   n0 -> n1 -> n2 -> N
        self.tu1 = TransitionUp(feat_dim, feat_dim, rngs=rngs)
        self.tu2 = TransitionUp(feat_dim, feat_dim, rngs=rngs)
        self.tu3 = TransitionUp(feat_dim, feat_dim, rngs=rngs)

        # Residual upsamplers (direct from PT1 / PT2 to the final N points)
        self.res_up1 = TransitionUp(feat_dim, feat_dim, rngs=rngs)
        self.res_up2 = TransitionUp(feat_dim, feat_dim, rngs=rngs)

        # Final dense head: concat(main, res1, res2) -> 5 GMM channels
        self.head = MLP(3 * feat_dim, feat_dim, 4, last_kernel_init=nnx.initializers.normal(1e-4), rngs=rngs)

    def __call__(self, z: jax.Array, geom: Geometry) -> jax.Array:
        """
        Parameters
        ----------
        z    : (B, latent_dim)
        geom : pre-computed Geometry object

        Returns
        -------
        gmm  : (B, N, 4)  — x, y, z, amplitude, sigma for each Gaussian.
        """
        B = z.shape[0]

        # 1) MLP -> (B, n0, feat_dim)
        x = self.input_mlp(z).reshape(B, self.n0, self.feat_dim)

        # 2) PT block 1 on n0 points
        x1 = self.pt1(x, geom.neighbours[0], geom.rel_pos[0])

        # 3) Upsample n0 -> n1 + PT block 2
        x = self.tu1(x1, geom.up_idx[0], geom.up_w[0])
        x2 = self.pt2(x, geom.neighbours[1], geom.rel_pos[1])

        # 4) Upsample n1 -> n2 + PT block 3
        x = self.tu2(x2, geom.up_idx[1], geom.up_w[1])
        x3 = self.pt3(x, geom.neighbours[2], geom.rel_pos[2])

        # 5) Final main upsample n2 -> N
        x_main = self.tu3(x3, geom.up_idx[2], geom.up_w[2])

        # 6) Residual branches: PT1 (n0) and PT2 (n1) directly to N points
        x1_up = self.res_up1(x1, geom.res_idx[0], geom.res_w[0])
        x2_up = self.res_up2(x2, geom.res_idx[1], geom.res_w[1])

        # 7) Concatenate features and project to 5 GMM channels
        cat = jnp.concatenate([x_main, x1_up, x2_up], axis=-1)
        return self.head(cat)


# ---------------------------------------------------------------------------
# Geometry pre-computation
# ---------------------------------------------------------------------------

def build_geometry(
    gmm_positions: np.ndarray,
    hierarchical_sizes: Sequence[int] = (64, 256, 1024),
    k_attn: int = 16,
    k_up: int = 3,
    *,
    seed: int = 0,
) -> Geometry:
    """Pre-compute everything the decoder needs that depends only on geometry.

    Parameters
    ----------
    gmm_positions : (N, 3) float array
        The 3-D coordinates of the Gaussian centres of the *base* (mean) GMM.
        Typically obtained by fitting the consensus map of the dataset.
    hierarchical_sizes
        Number of points in each PT layer.  Must be ascending and the last
        value must be < N.
    k_attn
        Number of neighbours used by the self-attention of every PT layer.
    k_up
        Number of source neighbours used to interpolate each fine point in
        the TransitionUp layers (inverse-distance weighting).
    """
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    gmm_positions = np.asarray(gmm_positions, dtype=np.float32)
    N = gmm_positions.shape[0]
    assert hierarchical_sizes[-1] < N, \
        "the last PT layer must be coarser than the final GMM"

    # ---- hierarchical point sets via k-means
    coarse_positions: list[np.ndarray] = []
    for n in hierarchical_sizes:
        km = KMeans(n_clusters=n, n_init=10, random_state=seed).fit(gmm_positions)
        coarse_positions.append(km.cluster_centers_.astype(np.float32))
    all_positions = coarse_positions + [gmm_positions]

    # ---- k-NN inside each PT layer (for self-attention)
    neighbours, rel_pos = [], []
    for p in coarse_positions:
        nn = NearestNeighbors(n_neighbors=k_attn).fit(p)
        idx = nn.kneighbors(p, return_distance=False).astype(np.int32)
        rel = p[idx] - p[:, None, :]                      # (n, K, 3)
        neighbours.append(jnp.asarray(idx))
        rel_pos.append(jnp.asarray(rel))

    # ---- sequential upsampling indices/weights:  P_i -> P_{i+1}
    up_idx, up_w = [], []
    for i in range(len(all_positions) - 1):
        coarse, fine = all_positions[i], all_positions[i + 1]
        nn = NearestNeighbors(n_neighbors=k_up).fit(coarse)
        d, idx = nn.kneighbors(fine, return_distance=True)
        w = 1.0 / (d + 1e-8)                              # inverse distance
        w /= w.sum(axis=1, keepdims=True)
        up_idx.append(jnp.asarray(idx.astype(np.int32)))
        up_w.append(jnp.asarray(w.astype(np.float32)))

    # ---- residual upsampling indices/weights: P_0 -> P_final and P_1 -> P_final
    res_idx, res_w = [], []
    fine = all_positions[-1]
    for i in (0, 1):
        coarse = all_positions[i]
        nn = NearestNeighbors(n_neighbors=k_up).fit(coarse)
        d, idx = nn.kneighbors(fine, return_distance=True)
        w = 1.0 / (d + 1e-8)
        w /= w.sum(axis=1, keepdims=True)
        res_idx.append(jnp.asarray(idx.astype(np.int32)))
        res_w.append(jnp.asarray(w.astype(np.float32)))

    return Geometry(
        positions=list(jnp.asarray(p) for p in all_positions),
        neighbours=list(neighbours),
        rel_pos=list(rel_pos),
        up_idx=list(up_idx),
        up_w=list(up_w),
        res_idx=list(res_idx),
        res_w=list(res_w),
    )
