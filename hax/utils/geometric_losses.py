import jax
import jax.numpy as jnp


def _distance_activation(pairwise_distances, mean_neighbour_distance):
    """A continuous assignment of 'neighbour-like-ness'."""
    cutoff = mean_neighbour_distance
    # Clamp distances between tau and 1.5 * tau
    x2 = jnp.clip(pairwise_distances, a_min=cutoff, a_max=1.5 * cutoff)
    # Quadratic kernel logic from DynaMight
    activation = (1.0 - (4.0 / cutoff ** 2.) * (x2 - cutoff) ** 2.) ** 2.
    return activation


def _neighbour_activation(neighbours_per_point, minimum=1.0, maximum=3.0):
    """Quadratic penalisation on number of neighbours outside range."""
    # Penalty for too few neighbors
    x1 = jnp.clip(neighbours_per_point, a_max=minimum)
    # Penalty for too many neighbors
    x2 = jnp.clip(neighbours_per_point, a_min=maximum)
    return (x1 - minimum) ** 2. + (x2 - maximum) ** 2.


def calculate_deformation_regularity_loss(positions, radius_graph, consensus_distances, edge_weights, eps=1e-8):
    """Preserves local distances (Spring-like prior)."""
    i, j = radius_graph
    # Safe distance calculation
    diffs = positions[i] - positions[j]
    distances = jnp.sqrt(jnp.sum(diffs ** 2., axis=-1) + eps)

    # Square error compared to consensus
    loss = (distances - consensus_distances) ** 2.
    return jnp.mean(edge_weights * loss)

def calculate_arap_loss(positions, consensus_positions, radius_graph, edge_weights, num_points, eps=1e-6):
    """As-rigid-as-possible (ARAP) energy.

    For every node the best-fit local *rotation* is factored out before the
    deformation is penalised, so a locally rigid motion (rotation of a domain)
    costs nothing and only the genuinely non-rigid distortion is penalised. This
    is a stronger, more physical prior than distance preservation alone
    (:func:`calculate_deformation_regularity_loss`), which also penalises rigid
    local rotations and therefore over-stiffens hinge/domain motions while being
    weaker against noise-driven shear.

    ``E = Σ_i Σ_{j∈N(i)} w_ij || (p'_i - p'_j) - R_i (p_i - p_j) ||²`` where the
    optimal ``R_i`` is recovered per node from the local covariance
    ``S_i = Σ_j w_ij (p_i - p_j)(p'_i - p'_j)^T`` via SVD (Sorkine & Alexa 2007).

    Parameters
    ----------
    positions
        ``(N, 3)`` deformed point positions (``p'``).
    consensus_positions
        ``(N, 3)`` rest/consensus positions (``p``).
    radius_graph
        ``(2, E)`` edge index ``(i, j)`` (same graph as the other graph losses).
    edge_weights
        ``(E,)`` per-edge weights.
    num_points
        ``N`` (static).
    """
    i, j = radius_graph
    e0 = consensus_positions[i] - consensus_positions[j]        # rest edge   (E, 3)
    e = positions[i] - positions[j]                             # deformed    (E, 3)
    w = edge_weights

    # Per-node covariance S_i = Σ_j w_ij e0 e^T, accumulated at node i.
    outer = w[:, None, None] * (e0[:, :, None] * e[:, None, :])  # (E, 3, 3)
    S = jax.ops.segment_sum(outer, i, num_segments=num_points)   # (N, 3, 3)
    # Regularise so degenerate (collinear / few-neighbour) nodes give R ~ I.
    S = S + eps * jnp.eye(3)[None]

    U, _, Vt = jnp.linalg.svd(S)
    V = jnp.swapaxes(Vt, -1, -2)
    Ut = jnp.swapaxes(U, -1, -2)
    R = jnp.einsum('nij,njk->nik', V, Ut)                        # V U^T
    # Reflection correction -> proper rotation (det = +1).
    det = jnp.linalg.det(R)
    D = jnp.ones((num_points, 3)).at[:, -1].set(jnp.sign(det))
    R = jnp.einsum('nij,nj,njk->nik', V, D, Ut)                  # V diag(1,1,sign) U^T

    # Optimal rotation is a target (ARAP local step): detach so we do not
    # backprop through the SVD (whose gradient is unstable at degenerate spectra)
    # and so the energy is a clean quadratic in the positions.
    R = jax.lax.stop_gradient(R)

    resid = e - jnp.einsum('eij,ej->ei', R[i], e0)              # (E, 3)
    return jnp.mean(w * jnp.sum(resid ** 2., axis=-1))


def calculate_deformation_coherence_loss(displacements, radius_graph, edge_weights, eps=1e-8):
    """Enforces smooth motion (Nearby points move together)."""
    i, j = radius_graph
    # Difference in the *change* of position
    diffs = displacements[i] - displacements[j]
    dist_sq = jnp.sum(diffs ** 2., axis=-1)

    return jnp.mean(edge_weights * dist_sq)

def calculate_repulsion_loss(positions, radius_graph, tau, eps=1e-8):
    """Prevents collisions/overlapping density."""
    i, j = radius_graph
    diffs = positions[i] - positions[j]
    distances = jnp.sqrt(jnp.sum(diffs ** 2., axis=-1) + eps)

    # Quadratic penalty if distance is less than tau (cutoff)
    cutoff = jnp.maximum(0.5, tau)
    # This acts like a 'soft' version of your multiplier trick
    penalty = jnp.clip(distances, a_max=cutoff)
    penalty = jnp.abs(penalty - cutoff)
    return penalty.mean()

def calculate_outlier_loss(positions, knn_graph, tau, eps=1e-8):
    """Prevents Gaussians from detaching into the solvent."""
    i, j = knn_graph
    diffs = positions[i] - positions[j]
    distances = jnp.sqrt(jnp.sum(diffs ** 2., axis=-1) + eps)

    cutoff = 1.5 * tau
    penalty = jnp.clip(distances, a_min=cutoff)
    return jnp.mean((penalty - cutoff) ** 2.)

def calculate_neighbour_loss(positions, radius_graph, tau, num_points, eps=1e-8):
    """Maintains uniform density across the protein volume."""
    i, j = radius_graph

    # 1. Compute distances for graph edges
    diffs = positions[i] - positions[j]
    distances = jnp.sqrt(jnp.sum(diffs ** 2., axis=-1) + eps)

    # 2. Convert distances to a 'is_neighbor' score (0 to 1)
    dist_activation = _distance_activation(distances, tau)

    # 3. Sum scores for each point i (DynaMight uses scatter)
    # segment_sum expects i to be sorted or we use num_segments
    n_neighbours = jax.ops.segment_sum(
        dist_activation,
        segment_ids=i,
        num_segments=num_points
    )

    # 4. Penalize if the sum is outside [1, 3]
    neighbor_penalty = _neighbour_activation(n_neighbours, minimum=1.0, maximum=3.0)
    return jnp.mean(neighbor_penalty)


### KEEP FOR REFERENCE ###
# def distance_regularizer_from_graph(c0: jnp.ndarray,
#                                     c: jnp.ndarray,
#                                     edge_index: jnp.ndarray):
#     """
#     c0: (N, 3) reference centers (no grad needed).
#     c:  (N, 3) deformed centers (requires grad).
#     edge_index: (2, E) int array of edge indices [i, j].
#
#     Returns:
#         scalar loss (if mean/sum) or (E,) per-edge loss (if "none").
#     """
#     i, j = edge_index  # (E,), (E,)
#
#     d0 = safe_norm(c0[i] - c0[j], axis=-1)  # (E,)
#     d  = safe_norm(c[i]  - c[j],  axis=-1)  # (E,)
#
#     diff = d - d0
#     loss_per_edge = diff ** 2.  # (E,)
#
#     return jnp.mean(loss_per_edge)
#
#
# def repulsion_from_graph(c0: jnp.ndarray,
#                          c: jnp.ndarray,
#                          edge_index: jnp.ndarray):
#     """
#     c0: (N, 3) reference centers (no grad needed).
#     c:  (N, 3) deformed centers (requires grad).
#     edge_index: (2, E) int array of edge indices [i, j].
#
#     Returns:
#         scalar loss (if mean/sum) or (E,) per-edge loss (if "none").
#     """
#
#     i, j = edge_index  # (E,), (E,)
#
#     d  = safe_norm(c[i] - c[j],  axis=-1)  # (E,)
#
#     tau = safe_norm(c0[i] - c0[j], axis=-1).mean()
#
#     x = jnp.where(d < tau, 1.0, tau)
#
#     return jnp.mean(x * (d - tau) ** 2.).mean()