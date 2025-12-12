from functools import partial
import jax
from jax import numpy as jnp, lax as jlx
from jax.scipy.special import logsumexp
import numpy as np
import math
import chex
import dm_pix


def gradient_loss(s, penalty='l2'):
    dy = jnp.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = jnp.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = jnp.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = jnp.mean(dx) + jnp.mean(dy) + jnp.mean(dz)
    return d / 3.0


def diceLoss(y_true, y_pred):
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(axis=vol_axes)
    bottom = jnp.clip((y_true + y_pred).sum(axis=vol_axes), min=1e-5)
    dice = jnp.mean(top / bottom)
    return -dice


def compute_local_sums(I, J, win_spatial_dims, window_strides_arg, padding_values_arg, ndims):
    """
    Computes local sums for NCC calculation using JAX convolutions.
    This is a helper function for ncc_loss_jax.

    Args:
        I: Input JAX array with shape [batch_size, *vol_shape, nb_feats].
        J: Input JAX array with shape [batch_size, *vol_shape, nb_feats].
        win_spatial_dims: List of kernel sizes for spatial dimensions (e.g., [kH, kW] for 2D).
        window_strides_arg: Tuple of strides for each spatial dimension (e.g., (sH, sW)).
        padding_values_arg: Tuple of symmetric padding values for each spatial dimension (e.g., (pH, pW)).
        ndims: Number of spatial dimensions (1, 2, or 3).

    Returns:
        A tuple containing (I_var, J_var, cross_term), which are JAX arrays.
    """
    nb_feats = I.shape[-1]

    # Basic input validation
    if I.shape[:-1] != J.shape[:-1] or I.shape[-1] != J.shape[-1]:
        raise ValueError(f"Input arrays I and J must have the same dimensions. Got I: {I.shape}, J: {J.shape}")
    if I.ndim != ndims + 2:  # batch_size, *vol_shape (ndims), nb_feats
        raise ValueError(f"Input array I has incorrect number of dimensions ({I.ndim}) for ndims={ndims}.")
    if J.ndim != ndims + 2:
        raise ValueError(f"Input array J has incorrect number of dimensions ({J.ndim}) for ndims={ndims}.")
    if nb_feats == 0 and I.size > 0:  # Check I.size to avoid error on genuinely empty input
        raise ValueError("Input tensor I has 0 features (channels) but is not an empty array.")

    # Kernel for jax.lax.conv_general_dilated
    # Shape: (*kernel_spatial_dims, C_in_per_group, C_out_per_group)
    # We want to sum each feature independently using the same windowed filter.
    # So, C_in_per_group = 1 (filter takes one channel from the group at a time)
    # C_out_per_group = 1 (filter produces one channel for that group)
    # feature_group_count will be nb_feats (each input feature is its own group).
    kernel_spatial_shape = tuple(win_spatial_dims)
    sum_filt_kernel = jnp.ones(kernel_spatial_shape + (1, 1), dtype=I.dtype)

    # Strides for jax.lax.conv_general_dilated (e.g., (1,) or (1,1) or (1,1,1))
    strides_jax = window_strides_arg

    # Padding for jax.lax.conv_general_dilated
    # Needs to be a list of (pad_low, pad_high) pairs for each spatial dimension.
    padding_jax = []
    for i in range(ndims):
        # padding_values_arg contains symmetric padding amount for each dim, e.g., (pad_H, pad_W) for 2D
        pad_val = padding_values_arg[i]
        padding_jax.append((pad_val, pad_val))  # Symmetric padding

    # Set dimension_numbers for jax.lax.conv_general_dilated
    # lhs (input): (N, *Spatial, C) - N: batch, C: channels/features
    # rhs (kernel): (*KernelSpatial, I_group, O_group) - I: input feats/group, O: output feats/group
    # out (output): (N, *Spatial_out, C_total)
    if ndims == 1:
        dimension_numbers = ('NWC', 'WIO', 'NWC')  # W: width
    elif ndims == 2:
        dimension_numbers = ('NHWC', 'HWIO', 'NHWC')  # H: height, W: width
    elif ndims == 3:
        dimension_numbers = ('NDHWC', 'DHWIO', 'NDHWC')  # D: depth, H: height, W: width
    else:
        raise ValueError(f"Unsupported number of dimensions (ndims): {ndims}. Must be 1, 2, or 3.")

    def convolve(data_lhs):
        """Applies the convolution to sum values in a local window."""
        return jlx.conv_general_dilated(
            lhs=data_lhs,  # Input data
            rhs=sum_filt_kernel,  # Kernel (filter)
            window_strides=strides_jax,  # Strides for the convolution
            padding=padding_jax,  # Padding for each spatial dimension
            dimension_numbers=dimension_numbers,  # Specifies layout of dims
            feature_group_count=nb_feats  # Applies sum_filt_kernel to each input feature independently
        )

    # Calculate squared values and product
    I2 = I * I
    J2 = J * J
    IJ = I * J

    # Compute local sums using convolution
    I_sum = convolve(I)
    J_sum = convolve(J)
    I2_sum = convolve(I2)
    J2_sum = convolve(J2)
    IJ_sum = convolve(IJ)

    # Calculate window size (number of elements in the window)
    win_size_float = float(np.prod(win_spatial_dims))
    if win_size_float == 0:  # Should not happen with valid win_spatial_dims
        raise ValueError("Window size (product of win_spatial_dims) is zero.")

    # Local means
    u_I = I_sum / win_size_float
    u_J = J_sum / win_size_float

    # Local variance and cross-correlation terms
    # cross = sum_window((I - u_I)(J - u_J))
    cross_term = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size_float
    # I_var = sum_window((I - u_I)^2)
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size_float
    # J_var = sum_window((J - u_J)^2)
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size_float

    return I_var, J_var, cross_term

def ncc_loss(I, J, win_arg=None):
    """
    Calculates the Normalized Cross-Correlation (NCC) loss between I and J using JAX.

    The NCC is a measure of similarity between two images or signals. This loss
    is -NCC^2, meaning it encourages high magnitudes of correlation (either positive or negative).

    Args:
        I: Input JAX array with shape [batch_size, *vol_shape, nb_feats].
           Assumes channel-last format (e.g., NHWC for 2D images).
        J: Input JAX array with the same shape and format as I.
        win_arg: Optional. Window size for local NCC calculation.
                 - If None, defaults to [9]*ndims (e.g., 9x9 for 2D).
                 - If an int, used for all spatial dimensions (e.g., win_arg=7 means 7x7 for 2D).
                 - If a list/tuple, specifies window size for each spatial dimension
                   (e.g., win_arg=[7,5] for 2D HxW). Length must match ndims.

    Returns:
        A scalar JAX array representing the mean NCC loss (-NCC^2).
    """
    if not isinstance(I, jnp.ndarray) or not isinstance(J, jnp.ndarray):
        raise TypeError("Inputs I and J must be JAX numpy arrays.")

    # Determine number of spatial dimensions
    ndims = I.ndim - 2  # (batch_dim + feature_dim)
    if not (1 <= ndims <= 3):
        raise ValueError(
            f"Input volume dimensionality (ndims={ndims}, derived from I.ndim={I.ndim}) must be 1, 2, or 3."
        )

    # Process window argument
    if win_arg is None:
        win_spatial_dims = [9] * ndims  # Default window size
    else:
        if isinstance(win_arg, int):
            win_spatial_dims = [win_arg] * ndims
        elif isinstance(win_arg, (list, tuple)):
            if len(win_arg) != ndims:
                raise ValueError(
                    f"Length of win_arg ({len(win_arg)}) must match number of spatial dimensions ({ndims})."
                )
            win_spatial_dims = list(win_arg)
        else:
            raise TypeError("win_arg must be None, an int, or a list/tuple of ints.")

    # Validate window dimensions
    for k_dim_size in win_spatial_dims:
        if not isinstance(k_dim_size, int) or k_dim_size <= 0:
            raise ValueError("Window dimensions in win_arg must be positive integers.")

    # Determine padding for each spatial dimension (symmetric padding)
    # This ensures the output of convolution has the same spatial dimensions as input if stride is 1.
    pad_values_per_dim = []
    for k_dim_size in win_spatial_dims:
        pad_values_per_dim.append(math.floor(k_dim_size / 2))

    # Strides and padding tuples based on ndims, to be passed to compute_local_sums_jax
    # Strides are typically (1,) for 1D, (1,1) for 2D, (1,1,1) for 3D for standard NCC.
    # padding_arg_tuple for compute_local_sums_jax is (pad_dim1, pad_dim2, ...)
    if ndims == 1:
        stride_tuple = (1,)
        padding_arg_tuple = (pad_values_per_dim[0],)
    elif ndims == 2:
        stride_tuple = (1, 1)
        padding_arg_tuple = (pad_values_per_dim[0], pad_values_per_dim[1])
    else:  # ndims == 3
        stride_tuple = (1, 1, 1)
        padding_arg_tuple = (pad_values_per_dim[0], pad_values_per_dim[1], pad_values_per_dim[2])

    # Compute local variances and cross-correlation term
    I_var, J_var, cross = compute_local_sums(
        I, J, win_spatial_dims, stride_tuple, padding_arg_tuple, ndims
    )

    # Calculate squared NCC: (Cov(I,J)^2) / (Var(I) * Var(J))
    # Note: 'cross' term is N * Cov(I,J), and 'I_var'/'J_var' are N * Var(I)/Var(J)
    # So, (N*Cov)^2 / ((N*Var_I)*(N*Var_J)) = Cov^2 / (Var_I * Var_J), which is NCC_squared.
    # Adding a small epsilon to the denominator for numerical stability.
    ncc_squared = (cross * cross) / (I_var * J_var + 1e-5)

    # The loss is the negative mean of the squared NCC.
    # This encourages high correlation magnitude (positive or negative).
    loss = -1 * jnp.mean(ncc_squared)
    return loss

def correlation_coefficient_loss(x, y):
    epsilon = 10e-5
    mx = jnp.mean(x, axis=[1, 2], keepdims=True)
    my = jnp.mean(y, axis=[1, 2], keepdims=True)
    xm, ym = x - mx, y - my
    r_num = jnp.sum(xm * ym, axis=[1, 2])
    x_square_sum = jnp.sum(xm * xm, axis=[1, 2])
    y_square_sum = jnp.sum(ym * ym, axis=[1, 2])
    r_den = jnp.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return jnp.mean(1. - r)

def simae(
    a: chex.Array,
    b: chex.Array,
    *,
    ignore_nans: bool = False,
) -> chex.Numeric:
  """Returns the Scale-Invariant Mean Squared Error between `a` and `b`.

  For each image pair, a scaling factor for `b` is computed as the solution to
  the following problem:

    min_alpha || vec(a) - alpha * vec(b) ||_2^2

  where `a` and `b` are flattened, i.e., vec(x) = np.flatten(x). The MSE between
  the optimally scaled `b` and `a` is returned: mse(a, alpha*b).

  This is a scale-invariant metric, so for example: simse(x, y) == sims(x, y*5).

  This metric was used in "Shape, Illumination, and Reflectance from Shading" by
  Barron and Malik, TPAMI, '15.

  Args:
    a: First image (or set of images).
    b: Second image (or set of images).
    ignore_nans: If True, will ignore NaNs in the inputs.

  Returns:
    SIMAE between `a` and `b`.
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank([a, b], {3, 4})
  chex.assert_type([a, b], float)
  chex.assert_equal_shape([a, b])

  sum_fn = jnp.nansum if ignore_nans else jnp.sum
  a_dot_b = sum_fn((a * b), axis=(-3, -2, -1), keepdims=True)
  b_dot_b = sum_fn((b * b), axis=(-3, -2, -1), keepdims=True)
  alpha = a_dot_b / b_dot_b
  return dm_pix.mae(a, alpha * b, ignore_nans=ignore_nans)

def contrastive_ce_loss(
    dist_pos: jnp.ndarray,
    dist_neg: jnp.ndarray,
    temperature: float = 0.07,
    reduction: str = "mean",
):
    """
    InfoNCE‑style contrastive loss using pre‑computed distance matrices.

    Parameters
    ----------
    dist_pos : (M, P) array
        Euclidean (or other) distances from each of M anchors
        to their P *closest / similar* neighbours.
    dist_neg : (M, P) array
        Distances from the same anchors to their P *farthest / dissimilar* neighbours.
    temperature : float, default 0.07
        Soft‑max temperature τ used in SimCLR, MoCo, etc.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        Aggregation mode applied over all (anchor, positive) pairs.

    Returns
    -------
    jnp.ndarray
        • scalar loss if reduction is 'mean' or 'sum'
        • (M, P) array of individual losses if reduction == 'none'
    """
    if dist_pos.shape != dist_neg.shape:
        raise ValueError("dist_pos and dist_neg must have the same shape")

    # ------------------------------------------------------------------
    # 1.  Convert **distance** to **similarity**: s = −d
    #     (smaller distances → larger similarities).
    # 2.  Scale by temperature τ.
    # ------------------------------------------------------------------
    pos_logits = -dist_pos / temperature          # (M, P)
    neg_logits = -dist_neg / temperature          # (M, P)

    # ------------------------------------------------------------------
    # 3.  Compute log‑denominator        log( e^{s⁺/τ} + ∑ e^{s⁻/τ} )
    #     for every (anchor, positive) pair in a numerically stable way.
    # ------------------------------------------------------------------
    # log ∑ e^{s⁻/τ}   – one value per anchor, shape (M, 1)
    neg_lse = logsumexp(neg_logits, axis=1, keepdims=True)

    # log( e^{s⁺/τ} + ∑ e^{s⁻/τ} )  – broadcasts over the P positives
    log_denom = jnp.logaddexp(pos_logits, neg_lse)

    # ------------------------------------------------------------------
    # 4.  InfoNCE loss  −s⁺/τ + log‑denominator  (per positive sample)
    # ------------------------------------------------------------------
    loss_per_pair = -pos_logits + log_denom       # (M, P)

    # ------------------------------------------------------------------
    # 5.  Reduction
    # ------------------------------------------------------------------
    if reduction == "mean":
        return jnp.mean(loss_per_pair)
    if reduction == "sum":
        return jnp.sum(loss_per_pair)
    if reduction == "none":
        return loss_per_pair
    raise ValueError("reduction must be 'mean', 'sum' or 'none'")

def triplet_loss(
    dist_pos: jnp.ndarray,
    dist_neg: jnp.ndarray,
    margin: float = 1.0,
    reduction: str = "mean",
):
    """
    Triplet loss for pre‑computed distance matrices.

    Parameters
    ----------
    dist_pos : (M, P) array
        Distances from each of M anchors to their P *closest* (similar) neighbours.
    dist_neg : (M, P) array
        Distances from each of M anchors to their P *farthest* (dissimilar) neighbours.
    margin : float, default 1.0
        Desired minimum distance between dissimilar pairs.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        How to aggregate the per‑pair losses.

    Returns
    -------
    jnp.ndarray
        • scalar loss if reduction is 'mean' or 'sum'
        • (M, P) array of individual losses if reduction == 'none'
    """
    if dist_pos.shape != dist_neg.shape:
        raise ValueError("dist_pos and dist_neg must have the same shape")

    # Positive term ‑‑ pull similar samples together
    # loss_pos = jnp.square(dist_pos)
    loss_pos = dist_pos

    # Negative term ‑‑ push dissimilar samples apart (only if within margin)
    # loss_neg = jnp.square(jnp.clip(margin - dist_neg, a_min=0.0))
    loss_neg = jnp.clip(margin - dist_neg, a_min=0.0)

    per_pair = 0.5 * (loss_pos + loss_neg)   # 0.5 is conventional; optional

    if reduction == "mean":
        return jnp.mean(per_pair)
    if reduction == "sum":
        return jnp.sum(per_pair)
    if reduction == "none":
        return per_pair
    raise ValueError("reduction must be 'mean', 'sum' or 'none'")


@partial(jax.jit, static_argnames=['num_projections',])
def sliced_wasserstein_loss(x: jnp.ndarray, x_true: jnp.ndarray, key: jax.random.PRNGKey, num_projections: int = 128) -> jnp.ndarray:
    """
    Computes the Sliced-Wasserstein-2 distance to a uniform distribution.

    Args:
        x: Input array of shape (N, 3).
        x_true: True data of shape (N, 3).
        num_projections: The number of random 1D projections to use.

    Returns:
        A scalar loss value.
    """
    N, D = x.shape

    # 1. Generate random projections
    key, proj_key, true_key = jax.random.split(key, 3)
    projections = jax.random.normal(proj_key, shape=(D, num_projections))
    projections = projections / jnp.linalg.norm(projections, axis=0, keepdims=True)

    # 2. Project both the input data and the true uniform data
    x_proj = x @ projections
    x_true_proj = x_true @ projections

    # 3. Sort the projections along the N-axis
    x_proj_sorted = jnp.sort(x_proj, axis=0)
    x_true_proj_sorted = jnp.sort(x_true_proj, axis=0)

    # 4. Compute the L2 distance between sorted projections and average over all projections
    # This is the squared Sliced-Wasserstein-2 distance
    loss = jnp.mean((x_proj_sorted - x_true_proj_sorted) ** 2)

    return loss


def distance_regularizer_from_graph(c0: jnp.ndarray,
                                    c: jnp.ndarray,
                                    edge_index: jnp.ndarray):
    """
    c0: (N, 3) reference centers (no grad needed).
    c:  (N, 3) deformed centers (requires grad).
    edge_index: (2, E) int array of edge indices [i, j].

    Returns:
        scalar loss (if mean/sum) or (E,) per-edge loss (if "none").
    """
    i, j = edge_index  # (E,), (E,)

    d0 = jnp.linalg.norm(c0[i] - c0[j], axis=-1)  # (E,)
    d  = jnp.linalg.norm(c[i]  - c[j],  axis=-1)  # (E,)

    diff = d - d0
    loss_per_edge = diff ** 2  # (E,)

    return jnp.mean(loss_per_edge)


def repulsion_from_graph(c0: jnp.ndarray,
                         c: jnp.ndarray,
                         edge_index: jnp.ndarray):
    """
    c0: (N, 3) reference centers (no grad needed).
    c:  (N, 3) deformed centers (requires grad).
    edge_index: (2, E) int array of edge indices [i, j].

    Returns:
        scalar loss (if mean/sum) or (E,) per-edge loss (if "none").
    """

    i, j = edge_index  # (E,), (E,)

    d  = jnp.linalg.norm(c[i]  - c[j],  axis=-1)  # (E,)

    tau = jnp.linalg.norm(c0[i] - c0[j], axis=-1).mean()

    x = jnp.where(d < tau, 1.0, tau)

    return jnp.mean(x * (d - tau) ** 2.).mean()
