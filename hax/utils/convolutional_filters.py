import jax
import jax.numpy as jnp
from jax import lax
from typing import Union, Tuple, Optional
from functools import partial


def get_gaussian_kernel1d(sigma: float, radius: int = 9,  dtype=jnp.float32) -> jnp.ndarray:
    """
    Generates a 1D Gaussian kernel.
    """
    # radius must be a concrete integer for arange to work during JIT
    x = jnp.arange(-radius, radius + 1)
    phi = jnp.exp(-0.5 / (sigma ** 2) * x ** 2)
    return (phi / phi.sum()).astype(dtype)


@partial(jax.jit, static_argnames=['mode', 'radius'])
def fast_gaussian_filter_3d(
        volume: jnp.ndarray,
        sigma: Union[jnp.ndarray, float, Tuple[float, float, float]],
        radius: int,
        mode: str = 'constant',
        dtype: Optional[jnp.dtype] = jnp.float32
) -> jnp.ndarray:
    """
    A highly optimized 3D Gaussian Low Pass Filter using separable depthwise convolutions.

    Handles inputs of shape (M, M, M, C) or (B, M, M, M, C).

    Args:
        volume: Input JAX array. Must be 4D (D, H, W, C) or 5D (B, D, H, W, C).
        sigma: Standard deviation for Gaussian kernel. Can be scalar or tuple (sigma_d, sigma_h, sigma_w).
               Must be a python float/int or tuple (not a JAX array) because it determines kernel shape.
        mode: Padding mode. Options: 'constant', 'edge' (replicate), 'reflect'.
              Note: 'reflect' in JAX pad is symmetric.

    Returns:
        Filtered volume with the same shape as input.
    """

    # Handle data type
    out_dtype = volume.dtype
    if dtype is None:
        dtype = out_dtype

    # Cast input immediately to save memory/compute
    volume = volume.astype(dtype)

    # 1. Normalize Input Shape to 5D: (Batch, Depth, Height, Width, Channel)
    #    This allows uniform handling of both (M, M, M, C) and (B, M, M, M, C)
    orig_ndim = volume.ndim
    if orig_ndim == 4:
        # Add fake batch dim: (D, H, W, C) -> (1, D, H, W, C)
        input_5d = volume[jnp.newaxis, ...]
    elif orig_ndim == 5:
        input_5d = volume
    else:
        raise ValueError(f"Expected 4D or 5D input, got ndim={orig_ndim}")

    B, D, H, W, C = input_5d.shape

    # 2. Parse Sigma
    #    We need robust parsing here because if sigma is a tuple, 
    #    assigning it directly to sigma_d would cause shape errors later.
    if isinstance(sigma, (int, float)):
        sigma_d = sigma_h = sigma_w = float(sigma)
    elif isinstance(sigma, jnp.ndarray):
        if sigma.size == 1:
            sigma_d = sigma_h = sigma_w = sigma
        else:
            sigma_d, sigma_h, sigma_w = sigma
    elif isinstance(sigma, (tuple, list)):
        sigma_d, sigma_h, sigma_w = sigma
    else:
        # Fallback for other iterables
        sigma_d, sigma_h, sigma_w = sigma

    # 3. Define Padding Mapping
    #    JAX pad modes: 'constant', 'edge', 'reflect', 'symmetric', 'wrap'
    pad_mode = mode if mode != 'nearest' else 'edge'  # JAX calls it edge

    # 4. Separable Convolutions
    #    We apply 1D convolution sequentially along Depth, Height, and Width.
    #    We use 'feature_group_count=C' to treat channels independently (Depthwise Conv).

    # --- Pass 1: Depth (Axis 1) ---
    k_d = get_gaussian_kernel1d(sigma_d, radius=radius, dtype=dtype)
    radius_d = k_d.shape[0] // 2

    # Reshape kernel for lax.conv: (Spatial_D, Spatial_H, Spatial_W, In_C, Out_C)
    # For Depth pass: (K, 1, 1, 1, C) with groups=C
    # We tile the 1D kernel C times to match the group count requirement.
    k_d_blob = k_d.reshape(-1, 1, 1, 1, 1)  # (K, 1, 1, 1, 1)
    k_d_blob = jnp.tile(k_d_blob, (1, 1, 1, 1, C))  # (K, 1, 1, 1, C)

    # Pad only the depth dimension
    pad_width = ((0, 0), (radius_d, radius_d), (0, 0), (0, 0), (0, 0))
    padded = jnp.pad(input_5d, pad_width, mode=pad_mode)

    # Apply Convolution
    # dim_numbers=('NDHWC', 'DHWIO', 'NDHWC') ensures we map Input->Output correctly
    input_5d = lax.conv_general_dilated(
        lhs=padded,
        rhs=k_d_blob,
        window_strides=(1, 1, 1),
        padding='VALID',  # We handled padding manually
        lhs_dilation=(1, 1, 1),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=C,
        preferred_element_type=dtype
    )

    # --- Pass 2: Height (Axis 2) ---
    k_h = get_gaussian_kernel1d(sigma_h, radius=radius, dtype=dtype)
    radius_h = k_h.shape[0] // 2

    # Kernel shape: (1, K, 1, 1, C)
    k_h_blob = k_h.reshape(1, -1, 1, 1, 1)
    k_h_blob = jnp.tile(k_h_blob, (1, 1, 1, 1, C))

    pad_width = ((0, 0), (0, 0), (radius_h, radius_h), (0, 0), (0, 0))
    padded = jnp.pad(input_5d, pad_width, mode=pad_mode)

    input_5d = lax.conv_general_dilated(
        lhs=padded,
        rhs=k_h_blob,
        window_strides=(1, 1, 1),
        padding='VALID',
        lhs_dilation=(1, 1, 1),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=C,
        preferred_element_type=dtype
    )

    # --- Pass 3: Width (Axis 3) ---
    k_w = get_gaussian_kernel1d(sigma_w, radius=radius, dtype=dtype)
    radius_w = k_w.shape[0] // 2

    # Kernel shape: (1, 1, K, 1, C)
    k_w_blob = k_w.reshape(1, 1, -1, 1, 1)
    k_w_blob = jnp.tile(k_w_blob, (1, 1, 1, 1, C))

    pad_width = ((0, 0), (0, 0), (0, 0), (radius_w, radius_w), (0, 0))
    padded = jnp.pad(input_5d, pad_width, mode=pad_mode)

    input_5d = lax.conv_general_dilated(
        lhs=padded,
        rhs=k_w_blob,
        window_strides=(1, 1, 1),
        padding='VALID',
        lhs_dilation=(1, 1, 1),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=C,
        preferred_element_type=dtype
    )

    # 5. Restore Original Shape
    if orig_ndim == 4:
        return jnp.squeeze(input_5d, axis=0)

    return input_5d.astype(out_dtype)