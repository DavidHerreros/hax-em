from jax import random as jnr
from jax._src.nn.initializers import _compute_fans
from jax import numpy as jnp

from flax import nnx


def siren_init(omega=1.0, c=1.0, in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        # shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = jnp.sqrt(c / (3. * fan_in)) / omega
        return jnr.uniform(key, shape, dtype, minval=-variance, maxval=variance)

    return init

def siren_init_original(omega=1.0, c=1.0, in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        # shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = jnp.sqrt(c / fan_in) / omega
        return jnr.uniform(key, shape, dtype, minval=-variance, maxval=variance)

    return init


def siren_init_first(c=1.0, in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        # shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = c * (1. / fan_in)
        return jnr.uniform(key, shape, dtype, minval=-variance, maxval=variance)

    return init


def bias_uniform(in_axis=-2, out_axis=-1, dtype=jnp.float32):
    # this is what Pytorch default Linear uses.
    def init(key, shape, dtype=dtype):
        # shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = jnp.sqrt(1 / fan_in)
        return jnr.uniform(
            key, (int(fan_out),), dtype, minval=-variance, maxval=variance
        )

    return init


def calculate_spectral_centroid_3d(data_grid):
    """
    Computes the spectral centroid of a 3D volume to set s0, s1.
    data_grid: A 3D array of your target (e.g., occupancy or SDF)
    """
    # Compute 3D FFT and Power Spectrum
    fft_data = jnp.abs(jnp.fft.fftn(data_grid)) ** 2

    # Create frequency coordinate grids
    shape = data_grid.shape
    freqs = [jnp.fft.fftfreq(n) for n in shape]
    mesh = jnp.array(jnp.meshgrid(*freqs, indexing='ij'))

    # Calculate radius (frequency magnitude) for each bin
    k_mag = jnp.sqrt(jnp.sum(mesh ** 2, axis=0))

    # Weighted average of frequencies (the centroid)

    centroid = jnp.sum(k_mag * fft_data) / jnp.sum(fft_data)

    # Normalize by max possible frequency (0.5)
    return float(centroid / 0.5)


class Siren2Linear(nnx.Module):
    def __init__(self, in_features, out_features, rngs, is_first=False, custom_init=False, is_residual=False, w0=30.0, s=0.0, dtype=jnp.float32, use_bias=True):
        self.w0 = w0
        self.is_first = is_first
        self.is_residual = is_residual

        # Standard SIREN Initialization
        if is_first:
            kernel_init = siren_init_first(c=1.0)
        else:
            if custom_init:
                kernel_init = siren_init(c=6.0, omega=1.0)
            else:
                kernel_init = siren_init_original(c=6.0, omega=1.0)

        # Create the base linear layer
        self.linear = nnx.Linear(
            in_features, out_features,
            kernel_init=kernel_init,
            rngs=rngs,
            dtype=dtype,
            use_bias=use_bias
        )

        # SIREN2 (WINNER) Perturbation
        if s > 0:
            noise_key = rngs.params()  # Use a distinct key for the noise
            if is_first:
                noise_std = s / jnp.mean(self.w0)
            else:
                noise_std = s / w0
            noise = jnr.normal(noise_key, self.linear.kernel.value.shape) * noise_std
            # Perturb the weights
            self.linear.kernel.value += noise

    def __call__(self, x):
        if self.is_residual:
            return jnp.sin(self.w0 * (x + self.linear(x)))
        else:
            return jnp.sin(self.w0 * self.linear(x))
