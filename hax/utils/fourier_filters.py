import jax
from jax import numpy as jnp, lax as jlx
from jax.scipy.ndimage import map_coordinates
from flax import nnx
import numpy as np
from scipy import signal
from functools import partial


class FastVariableBlur2D(nnx.Module):
    def __init__(self, shape: tuple[int, int]):
        self.h, self.w = shape

        # Precompute ONLY the frequency grid coordinates (constant)
        fy = jnp.fft.fftfreq(self.h)[:, None]  # (H, 1)
        fx = jnp.fft.rfftfreq(self.w)[None, :]  # (1, W/2 + 1)

        # Precompute squared frequency radius
        self.f_sq = fx ** 2 + fy ** 2

    def __call__(self, x: jax.Array, sigma: float) -> jax.Array:
        """
        Args:
            x: Input image batch (B, H, W, C)
            sigma: The blur strength (pixels) for this specific step.
        """
        # Generate Gaussian Mask on-the-fly
        mask = jnp.exp(-2 * jnp.pi ** 2 * sigma ** 2 * self.f_sq)

        # RFFT (Real -> Complex)
        spectrum = jnp.fft.rfft2(x, axes=(1, 2))

        # Apply Mask
        mask = mask[None, ..., None]
        filtered_spectrum = spectrum * mask

        # IRFFT (Complex -> Real)
        return jnp.fft.irfft2(filtered_spectrum, s=(self.h, self.w), axes=(1, 2))

def low_pass_3d(x, std=1.0, kernel_size=9):
    size = x.shape

    n = jnp.arange(kernel_size)
    center = (kernel_size - 1.0) / 2.0
    gauss_1d = jnp.exp(-0.5 * ((n - center) / std) ** 2)
    gauss_1d = gauss_1d / jnp.sum(gauss_1d)
    kernel = jnp.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)

    # Calculate how much padding is needed on each side to reach target_shape
    pad_width = []
    for i in range(3):
        total_pad = size[i] - kernel.shape[i]
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))

    # Pad the small kernel with zeros to match the image size (e.g., 128^3)
    padded_kernel = jnp.pad(kernel, pad_width)

    # Shift the kernel center to [0, 0, 0] to prevent spatial translation
    shifted_kernel = jnp.fft.ifftshift(padded_kernel)

    # Compute the 3D FFT (The kernel will be complex numbers)
    ft_kernel = jnp.fft.fftn(shifted_kernel)

    # Apply kernel Fourier
    ft_x = jnp.fft.fftn(x)
    ft_x = ft_x * ft_kernel
    return jnp.fft.ifftn(ft_x).real


def _radial_frequency_grid(shape, pixel_size_A):
    """|k| in 1/A on the (H, W) grid, in unshifted FFT layout."""
    fy = jnp.fft.fftfreq(shape[0], d=pixel_size_A)  # cycles / A
    fx = jnp.fft.fftfreq(shape[1], d=pixel_size_A)
    ky, kx = jnp.meshgrid(fy, fx, indexing="ij")
    return jnp.sqrt(ky ** 2 + kx ** 2)


def _bandpass_response(k, k_hp, k_lp, filter_type, order):
    """Multiplicative frequency response in [0, 1], same shape as k."""
    hp = jnp.ones_like(k)
    lp = jnp.ones_like(k)

    if filter_type == "gaussian":
        if k_hp is not None:
            hp = 1.0 - jnp.exp(-0.5 * (k / k_hp) ** 2)  # 0 at DC, ->1 above k_hp
        if k_lp is not None:
            lp = jnp.exp(-0.5 * (k / k_lp) ** 2)  # 1 at DC, ->0 above k_lp

    elif filter_type == "butterworth":
        if k_hp is not None:
            safe_k = jnp.where(k > 0, k, 1.0)  # avoid 0/0 at DC
            hp = 1.0 / (1.0 + (k_hp / safe_k) ** (2 * order))
            hp = jnp.where(k > 0, hp, 0.0)  # exactly 0 at DC
        if k_lp is not None:
            lp = 1.0 / (1.0 + (k / k_lp) ** (2 * order))

    else:
        raise ValueError(f"unknown filter_type: {filter_type!r}")

    return hp * lp


@partial(jax.jit, static_argnames=("highpass_A", "lowpass_A", "filter_type", "order"))
def bandpass_filter(image, pixel_size_A, highpass_A=40.0, lowpass_A=None,
                    filter_type="gaussian", order=4):
    """Band-pass a real-space image or stack of images.

    Args:
        image: real array with shape (H, W) or (..., H, W). Leading batch
            dims are handled automatically (jnp.fft.fft2 acts on the last 2).
        pixel_size_A: pixel spacing in Angstrom.
        highpass_A: suppress features COARSER than this (Angstrom) -> kills the
            membrane. Set it to the membrane scale, ~30-50 A. None disables it.
        lowpass_A: suppress features FINER than this (Angstrom) -> kills noise.
            None disables it.
        filter_type: "gaussian" (ringing-free) or "butterworth" (flatter
            passband, sharper edge, slight ringing at high order).
        order: Butterworth order (ignored when filter_type == "gaussian").

    Returns:
        Filtered real array, same shape and dtype family as input.
    """
    k = _radial_frequency_grid(image.shape[-2:], pixel_size_A)
    k_hp = None if highpass_A is None else 1.0 / highpass_A
    k_lp = None if lowpass_A is None else 1.0 / lowpass_A
    resp = _bandpass_response(k, k_hp, k_lp, filter_type, order)  # (H, W)
    f = jnp.fft.fft2(image)  # over last 2 axes, broadcasts batch
    return jnp.real(jnp.fft.ifft2(f * resp))

def bspline_3d(x):
    size = x.shape[0]

    b_spline_1d = np.asarray([0.0, 0.5, 1.0, 0.5, 0.0])

    pad_before = (size - len(b_spline_1d)) // 2
    pad_after = size - pad_before - len(b_spline_1d)

    kernel = np.einsum('i,j,k->ijk', b_spline_1d, b_spline_1d, b_spline_1d)
    kernel = np.pad(kernel, (pad_before, pad_after), 'constant', constant_values=(0.0,))
    kernel = jnp.array(kernel).astype(jnp.complex64)
    ft_kernel = jnp.abs(jnp.fft.fftshift(jnp.fft.fftn(kernel)))

    # Apply kernel Fourier
    ft_x = jnp.fft.fftshift(jnp.fft.fftn(x))
    ft_x_real = ft_x.real * ft_kernel
    ft_x_imag = ft_x.imag * ft_kernel
    ft_x = jlx.complex(ft_x_real, ft_x_imag)
    return jnp.fft.ifftn(jnp.fft.ifftshift(ft_x)).real

def fourier_resize(x, new_size):
    """
    Resize tensor using Fourier transform. Supports 4D and 5D tensors.

    :param x: A 4D (B, H, W, C) or 5D (B, D, H, W, C) tensor.
    :param new_size: A tuple indicating the new size (new_d, new_h, new_w) for 5D or (new_h, new_w) for 4D. It
    could also be an integer to specify equal resizing for all dimensions.
    :return: Resized tensor.
    """
    original_shape = x.shape
    num_dims = len(original_shape)

    # Check if the tensor is 4D or 5D
    if num_dims not in [4, 5]:
        raise ValueError("Input tensor must be 4D or 5D.")

    # Check new_size param
    if isinstance(new_size, int):
        if num_dims == 5:
            new_size = (new_size, new_size, new_size)
        else:
            new_size = (new_size, new_size)

    # FFT operation
    if num_dims == 5:
        B, D, H, W, C = original_shape
        new_d, new_h, new_w = new_size
        f_x = jnp.fft.fftn(x, axes=(1, 2, 3))
        resized_f_x = jnp.zeros((B, new_d, new_h, new_w, C), dtype=jnp.complex64)
    else:
        B, H, W, C = original_shape
        new_h, new_w = new_size
        f_x = jnp.fft.fftn(x, axes=(1, 2))
        resized_f_x = jnp.zeros((B, new_h, new_w, C), dtype=jnp.complex64)

    # Central crop/padding for resizing
    slicing = tuple(
        slice(max((old - new) // 2, 0), max((old - new) // 2, 0) + min(new, old))
        for old, new in zip(original_shape[-num_dims + 2:], new_size)
    )
    padding = tuple(
        slice(max((new - old) // 2, 0), max((new - old) // 2, 0) + min(new, old))
        for old, new in zip(original_shape[-num_dims + 2:], new_size)
    )

    # Resizing in Fourier domain
    if num_dims == 5:
        resized_f_x.at[:, padding[0], padding[1], padding[2], :].set(f_x[:, slicing[0], slicing[1], slicing[2], :])
    else:
        resized_f_x.at[:, padding[0], padding[1], :].set(f_x[:, slicing[0], slicing[1], :])

    # Inverse FFT and conversion to real
    if num_dims == 5:
        resized_x = jnp.fft.ifftn(resized_f_x, axes=(1, 2, 3)).real
    else:
        resized_x = jnp.fft.ifftn(resized_f_x, axes=(1, 2)).real

    return resized_x

def _radial_bins(shape):
    """Integer radial-shell index for each pixel of a ``fftshift(rfft2(.))`` grid.

    ``shape`` is the (P, Qr) spatial layout of the half-spectrum (Qr = P // 2 + 1).
    Returns the per-pixel shell index and the number of shells, both matching the
    frequency layout used by :func:`wiener2DFilter` (fftfreq on axis -2, rfftfreq
    on axis -1, followed by an ``fftshift`` over both spatial axes).
    """
    P, Qr = shape
    fy = jnp.fft.fftfreq(P)[:, None]        # (P, 1)   cycles / pixel
    fx = jnp.fft.rfftfreq(P)[None, :]       # (1, Qr)  (image is square)
    r_pix = jnp.sqrt((fy * P) ** 2 + (fx * P) ** 2)
    r_pix = jnp.fft.fftshift(r_pix)         # match the spatial fftshift of ctf / ft
    nbins = P // 2 + 1
    r_bin = jnp.clip(jnp.round(r_pix).astype(jnp.int32), 0, nbins - 1)
    return r_bin, nbins


def _spectral_wiener_epsilon(ft_images, ctf_2, reg_frac=1e-2, noise_band=0.4, noise_quantile=0.25):
    """Frequency-dependent Wiener regularizer ``eps(k) = <CTF^2>_shell(k) / SSNR_obs(k)``.

    The signal-to-noise ratio is estimated per image from the radially-averaged
    observed power spectrum: the noise floor is read off the low quantile of the
    outer shells (the CTF-zero troughs there expose the pure noise level), and
    ``SSNR_obs(k) = relu(P_obs(k) - P_noise) / P_noise``. The regularizer uses the
    shell-averaged CTF^2 (smooth and positive at per-pixel CTF zeros), while the
    caller keeps the per-pixel CTF in the filter numerator so the exact zeros stay
    at zero gain. Replaces the previous flat ``0.1 * mean(CTF^2)`` (white-noise)
    term, which under-regularized the high frequencies and blew up the value range.
    """
    P, Qr = ft_images.shape[-2], ft_images.shape[-1]
    ctf_2 = jnp.broadcast_to(ctf_2, ft_images.shape)
    r_bin, nbins = _radial_bins((P, Qr))

    # Radial averaging via segment_sum over the flattened shell index (no dense
    # membership matrix -> memory scales with the spectrum, not spectrum x shells).
    seg = r_bin.reshape(-1)
    counts = jax.ops.segment_sum(jnp.ones_like(seg, dtype=ft_images.real.dtype), seg, num_segments=nbins) + 1e-8

    def radial(power):
        pw = power.reshape(power.shape[0], -1).T                 # (N, batch)
        return (jax.ops.segment_sum(pw, seg, num_segments=nbins) / counts[:, None]).T

    p_obs = radial(ft_images.real ** 2 + ft_images.imag ** 2)   # (batch, nbins)
    h2 = radial(ctf_2)                                           # (batch, nbins)

    # Noise floor from the outer shells (robust low quantile catches the troughs).
    k_lo = int(noise_band * nbins)
    p_noise = jnp.quantile(p_obs[:, k_lo:], noise_quantile, axis=1, keepdims=True)

    # Floor the noise estimate to a tiny fraction of the observed signal power.
    # On noiseless (e.g. simulated) data the outer shells - and the zero padding
    # added by the caller - have exactly zero power, so p_noise -> 0; without this
    # floor the shells where p_obs is also 0 give nsr = (h2*0)/(0+0) = 0/0 = NaN,
    # which propagates through the Wiener gain into the loss and then produces
    # NaN Gaussian coordinates (illegal-memory scatter). Scaled to the signal so
    # it stays scale-invariant and is negligible on real, noisy data.
    p_scale = jnp.maximum(jnp.max(p_obs, axis=1, keepdims=True), 1e-12)
    p_noise = jnp.maximum(p_noise, 1e-6 * p_scale)

    ssnr_num = jnp.maximum(p_obs - p_noise, 0.0)                 # (batch, nbins)
    nsr = h2 * p_noise / (ssnr_num + reg_frac * p_noise)         # eps(k), capped at h2 / reg_frac

    return nsr[:, r_bin]                                         # (batch, P, Qr)


def wiener2DFilter(images, ctf, pad_factor=2, epsilon=None, reg_frac=1e-2,
                   noise_band=0.4, noise_quantile=0.25):
    """Regularized CTF inversion (Wiener deconvolution).

    By default the regularizer is a per-image, frequency-dependent spectral SNR
    (see :func:`_spectral_wiener_epsilon`), which keeps the output range realistic
    by suppressing noise amplification where the SNR collapses. Pass an explicit
    ``epsilon`` (scalar or broadcastable array) to fall back to a fixed
    regularization, e.g. ``0.1 * jnp.mean(ctf * ctf, axis=(-2, -1), keepdims=True)``
    for the previous flat white-noise behaviour.
    """
    xsize = images.shape[1]

    ctf_2 = ctf * ctf

    if pad_factor > 1:
        pad_diff = xsize * (pad_factor - 1) // pad_factor
        images = jnp.pad(images, ((0, 0), (pad_diff, pad_diff), (pad_diff, pad_diff)), mode="constant")

    ft_images = jnp.fft.fftshift(jnp.fft.rfft2(images))

    if epsilon is None:
        # The regularizer is a per-batch noise-model *statistic* (built from
        # radial quantiles/maxima of the observed power). Differentiating through
        # it is both wrong and numerically unstable: for a near-zero input (e.g.
        # the model projections early in training) the gradient of the quantile /
        # max over the degenerate all-equal spectrum is NaN, which then poisons
        # the whole update. Stop-gradient keeps epsilon adaptive per step while
        # letting gradients flow only through the actual filtering, matching the
        # old fixed-epsilon behaviour.
        epsilon = jax.lax.stop_gradient(
            _spectral_wiener_epsilon(ft_images, ctf_2, reg_frac, noise_band, noise_quantile))

    wiener_gain = ctf / (ctf_2 + epsilon)
    ft_ctf_images_real = ft_images.real * wiener_gain
    ft_ctf_images_imag = ft_images.imag * wiener_gain
    ft_ctf_images = jlx.complex(ft_ctf_images_real, ft_ctf_images_imag)
    images = jnp.fft.irfft2(jnp.fft.ifftshift(ft_ctf_images))

    if pad_factor > 1:
        images = images[:, pad_diff:-pad_diff, pad_diff:-pad_diff]

    return images

def ctfFilter(images, ctf, pad_factor=2):
    xsize = images.shape[1]

    if pad_factor > 1:
        pad_diff = xsize * (pad_factor - 1) // pad_factor
        images = jnp.pad(images, ((0, 0), (pad_diff, pad_diff), (pad_diff, pad_diff)), mode="constant")

    ft_images = jnp.fft.fftshift(jnp.fft.rfft2(images))
    ft_ctf_images_real = ft_images.real * ctf
    ft_ctf_images_imag = ft_images.imag * ctf
    ft_ctf_images = jlx.complex(ft_ctf_images_real, ft_ctf_images_imag)
    images = jnp.fft.irfft2(jnp.fft.ifftshift(ft_ctf_images))

    if pad_factor > 1:
        images = images[:, pad_diff:-pad_diff, pad_diff:-pad_diff]

    return images

def rfft2_padded(images, pad_factor=2):
    xsize = images.shape[1] if images.ndim > 2 else images.shape[0]

    pad_diff = xsize * (pad_factor - 1) // pad_factor
    if images.ndim > 2:
        images = jnp.pad(images, ((0, 0), (pad_diff, pad_diff), (pad_diff, pad_diff)), mode="constant")
    else:
        images = jnp.pad(images, ((pad_diff, pad_diff), (pad_diff, pad_diff)), mode="constant")

    return jnp.fft.fftshift(jnp.fft.rfft2(images))

def irfft2_padded(ft_images, pad_factor=2):
    pad_factor_inv = 1. / pad_factor
    xsize = ft_images.shape[1] if ft_images.ndim > 2 else ft_images.shape[0]
    pad_diff = int(xsize * (1. - pad_factor_inv) // 2)

    images = jnp.fft.irfft2(jnp.fft.ifftshift(ft_images))
    if images.ndim > 2:
        images = images[:, pad_diff:-pad_diff, pad_diff:-pad_diff]
    else:
        images = images[pad_diff:-pad_diff, pad_diff:-pad_diff]

    return images

def fourier_slice_interpolator(
        volumes: jax.Array,
        rotations: jax.Array,
        shifts: jax.Array
) -> jax.Array:
    """
    Generates a single projection for each volume in a batch, using a
    corresponding rotation and shift for each.

    Args:
        volumes (jax.Array): A batch of 3D volumes.
            Shape: `(N, M, M, M)`.
        rotations (jax.Array): A batch of 3x3 rotation matrices.
            Shape: `(N, 3, 3)`.
        shifts (jax.Array): A batch of 2D shifts (dy, dx) in pixels.
            Shape: `(N, 2)`.

    Returns:
        jax.Array: The generated 2D projections. Shape: `(N, M, M)`.
    """
    # Assert that the batch dimension N is consistent across inputs.
    N = volumes.shape[0]
    assert rotations.shape[0] == N and shifts.shape[0] == N, "Batch dimensions must match."

    # Define the projection logic for a single item.
    # This function will be vectorized over the batch dimension N.
    def _project_one(volume, rotation, shift):
        M = volume.shape[-1]

        # Create the base 2D grid for slicing
        grid_1d = jnp.arange(-(M // 2), M // 2 + (M % 2), dtype=jnp.float32)
        x, y = jnp.meshgrid(grid_1d, grid_1d, indexing='ij')
        slice_coords = jnp.stack([x, y, jnp.zeros_like(x)], axis=0)

        # 1. Get Fourier Slice (Rotation)
        f_volume = jnp.fft.fftn(volume)
        f_volume_shifted = jnp.fft.fftshift(f_volume)

        rotated_coords = (rotation @ slice_coords.reshape(3, -1)).reshape(3, M, M)
        sampling_coords = rotated_coords + (M - 1) / 2.0

        real_slice = map_coordinates(f_volume_shifted.real, sampling_coords, order=1, mode='constant', cval=0.0)
        imag_slice = map_coordinates(f_volume_shifted.imag, sampling_coords, order=1, mode='constant', cval=0.0)
        ft_slice = real_slice + 1j * imag_slice

        # 2. Apply Phase Shift (Translation)
        ky, kx = jnp.fft.fftfreq(M), jnp.fft.fftfreq(M)
        k_coords = jnp.stack(jnp.meshgrid(ky, kx, indexing='ij'), axis=0)

        phase_dot_product = jnp.einsum('i,ixy->xy', shift, k_coords)
        phase_shift = jnp.exp(-1j * 2 * jnp.pi * phase_dot_product)
        shifted_ft_slice = ft_slice * phase_shift

        # 3. Inverse FFT
        ft_slice_unshifted = jnp.fft.ifftshift(shifted_ft_slice)
        projection_complex = jnp.fft.ifft2(ft_slice_unshifted)

        return projection_complex.real

    # Vectorize the projection function over the batch dimension (axis 0) for all inputs.
    return jax.vmap(_project_one)(volumes, rotations, shifts)

