import jax
from jax import numpy as jnp, lax as jlx
from jax.scipy.ndimage import map_coordinates
from flax import nnx
import numpy as np
from scipy import signal


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

def wiener2DFilter(images, ctf, pad_factor=2):
    xsize = images.shape[1]

    ctf_2 = ctf * ctf
    epsilon = 0.1 * jnp.mean(ctf_2, axis=(-2, -1), keepdims=True)

    if pad_factor > 1:
        pad_diff = xsize * (pad_factor - 1) // pad_factor
        images = jnp.pad(images, ((0, 0), (pad_diff, pad_diff), (pad_diff, pad_diff)), mode="constant")

    ft_images = jnp.fft.fftshift(jnp.fft.rfft2(images))
    ft_ctf_images_real = ft_images.real * ctf / (ctf_2 + epsilon)
    ft_ctf_images_imag = ft_images.imag * ctf / (ctf_2 + epsilon)
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

