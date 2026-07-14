"""Consensus reconstruction from posed particles (Wiener-filtered Fourier gridding).

The images already carry poses, so a consensus map can be recovered directly, without
any iterative refinement: each particle contributes a central slice of the 3D Fourier
transform (the projection-slice theorem), so the map is obtained by inserting every
CTF-weighted slice at its posed orientation and dividing by the accumulated CTF power::

    V(k) = sum_i CTF_i(k) F_i(k)  /  ( sum_i CTF_i(k)^2 + tau )

This is the standard gridding reconstruction: one streaming pass, no iteration, and it
reaches whatever resolution the data supports.

Three properties are deliberately engineered here.

**It respects the line integral.** The discrete projection-slice identity carries no
scale factor -- ``fft2(sum_z V)[f0, f1] == fftn(V)[0, f0, f1]`` exactly -- and the Wiener
quotient above is the least-squares fit of the map to the very images it was built from.
So summing the reconstruction along an axis reproduces a real projection, and re-projecting
it reproduces the contrast and value range of the input images. A small residual shrinkage
(from ``tau`` and from trilinear interpolation) is removed by an explicit global gray-scale
calibration, computed in closed form from the accumulators at no extra cost.

**It knows its own resolution.** The particles are split into two independent halves and
reconstructed separately, and the FSC between the half-maps gives the spectral
signal-to-noise as a function of frequency. That is a gold-standard estimate: it assumes
nothing about the noise, it is simply measured.

**It denoises optimally.** The half-map FSC is turned into the MMSE filter
``C(k) = sqrt(2 FSC / (1 + FSC))`` and applied to the combined map. Shells where the
signal is real (FSC -> 1) pass untouched, so no resolution is thrown away; shells that are
noise (FSC -> 0) are driven to zero, so none is kept. This is what makes the map usable as
a reference: noise in it would otherwise propagate straight into the mask, the fitted point
cloud, and the density gauge's gradients.

Geometry (matching ``PhysDecoder`` / ``tests.phantom._project_batch``):

* real space ``p = R c`` for a 3D point ``c`` in component order ``(x, y, z)``;
  ``image[row, col] = image[p_y, p_x]`` and the projection runs along ``p_z``;
  the volume array is indexed ``[z, y, x]``.
* hence a 2D Fourier coefficient at image frequency ``(f0, f1) = (f_row, f_col)`` lies on
  the central slice at ``R^T (f1, f0, 0)`` in component order, whose index in the
  ``[z, y, x]``-ordered volume transform is that vector **reversed**.

``ifftshift`` before every FFT is not cosmetic. The object is centred at ``box // 2``, so
an un-shifted transform carries a ``(-1)^k`` phase ramp. It cancels for the identity pose
but *not* for a rotated one (the rotated ``k`` is not integral), which silently destroys
every rotated slice while leaving the identity case looking perfect.
"""

import sys
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp

from .ctf import eval_ctf
from .euler import euler_matrix_batch
from .loggers import bcolors

__all__ = ["reconstruct_consensus_volume", "consensus_mask"]


def _slice_geometry(box, sr):
    """Centred frequency grids and the identity-pose central slice, in index units."""
    g = (np.fft.fftshift(np.fft.fftfreq(box)) * box).astype(np.float32)
    f0, f1 = np.meshgrid(g, g, indexing="ij")                     # f0 <-> row, f1 <-> col
    s = np.sqrt((f0 / box) ** 2 + (f1 / box) ** 2) / sr           # spatial frequency, 1/A
    a = np.arctan2(f0, f1)                                        # azimuth, for astigmatism
    k_rot = np.stack([f1, f0, np.zeros_like(f0)], 0)              # component order (x, y, z)
    return (jnp.asarray(s), jnp.asarray(a), jnp.asarray(k_rot),
            jnp.asarray(f0), jnp.asarray(f1))


@partial(jax.jit, static_argnums=(7,))
def _insert_slices(num, den, images, rotations, shifts, ctf, k_rot, box, f0, f1):
    """Accumulate one batch of CTF-weighted central slices into the 3D transform."""
    ft = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(images, axes=(-2, -1))), axes=(-2, -1))

    # Re-centre each particle: the projector places its content at
    # (row, col) = (p_y - shift_y, p_x - shift_x), so divide the corresponding phase out.
    phase = jnp.exp(-2j * jnp.pi * (shifts[:, 1, None, None] * f0[None]
                                    + shifts[:, 0, None, None] * f1[None]) / box)
    ft = ft * phase

    data = ft * ctf
    weight = ctf ** 2

    k = jnp.einsum("bji,jhw->bihw", rotations, k_rot)             # R^T @ (f1, f0, 0), components
    k = jnp.stack([k[:, 2], k[:, 1], k[:, 0]], 1)                 # reversed -> volume array order

    def scatter(num, den, coords, value, weight):
        pos = coords + box // 2
        base = jnp.floor(pos).astype(jnp.int32)
        frac = pos - base
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    idx = base + jnp.array([dz, dy, dx], jnp.int32)[None, :, None, None]
                    w = ((frac[:, 0] if dz else 1.0 - frac[:, 0]) *
                         (frac[:, 1] if dy else 1.0 - frac[:, 1]) *
                         (frac[:, 2] if dx else 1.0 - frac[:, 2]))
                    w = w * jnp.all((idx >= 0) & (idx <= box - 1), axis=1)
                    z, y, x = [jnp.clip(idx[:, i], 0, box - 1) for i in range(3)]
                    num = num.at[z, y, x].add(value * w)
                    den = den.at[z, y, x].add(weight * w)
        return num, den

    num, den = scatter(num, den, k, data, weight)
    # Friedel mate F(-k) = conj(F(k)): doubles the coverage and forces a real volume.
    num, den = scatter(num, den, -k, jnp.conj(data), weight)
    return num, den


def _gridding_correction(box):
    """Trilinear insertion convolves the transform with a triangle kernel, which multiplies
    the real-space volume by sinc^2. Divide it back out."""
    r = np.fft.fftshift(np.fft.fftfreq(box))
    z, y, x = np.meshgrid(r, r, r, indexing="ij")
    return np.maximum((np.sinc(x) * np.sinc(y) * np.sinc(z)) ** 2, 1e-2)


def _shell_index(box):
    """Radial shell index of every voxel of the centred 3D transform."""
    g = np.fft.fftshift(np.fft.fftfreq(box)) * box
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    return np.clip(np.rint(np.sqrt(x ** 2 + y ** 2 + z ** 2)).astype(int), 0, box // 2)


def _invert(num, den, tau, box):
    """Wiener quotient -> real volume, with the gridding envelope removed."""
    volume_ft = num / (den + tau * jnp.mean(den))
    volume = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(volume_ft))))
    return np.asarray(volume) / _gridding_correction(box)


def _half_map_fsc(vol_a, vol_b, shells, n_shells):
    """FSC between two independently reconstructed half-maps."""
    fa = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol_a)))
    fb = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol_b)))
    fsc = np.zeros(n_shells, np.float64)
    for i in range(n_shells):
        m = shells == i
        if not m.any():
            continue
        cross = np.sum(fa[m] * np.conj(fb[m])).real
        norm = np.sqrt(np.sum(np.abs(fa[m]) ** 2) * np.sum(np.abs(fb[m]) ** 2))
        fsc[i] = cross / norm if norm > 0 else 0.0
    return np.clip(fsc, 0.0, 1.0)


def _fsc_filter(fsc):
    """MMSE filter for the combined map given the half-map FSC.

    ``C = sqrt(2 FSC / (1 + FSC))`` is the standard relation between the half-map FSC and
    the optimal filter for the full map (which has twice the particles, hence twice the
    SSNR). Everything past the first shell that drops to zero correlation is truncated:
    beyond it there is nothing but noise, and letting an upward FSC fluctuation resurrect a
    shell is how noise gets back into a "denoised" map.
    """
    fsc = np.asarray(fsc, np.float64).copy()
    dead = np.flatnonzero(fsc <= 0.0)
    if dead.size:
        fsc[dead[0]:] = 0.0
    return np.sqrt(np.maximum(2.0 * fsc / (1.0 + fsc), 0.0))


def _resolution(fsc, box, sr, threshold=0.143):
    """Resolution (A) at which the half-map FSC first falls below ``threshold``."""
    for i in range(1, len(fsc)):
        if fsc[i] < threshold:
            return (box * sr) / i if i > 0 else float("inf")
    return 2.0 * sr


@partial(jax.jit, static_argnums=(5,))
def _project(volume_ft, rotations, shifts, ctf, k_rot, box, f0, f1):
    """Forward-project the map: the exact adjoint of ``_insert_slices``.

    Extract each central slice, re-apply that particle's CTF and in-plane shift, and invert.
    The result is what the map predicts the particle should look like.
    """
    k = jnp.einsum("bji,jhw->bihw", rotations, k_rot)
    k = jnp.stack([k[:, 2], k[:, 1], k[:, 0]], 1) + box // 2

    def one(coords):
        re = jax.scipy.ndimage.map_coordinates(volume_ft.real, coords, order=1, mode="constant")
        im = jax.scipy.ndimage.map_coordinates(volume_ft.imag, coords, order=1, mode="constant")
        return re + 1j * im

    ft = jax.vmap(one)(k) * ctf
    phase = jnp.exp(2j * jnp.pi * (shifts[:, 1, None, None] * f0[None]
                                   + shifts[:, 0, None, None] * f1[None]) / box)
    ft = ft * phase
    return jnp.real(jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(ft, axes=(-2, -1))), axes=(-2, -1)))


def _gray_scale(volume, md, columns, sr, box, has_ctf, k_rot, f0, f1, s, a, n_probe=2000,
                batch_size=256):
    """Global scale putting the map's projections on the gray scale of the input images.

    Measured, not derived: forward-project the finished map at a subset of the real poses,
    re-apply each CTF and shift, and take the least-squares scale against the actual images.
    A closed form in terms of the accumulators looks tempting but is wrong -- the insertion
    and extraction kernels are adjoint rather than identical, and the shells with almost no
    CTF power dominate the sum while carrying no reliable scale information.

    The residual it removes (a few per cent to ~15%) comes from the Wiener floor and from
    the interpolation, and it matters: the reference's projections are compared directly
    against the images during training, where a systematic amplitude error cannot be undone.
    """
    n = min(int(n_probe), len(md))
    if n < 8:
        return 1.0

    idx = np.linspace(0, len(md) - 1, n).astype(np.int64)
    volume_ft = jnp.asarray(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(np.asarray(volume)))))
    angles = np.asarray(columns["euler_angles"], np.float32)
    shifts = np.asarray(columns["shifts"], np.float32)
    kv = float(np.asarray(columns["ctfVoltage"]).ravel()[0]) if has_ctf else 0.0

    cross = 0.0
    energy = 0.0
    for start in range(0, n, batch_size):
        chunk = idx[start:start + batch_size]
        images = np.asarray(md.getMetaDataImage(chunk), np.float32)

        ang = jnp.asarray(angles[chunk])
        rotations = euler_matrix_batch(ang[:, 0], ang[:, 1], ang[:, 2])
        if has_ctf:
            b = chunk.shape[0]
            ctf = eval_ctf(jnp.tile(s[None], (b, 1, 1)), jnp.tile(a[None], (b, 1, 1)),
                           jnp.asarray(np.asarray(columns["ctfDefocusU"])[chunk]),
                           jnp.asarray(np.asarray(columns["ctfDefocusV"])[chunk]),
                           angast=jnp.asarray(np.asarray(columns["ctfDefocusAngle"])[chunk]),
                           cs=jnp.asarray(np.asarray(columns["ctfSphericalAberration"])[chunk]),
                           kv=kv)
        else:
            ctf = jnp.ones((chunk.shape[0], box, box), jnp.float32)

        predicted = _project(volume_ft, rotations, jnp.asarray(shifts[chunk]), ctf,
                             k_rot, box, f0, f1)
        cross += float(jnp.sum(predicted * jnp.asarray(images)))
        energy += float(jnp.sum(predicted ** 2))

    return cross / energy if energy > 0 else 1.0


def _read_chunk(md, start, stop):
    return np.asarray(md.getMetaDataImage(np.arange(start, stop, dtype=np.int64)), np.float32)


def reconstruct_consensus_volume(md, columns, sr, tau=0.05, batch_size=1024, threads=8,
                                 use_ctf=True, denoise=True, calibrate_gray_scale=True,
                                 quiet=False):
    """Reconstruct a consensus volume from posed particles in a single streaming pass.

    ``md`` is an ``XmippMetaData``; ``columns`` the dict from ``extract_columns`` (it
    supplies ``euler_angles``, ``shifts`` and, when present, the CTF parameters).
    ``use_ctf`` must reflect the data: weighting CTF-free images by a CTF is not a harmless
    no-op, it reweights the slices and degrades the map.

    ``tau`` is only a numerical floor for the Wiener quotient -- the resolution is set by
    the data, not by this. With ``denoise`` the particles are split into two halves, the
    FSC between the half-maps measures the spectral signal-to-noise, and the combined map
    is filtered so that shells carrying signal pass untouched and shells that are noise are
    removed.

    Returns the volume; when ``denoise`` it also prints the measured resolution.
    Images are read in chunks on a thread pool while the GPU accumulates, so peak RAM
    tracks ``batch_size`` rather than the particle count.
    """
    n = len(md)
    box = int(md.getMetaDataImage(0).shape[0])

    s, a, k_rot, f0, f1 = _slice_geometry(box, sr)
    zeros_c = jnp.zeros((box,) * 3, jnp.complex64)
    zeros_f = jnp.zeros((box,) * 3, jnp.float32)
    # Two independent half-sets: same total insertion work, but the FSC between them is what
    # tells us how far the data actually goes.
    num = [zeros_c, zeros_c]
    den = [zeros_f, zeros_f]

    angles = np.asarray(columns["euler_angles"], np.float32)
    shifts = np.asarray(columns["shifts"], np.float32)
    has_ctf = use_ctf and "ctfDefocusU" in columns
    if has_ctf:
        kv = float(np.asarray(columns["ctfVoltage"]).ravel()[0])

    if not quiet:
        print(f"{bcolors.OKCYAN}\n###### Reconstructing consensus volume from {n} posed particles... ######{bcolors.ENDC}")

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [(s0, pool.submit(_read_chunk, md, s0, min(s0 + batch_size, n)))
                   for s0 in range(0, n, batch_size)]
        for start, future in tqdm(futures, file=sys.stdout, ascii=" >=", colour="green", disable=quiet):
            images = future.result()
            stop = start + images.shape[0]

            for half in (0, 1):
                # Interleave the halves so both see the same pose and defocus distribution.
                sel = np.arange(start, stop) % 2 == half
                if not sel.any():
                    continue
                idx = np.arange(start, stop)[sel]

                ang = jnp.asarray(angles[idx])
                rotations = euler_matrix_batch(ang[:, 0], ang[:, 1], ang[:, 2])

                if has_ctf:
                    b = idx.shape[0]
                    ctf = eval_ctf(jnp.tile(s[None], (b, 1, 1)), jnp.tile(a[None], (b, 1, 1)),
                                   jnp.asarray(np.asarray(columns["ctfDefocusU"])[idx]),
                                   jnp.asarray(np.asarray(columns["ctfDefocusV"])[idx]),
                                   angast=jnp.asarray(np.asarray(columns["ctfDefocusAngle"])[idx]),
                                   cs=jnp.asarray(np.asarray(columns["ctfSphericalAberration"])[idx]),
                                   kv=kv)
                else:
                    ctf = jnp.ones((idx.shape[0], box, box), jnp.float32)

                num[half], den[half] = _insert_slices(
                    num[half], den[half], jnp.asarray(images[sel]), rotations,
                    jnp.asarray(shifts[idx]), ctf, k_rot, box, f0, f1)

    total_num, total_den = num[0] + num[1], den[0] + den[1]
    volume = _invert(total_num, total_den, tau, box)

    if denoise:
        shells = _shell_index(box)
        n_shells = box // 2 + 1
        fsc = _half_map_fsc(_invert(num[0], den[0], tau, box),
                            _invert(num[1], den[1], tau, box), shells, n_shells)
        resolution = _resolution(fsc, box, sr)

        # Apply the MMSE filter shell by shell, in Fourier space.
        curve = _fsc_filter(fsc)
        v_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(volume)))
        volume = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(v_ft * curve[shells]))))

        if not quiet:
            print(f"{bcolors.OKGREEN}Half-map FSC = 0.143 at {resolution:.1f} A "
                  f"(Nyquist {2.0 * sr:.1f} A); the map is filtered to that limit.{bcolors.ENDC}")

    if calibrate_gray_scale:
        scale = _gray_scale(volume, md, columns, sr, box, has_ctf, k_rot, f0, f1, s, a)
        volume = volume * scale
        if not quiet:
            print(f"{bcolors.OKGREEN}Gray-scale calibrated to the input images (x{scale:.3f}); "
                  f"projecting the map reproduces their contrast.{bcolors.ENDC}")

    return np.asarray(volume, np.float32)


def consensus_mask(volume, threshold=0.02, dilate=2):
    """A binary mask of the protein region of a reconstructed consensus volume.

    ``threshold`` is a fraction of the volume's maximum, applied after a light blur so the
    mask is connected rather than speckled; ``dilate`` grows it by that many voxels so the
    deformation has somewhere to move into.
    """
    from scipy.ndimage import gaussian_filter, binary_dilation

    smooth = gaussian_filter(np.asarray(volume, np.float32), 1.5)
    mask = smooth > threshold * smooth.max()
    if dilate and dilate > 0:
        mask = binary_dilation(mask, iterations=int(dilate))
    return mask.astype(np.float32)
