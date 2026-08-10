"""
fourier_residual_jax.py

JAX / jitted version of `compute_fourier_residual` (drop-in compatible with
`fourier_residual_batched.py`). Meant to be called inside a JAX training loop
(e.g. flax nnx) without breaking the XLA compilation pipeline.

Public API unchanged with respect to the numpy version:

    result = compute_fourier_residual(projection, experimental, n_shells=n_shells)
    result.frc_curve        # (B, n_shells)
    result.residual_map     # same shape as input (e.g. (B,H,W,1))
    result.weight_map       # same shape as input
    result.ssnr_curve       # (B, n_shells)
    result.wiener_weight_curve  # (B, n_shells)
    result.whitening_curve  # (B, n_shells)
    result.residual_energy  # (B,)
    result.freqs            # (n_shells,) — numpy, not jnp (it's static)

`resolution_at_frc_threshold` is NO LONGER part of the main result (it was a
diagnostic with dynamic branching that isn't jittable) — see the separate
`compute_resolution_diagnostic` function at the bottom of the file, to be
called only when you need that number (e.g. in the one-shot debug block),
not on every training step, so you don't force a device->host sync on every
call.

Why it's jitted
----------------
- All numerical operations (FFT, radial reductions, weights, residual) are
  in `jax.numpy`, traceable and compilable by XLA.
- `n_shells`, the `apply_wiener`/`apply_whitening` flags, and whether a
  `noise_mask` is present are marked as static arguments (`static_argnames`):
  they determine the *structure* of the computation (shape, branching), not
  data you differentiate through, so they must be fixed at compile time.
- `shell_idx` (the grid of radial shell membership) depends only on
  (H, W, n_shells), all static values under jit — it is therefore computed
  with plain `numpy` *during tracing* and ends up baked in as a constant in
  the XLA graph, at zero runtime cost.
"""

from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp


class FourierResidualResult(NamedTuple):
    freqs: np.ndarray                 # (n_shells,) — numpy, static
    frc_curve: jnp.ndarray             # (B, n_shells)
    ssnr_curve: jnp.ndarray            # (B, n_shells)
    wiener_weight_curve: jnp.ndarray   # (B, n_shells)
    whitening_curve: jnp.ndarray       # (B, n_shells)
    weight_map: jnp.ndarray            # original input shape
    F_residual: jnp.ndarray            # (B, H, W) complex
    residual_map: jnp.ndarray          # original input shape
    residual_energy: jnp.ndarray       # (B,)


# ---------------------------------------------------------------------------
# Shape handling (host-side, outside jit — purely structural)
# ---------------------------------------------------------------------------
def _normalize_input(x):
    x = jnp.asarray(x)
    if x.ndim == 4:
        if x.shape[-1] != 1:
            raise ValueError(f"Expected a single grayscale channel (last dim == 1), got shape={x.shape}.")
        return x[..., 0], 4
    elif x.ndim == 3:
        return x, 3
    elif x.ndim == 2:
        return x[None, ...], 2
    else:
        raise ValueError(f"Unsupported ndim: {x.ndim} (expected 2, 3, or 4).")


def _restore_shape(arr, orig_ndim):
    if orig_ndim == 4:
        return arr[..., None]
    elif orig_ndim == 2:
        return arr[0]
    else:
        return arr


def _normalize_mask(mask, B, H, W):
    if mask is None:
        return None
    mask = jnp.asarray(mask)
    if mask.ndim == 4:
        mask = mask[..., 0]
    if mask.ndim == 2:
        mask = jnp.broadcast_to(mask, (B, H, W))
    elif mask.ndim == 3:
        if mask.shape != (B, H, W):
            raise ValueError(f"noise_mask shape {mask.shape} incompatible with {(B, H, W)}.")
    else:
        raise ValueError(f"Unsupported noise_mask ndim: {mask.ndim}.")
    return mask.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Radial geometry — computed with plain numpy (depends only on static
# shapes, gets "frozen" as a constant in the XLA graph during tracing)
# ---------------------------------------------------------------------------
def _radial_shell_indices_np(H, W, n_shells):
    fy = np.fft.fftshift(np.fft.fftfreq(H))
    fx = np.fft.fftshift(np.fft.fftfreq(W))
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(kx ** 2 + ky ** 2)

    max_r = radius.max()
    shell_idx = np.floor(radius / max_r * n_shells).astype(np.int32)
    shell_idx = np.clip(shell_idx, 0, n_shells - 1)
    shell_centers = (np.arange(n_shells) + 0.5) / n_shells * max_r
    return shell_idx, shell_centers


def _batched_radial_sum(values, shell_idx_flat, n_shells, B):
    """
    values: (B, H, W) real jnp array.
    shell_idx_flat: (H*W,) integer array, static constant.
    Returns: (B, n_shells).
    """
    HW = shell_idx_flat.shape[0]
    combined_idx = (jnp.arange(B)[:, None] * n_shells + shell_idx_flat[None, :]).reshape(-1)
    weights = values.reshape(B, HW).reshape(-1)
    sums = jnp.bincount(combined_idx, weights=weights, length=B * n_shells)
    return sums.reshape(B, n_shells)


# ---------------------------------------------------------------------------
# Jitted core
# ---------------------------------------------------------------------------
@partial(
    jax.jit,
    static_argnames=("n_shells", "apply_wiener", "apply_whitening", "use_mask", "high_freq_fraction"),
)
def _compute_fourier_residual_core(
    proj_arr,          # (B, H, W) float
    exp_arr,           # (B, H, W) float
    noise_mask_arr,    # (B, H, W) float — ignored if use_mask=False (can be a dummy)
    n_shells: int,
    apply_wiener: bool,
    apply_whitening: bool,
    use_mask: bool,
    high_freq_fraction: float = 0.2,
    eps: float = 1e-8,
):
    B, H, W = proj_arr.shape

    # --- radial geometry, constant (numpy, frozen at compile time) ---
    shell_idx_np, shell_centers_np = _radial_shell_indices_np(H, W, n_shells)
    shell_idx_flat = jnp.asarray(shell_idx_np.reshape(-1))
    shell_idx_2d = jnp.asarray(shell_idx_np)

    # --- FFT ---
    F_proj = jnp.fft.fftshift(jnp.fft.fft2(proj_arr, axes=(1, 2)), axes=(1, 2))
    F_exp = jnp.fft.fftshift(jnp.fft.fft2(exp_arr, axes=(1, 2)), axes=(1, 2))

    # --- 1. FRC ---
    cross = jnp.real(F_proj * jnp.conj(F_exp))
    p1 = jnp.abs(F_proj) ** 2
    p2 = jnp.abs(F_exp) ** 2

    num = _batched_radial_sum(cross, shell_idx_flat, n_shells, B)
    den1 = _batched_radial_sum(p1, shell_idx_flat, n_shells, B)
    den2 = _batched_radial_sum(p2, shell_idx_flat, n_shells, B)

    denom = jnp.sqrt(den1 * den2)
    frc_curve = jnp.where(denom > eps, num / jnp.where(denom > eps, denom, 1.0), 0.0)

    # --- 2. SSNR / Wiener weight (clean projection vs. noisy experimental case) ---
    frc_clipped = jnp.clip(frc_curve, -1 + 1e-6, 1 - 1e-6)
    ssnr_curve = (frc_clipped ** 2) / (1 - frc_clipped ** 2)
    wiener_weight_curve = frc_clipped ** 2  # = SSNR / (1+SSNR)

    # --- 3. Whitening ---
    if apply_whitening:
        counts_single = jnp.bincount(shell_idx_flat, length=n_shells).astype(jnp.float32)

        if use_mask:
            duty_cycle = jnp.mean(noise_mask_arr.reshape(B, -1), axis=1)
            duty_cycle = jnp.maximum(duty_cycle, 1e-8)
            masked = exp_arr * noise_mask_arr
            F_noise = jnp.fft.fftshift(jnp.fft.fft2(masked, axes=(1, 2)), axes=(1, 2))
            power = jnp.abs(F_noise) ** 2
            sums = _batched_radial_sum(power, shell_idx_flat, n_shells, B)
            counts = jnp.broadcast_to(counts_single, (B, n_shells))
            mean_power = jnp.where(counts > 0, sums / jnp.where(counts > 0, counts, 1.0), 0.0)
            noise_power_radial = mean_power / duty_cycle[:, None]
        else:
            sums = _batched_radial_sum(p2, shell_idx_flat, n_shells, B)  # reuse p2, already computed for FRC
            counts = jnp.broadcast_to(counts_single, (B, n_shells))
            mean_power = jnp.where(counts > 0, sums / jnp.where(counts > 0, counts, 1.0), 0.0)
            cutoff = min(int((1 - high_freq_fraction) * n_shells), n_shells - 1)
            plateau_level = jnp.mean(mean_power[:, cutoff:], axis=1)
            noise_power_radial = jnp.broadcast_to(plateau_level[:, None], (B, n_shells))

        whitening_curve = 1.0 / jnp.sqrt(noise_power_radial + eps)
        whitening_curve = whitening_curve / (jnp.mean(whitening_curve, axis=1, keepdims=True) + eps)
    else:
        whitening_curve = jnp.ones((B, n_shells))

    # --- combine, broadcast onto the 2D map ---
    if apply_wiener and apply_whitening:
        combined_curve = wiener_weight_curve * whitening_curve
    elif apply_wiener:
        combined_curve = wiener_weight_curve
    elif apply_whitening:
        combined_curve = whitening_curve
    else:
        combined_curve = jnp.ones((B, n_shells))

    weight_map = combined_curve[:, shell_idx_2d]   # (B, H, W)

    # --- weighted residual ---
    F_residual = weight_map * (F_exp - F_proj)
    residual_map = jnp.fft.ifft2(jnp.fft.ifftshift(F_residual, axes=(1, 2)), axes=(1, 2)).real

    residual_energy = jnp.sum(jnp.abs(F_residual) ** 2, axis=(1, 2)) / (H * W)

    return (
        shell_centers_np,   # numpy, static — not a "real" traced output
        frc_curve,
        ssnr_curve,
        wiener_weight_curve,
        whitening_curve,
        weight_map,
        F_residual,
        residual_map,
        residual_energy,
    )


# ---------------------------------------------------------------------------
# Public wrapper (not jitted: handles shape/None at the Python level, then
# delegates to the jitted core — same interface as the numpy version)
# ---------------------------------------------------------------------------
def compute_fourier_residual(
    projection,
    experimental,
    n_shells: int,
    apply_wiener: bool = True,
    apply_whitening: bool = True,
    noise_mask=None,
    eps: float = 1e-8,
):
    """
    Jitted drop-in replacement for `compute_fourier_residual` (numpy batched
    version). `n_shells` is now a required parameter (compute it once in
    `main()` and pass it explicitly, as you already do in your code).

    Always computes the full result (frc_curve, ssnr_curve,
    wiener_weight_curve, whitening_curve, weight_map, F_residual,
    residual_map, residual_energy) — take whichever fields you need from
    `result`.

    Note: `frc_threshold` / `pixel_size` / `resolution_at_frc_threshold` are
    no longer part of this function — use `compute_resolution_diagnostic`
    below, separately, only when you need that number (e.g. in the debug
    block), so you don't force a GPU->CPU sync on every step.
    """
    proj_arr, orig_ndim = _normalize_input(projection)
    exp_arr, _ = _normalize_input(experimental)
    if proj_arr.shape != exp_arr.shape:
        raise ValueError(f"projection and experimental must have the same shape: {proj_arr.shape} vs {exp_arr.shape}")

    B, H, W = proj_arr.shape

    use_mask = noise_mask is not None
    mask_arr = _normalize_mask(noise_mask, B, H, W)
    if mask_arr is None:
        # dummy: never read when use_mask=False, but a fixed-shape argument is required
        mask_arr = jnp.zeros((B, H, W), dtype=jnp.float32)

    (
        shell_centers_np,
        frc_curve,
        ssnr_curve,
        wiener_weight_curve,
        whitening_curve,
        weight_map,
        F_residual,
        residual_map,
        residual_energy,
    ) = _compute_fourier_residual_core(
        proj_arr, exp_arr, mask_arr,
        n_shells=n_shells,
        apply_wiener=apply_wiener,
        apply_whitening=apply_whitening,
        use_mask=use_mask,
        eps=eps,
    )

    return FourierResidualResult(
        freqs=shell_centers_np,
        frc_curve=frc_curve,
        ssnr_curve=ssnr_curve,
        wiener_weight_curve=wiener_weight_curve,
        whitening_curve=whitening_curve,
        weight_map=_restore_shape(weight_map, orig_ndim),
        F_residual=F_residual,
        residual_map=_restore_shape(residual_map, orig_ndim),
        residual_energy=residual_energy,
    )


# ---------------------------------------------------------------------------
# Optional diagnostic (NOT jitted, NOT meant to be called on every training
# step — it does dynamic branching on concrete values, so it must be used
# outside jit, typically only in the one-shot debug block / validation
# display)
# ---------------------------------------------------------------------------
def compute_resolution_diagnostic(frc_curve, freqs, frc_threshold=0.143, pixel_size=1.0):
    """
    frc_curve: (B, n_shells) — pass an array already moved to host
               (np.array(...)) or a jnp array (will be converted here,
               incurring a sync).
    freqs: (n_shells,) numpy, as returned in result.freqs.
    Returns: np.ndarray (B,) of resolutions (same units as pixel_size), NaN
             where no crossing was found.
    """
    frc_curve = np.asarray(frc_curve)
    B, n_shells = frc_curve.shape
    resolution = np.full(B, np.nan)

    for b in range(B):
        below = np.where(frc_curve[b] < frc_threshold)[0]
        if below.size > 0 and below[0] > 0:
            i = below[0]
            f0, f1 = freqs[i - 1], freqs[i]
            y0, y1 = frc_curve[b, i - 1], frc_curve[b, i]
            if y1 != y0:
                frac = (frc_threshold - y0) / (y1 - y0)
                k_cross = f0 + frac * (f1 - f0)
            else:
                k_cross = f0
            if k_cross > 0:
                resolution[b] = pixel_size / k_cross

    return resolution