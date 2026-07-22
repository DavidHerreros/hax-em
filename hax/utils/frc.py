"""
fourier_residual_batched.py

Batched version: computes a noise-robust Fourier-domain residual between a
batch of (noise-free, CTF-applied) theoretical projections and a batch of
noisy experimental cryo-EM images.

Accepted input shapes (auto-detected):
    (B, H, W, 1)  <-- primary target shape (batched, single grayscale channel)
    (B, H, W)
    (H, W)        <-- single image, treated as B=1 internally

All outputs are reshaped back to match whatever shape was given for
`projection` (so if you pass (B,H,W,1) you get (B,H,W,1) maps back; if you
pass a single (H,W) image you get plain (H,W) maps back).

See the single-image version's docstring / prior discussion for the
mathematical background (FRC, the clean-vs-noisy SSNR/Wiener-weight
derivation w(k) = FRC(k)^2, and whitening by the noise power spectrum).
Everything here is the same math, vectorized over an extra batch axis using
a single `np.bincount` call per computation instead of a Python loop over B.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FourierResidualResult:
    freqs: np.ndarray                 # (n_shells,) - shared across batch
    frc_curve: np.ndarray              # (B, n_shells)
    ssnr_curve: np.ndarray              # (B, n_shells)
    wiener_weight_curve: np.ndarray     # (B, n_shells)  = frc_curve**2
    whitening_curve: np.ndarray         # (B, n_shells)
    weight_map: np.ndarray              # matches input shape (e.g. (B,H,W,1))
    F_residual: np.ndarray              # (B, H, W) complex, fftshifted
    residual_map: np.ndarray            # matches input shape
    residual_energy: np.ndarray         # (B,)
    resolution_at_frc_threshold: np.ndarray  # (B,)


# ---------------------------------------------------------------------------
# Shape handling helpers
# ---------------------------------------------------------------------------
def _normalize_input(x):
    """
    Accepts (B,H,W,1), (B,H,W), or (H,W). Returns:
        arr        : float64 ndarray of shape (B, H, W)
        orig_ndim  : the ndim of the original input (2, 3, or 4)
    """
    x = np.asarray(x)
    if x.ndim == 4:
        if x.shape[-1] != 1:
            raise ValueError(
                f"Expected a single grayscale channel (last dim == 1), got shape {x.shape}."
            )
        return x[..., 0].astype(np.float64), 4
    elif x.ndim == 3:
        return x.astype(np.float64), 3
    elif x.ndim == 2:
        return x[None, ...].astype(np.float64), 2
    else:
        raise ValueError(f"Unsupported input ndim={x.ndim}, expected 2, 3, or 4.")


def _restore_shape(arr, orig_ndim):
    """Reshape a (B,H,W) array back to whatever shape the input originally had."""
    if orig_ndim == 4:
        return arr[..., None]
    elif orig_ndim == 2:
        return arr[0]
    else:
        return arr  # already (B,H,W)


def _normalize_mask(mask, B, H, W):
    """
    Accepts a noise_mask as (H,W), (B,H,W), or (B,H,W,1) (bool-like).
    Returns a float array of shape (B,H,W) broadcast across the batch if needed.
    """
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 4:
        mask = mask[..., 0]
    if mask.ndim == 2:
        mask = np.broadcast_to(mask, (B, H, W))
    elif mask.ndim == 3:
        if mask.shape != (B, H, W):
            raise ValueError(f"noise_mask shape {mask.shape} incompatible with data shape {(B, H, W)}.")
    else:
        raise ValueError(f"Unsupported noise_mask ndim={mask.ndim}.")
    return mask.astype(float)


# ---------------------------------------------------------------------------
# Radial shell geometry (shared across the batch: same H,W -> same shell map)
# ---------------------------------------------------------------------------
def _radial_shell_indices(shape, n_shells=None):
    H, W = shape
    fy = np.fft.fftshift(np.fft.fftfreq(H))
    fx = np.fft.fftshift(np.fft.fftfreq(W))
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(kx ** 2 + ky ** 2)

    if n_shells is None:
        n_shells = min(H, W) // 2

    max_r = radius.max()
    shell_idx = np.floor(radius / max_r * n_shells).astype(int)
    shell_idx = np.clip(shell_idx, 0, n_shells - 1)
    shell_centers = (np.arange(n_shells) + 0.5) / n_shells * max_r
    return shell_idx, shell_centers, n_shells


# ---------------------------------------------------------------------------
# Batched radial reductions, vectorized with a single bincount call using a
# combined (batch_index * n_shells + shell_index) key instead of looping
# over the batch in Python.
# ---------------------------------------------------------------------------
def _batched_radial_sum(values, shell_idx, n_shells, B):
    """
    values: (B, H, W) real array to sum per (batch, shell).
    Returns: (B, n_shells) array of per-shell sums.
    """
    HW = shell_idx.size
    flat_shell = shell_idx.ravel()
    combined_idx = (np.arange(B)[:, None] * n_shells + flat_shell[None, :]).ravel()
    weights = values.reshape(B, HW).ravel()
    sums = np.bincount(combined_idx, weights=weights, minlength=B * n_shells)
    return sums.reshape(B, n_shells)


def compute_frc_batch(F1, F2, shell_idx, n_shells, eps=1e-12):
    """
    F1, F2: (B, H, W) complex, fftshifted.
    Returns frc (B, n_shells), den1 (B, n_shells), den2 (B, n_shells).
    """
    B = F1.shape[0]
    cross = np.real(F1 * np.conj(F2))
    p1 = np.abs(F1) ** 2
    p2 = np.abs(F2) ** 2

    num = _batched_radial_sum(cross, shell_idx, n_shells, B)
    den1 = _batched_radial_sum(p1, shell_idx, n_shells, B)
    den2 = _batched_radial_sum(p2, shell_idx, n_shells, B)

    denom = np.sqrt(den1 * den2)
    frc = np.divide(num, denom, out=np.zeros_like(num), where=denom > eps)
    return frc, den1, den2


def estimate_noise_power_radial_batch(
    experimental, shell_idx, n_shells, noise_mask=None, high_freq_fraction=0.2
):
    """
    experimental: (B, H, W) real.
    noise_mask: (B, H, W) float in {0,1} (already normalized/broadcast), or None.
    Returns noise_power_radial: (B, n_shells).
    """
    B, H, W = experimental.shape
    counts_single = np.bincount(shell_idx.ravel(), minlength=n_shells).astype(float)  # (n_shells,) same for all batch items

    if noise_mask is not None:
        duty_cycle = noise_mask.reshape(B, -1).mean(axis=1)  # (B,)
        duty_cycle = np.maximum(duty_cycle, 1e-8)
        masked = experimental * noise_mask
        F_noise = np.fft.fftshift(np.fft.fft2(masked, axes=(1, 2)), axes=(1, 2))
        power = np.abs(F_noise) ** 2
        sums = _batched_radial_sum(power, shell_idx, n_shells, B)
        counts = np.broadcast_to(counts_single, (B, n_shells))
        mean_power = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        noise_power_radial = mean_power / duty_cycle[:, None]
    else:
        F_exp = np.fft.fftshift(np.fft.fft2(experimental, axes=(1, 2)), axes=(1, 2))
        power = np.abs(F_exp) ** 2
        sums = _batched_radial_sum(power, shell_idx, n_shells, B)
        counts = np.broadcast_to(counts_single, (B, n_shells))
        mean_power = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        cutoff = min(int((1 - high_freq_fraction) * n_shells), n_shells - 1)
        plateau_level = np.mean(mean_power[:, cutoff:], axis=1)  # (B,)
        noise_power_radial = np.broadcast_to(plateau_level[:, None], (B, n_shells)).copy()

    return noise_power_radial


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_fourier_residual(
    projection,
    experimental,
    n_shells=None,
    apply_wiener=True,
    apply_whitening=True,
    noise_mask=None,
    frc_threshold=0.143,
    pixel_size=1.0,
    eps=1e-8,
):
    """
    Batched, noise-robust Fourier-domain residual between theoretical
    projections (CTF already applied, noise-free) and noisy experimental
    images.

    Parameters
    ----------
    projection, experimental : ndarray
        Accepts shape (B,H,W,1), (B,H,W), or (H,W). Must match each other's
        shape.
    n_shells : int, optional
        Number of radial Fourier shells (shared across the batch, since H,W
        are shared). Defaults to min(H, W) // 2.
    apply_wiener : bool
        Weight residual by Wiener weight w(k) = FRC(k)^2 (valid for the
        clean-projection-vs-noisy-experimental case; see prior derivation).
    apply_whitening : bool
        Flatten residual by the (per-sample) estimated noise power spectrum.
    noise_mask : ndarray, optional
        Background/noise-only mask. Accepts (H,W) [shared across batch],
        (B,H,W), or (B,H,W,1). True/1 = background pixel.
    frc_threshold : float
        FRC cutoff used only for the rough resolution diagnostic.
    pixel_size : float
        Angstrom/pixel, only used to scale the resolution diagnostic.
    eps : float
        Numerical stability constant.

    Returns
    -------
    FourierResidualResult
        All per-image outputs carry a leading batch dimension B; map-like
        outputs (`weight_map`, `residual_map`) are reshaped to match
        whatever shape `projection` was given in (e.g. (B,H,W,1)).
    """
    proj_arr, orig_ndim = _normalize_input(projection)
    exp_arr, _ = _normalize_input(experimental)
    if proj_arr.shape != exp_arr.shape:
        raise ValueError(
            f"projection and experimental must match in shape; got "
            f"{proj_arr.shape} vs {exp_arr.shape} after normalization."
        )

    B, H, W = proj_arr.shape
    shell_idx, shell_centers, n_shells = _radial_shell_indices((H, W), n_shells)

    F_proj = np.fft.fftshift(np.fft.fft2(proj_arr, axes=(1, 2)), axes=(1, 2))
    F_exp = np.fft.fftshift(np.fft.fft2(exp_arr, axes=(1, 2)), axes=(1, 2))

    # --- 1. FRC, per batch item ---
    frc_curve, _, _ = compute_frc_batch(F_proj, F_exp, shell_idx, n_shells, eps=eps)

    # --- 2. SSNR / Wiener weight (clean-vs-noisy formula), per batch item ---
    frc_clipped = np.clip(frc_curve, -1 + 1e-6, 1 - 1e-6)
    ssnr_curve = (frc_clipped ** 2) / (1 - frc_clipped ** 2)
    wiener_weight_curve = frc_clipped ** 2  # = SSNR / (1+SSNR)

    # --- 3. Whitening, per batch item ---
    if apply_whitening:
        mask_norm = _normalize_mask(noise_mask, B, H, W)
        noise_power_radial = estimate_noise_power_radial_batch(
            exp_arr, shell_idx, n_shells, noise_mask=mask_norm
        )
        whitening_curve = 1.0 / np.sqrt(noise_power_radial + eps)
        whitening_curve = whitening_curve / (whitening_curve.mean(axis=1, keepdims=True) + eps)
    else:
        whitening_curve = np.ones((B, n_shells))

    # --- combine curves, then broadcast each batch item's curve onto its 2D shell map ---
    if apply_wiener and apply_whitening:
        combined_curve = wiener_weight_curve * whitening_curve
    elif apply_wiener:
        combined_curve = wiener_weight_curve
    elif apply_whitening:
        combined_curve = whitening_curve
    else:
        combined_curve = np.ones((B, n_shells))

    # combined_curve: (B, n_shells), shell_idx: (H, W) -> weight_map: (B, H, W)
    weight_map = combined_curve[:, shell_idx]

    # --- weighted complex residual, and its real-space form ---
    F_residual = weight_map * (F_exp - F_proj)
    residual_map = np.fft.ifft2(np.fft.ifftshift(F_residual, axes=(1, 2)), axes=(1, 2)).real

    residual_energy = np.sum(np.abs(F_residual) ** 2, axis=(1, 2)) / (H * W)  # (B,)

    # --- resolution diagnostic per batch item ---
    resolution = np.full(B, np.nan)
    for b in range(B):
        below = np.where(frc_curve[b] < frc_threshold)[0]
        if below.size > 0 and below[0] > 0:
            i = below[0]
            f0, f1 = shell_centers[i - 1], shell_centers[i]
            y0, y1 = frc_curve[b, i - 1], frc_curve[b, i]
            if y1 != y0:
                frac = (frc_threshold - y0) / (y1 - y0)
                k_cross = f0 + frac * (f1 - f0)
            else:
                k_cross = f0
            if k_cross > 0:
                resolution[b] = pixel_size / k_cross

    return FourierResidualResult(
        freqs=shell_centers,
        frc_curve=frc_curve,
        ssnr_curve=ssnr_curve,
        wiener_weight_curve=wiener_weight_curve,
        whitening_curve=whitening_curve,
        weight_map=_restore_shape(weight_map, orig_ndim),
        F_residual=F_residual,  # kept as (B,H,W) complex regardless of orig_ndim
        residual_map=_restore_shape(residual_map, orig_ndim),
        residual_energy=residual_energy,
        resolution_at_frc_threshold=resolution,
    )


# ---------------------------------------------------------------------------
# Demo / sanity check with synthetic batched data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    N = 128
    B = 4  # batch size

    y, x = np.mgrid[0:N, 0:N] - N / 2
    r = np.sqrt(x ** 2 + y ** 2)
    clean = np.exp(-(r ** 2) / (2 * 18 ** 2))
    clean += 0.6 * np.exp(-((r - 40) ** 2) / (2 * 6 ** 2))

    Fc = np.fft.fftshift(np.fft.fft2(clean))
    fy = np.fft.fftshift(np.fft.fftfreq(N))
    fx = np.fft.fftshift(np.fft.fftfreq(N))
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    k = np.sqrt(kx ** 2 + ky ** 2)
    envelope = np.exp(-(k ** 2) / (2 * 0.15 ** 2))
    ctf_like = np.cos(2 * np.pi * 300 * (k ** 2)) * envelope
    clean_ctf = np.fft.ifft2(np.fft.ifftshift(Fc * ctf_like)).real

    # Build a batch: same base projection for all B, but each experimental
    # image gets a different noise level and a different "extra density"
    # blob location/size, to demonstrate per-sample outputs.
    projections = np.stack([clean_ctf for _ in range(B)], axis=0)  # (B,H,W)
    experimentals = np.zeros_like(projections)
    noise_scales = [1.0, 2.0, 4.0, 6.0]  # increasing noise across the batch
    blob_offsets = [(0, 0), (20, -10), (-25, 15), (10, 30)]

    for b in range(B):
        dx, dy = blob_offsets[b]
        extra = 0.5 * np.exp(-(((x - dx) ** 2) + ((y - dy) ** 2)) / (2 * 5 ** 2))
        noise = rng.normal(scale=clean_ctf.std() * noise_scales[b], size=clean_ctf.shape)
        experimentals[b] = clean_ctf + extra + noise

    # Reshape to the target (B,H,W,1) format
    projection_batch = projections[..., None]
    experimental_batch = experimentals[..., None]

    result = compute_fourier_residual(
        projection_batch,
        experimental_batch,
        apply_wiener=True,
        apply_whitening=True,
        noise_mask=None,
        pixel_size=1.0,
    )

    print("Input shape:", projection_batch.shape)
    print("residual_map shape:", result.residual_map.shape)
    print("weight_map shape:", result.weight_map.shape)
    print("Per-sample residual energy:", np.round(result.residual_energy, 3))
    print("Per-sample resolution @ FRC=0.143:", np.round(result.resolution_at_frc_threshold, 2))

    fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
    for b in range(B):
        axes[b, 0].imshow(experimental_batch[b, ..., 0], cmap="gray")
        axes[b, 0].set_title(f"Sample {b}: experimental (noise x{noise_scales[b]})")
        axes[b, 0].axis("off")

        raw_diff = experimental_batch[b, ..., 0] - projection_batch[b, ..., 0]
        axes[b, 1].imshow(raw_diff, cmap="RdBu_r", vmin=-raw_diff.std() * 3, vmax=raw_diff.std() * 3)
        axes[b, 1].set_title("Naive real-space diff")
        axes[b, 1].axis("off")

        rmap = result.residual_map[b, ..., 0]
        axes[b, 2].imshow(rmap, cmap="RdBu_r", vmin=-rmap.std() * 3, vmax=rmap.std() * 3)
        axes[b, 2].set_title("Wiener + whitened residual")
        axes[b, 2].axis("off")

        axes[b, 3].plot(result.freqs, result.frc_curve[b], label="FRC(k)")
        axes[b, 3].axhline(0.143, color="gray", linestyle="--")
        axes[b, 3].set_ylim(-0.2, 1.05)
        axes[b, 3].set_title("FRC curve")
        axes[b, 3].legend()

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/fourier_residual_batched_demo.png", dpi=140)
    print("Saved demo figure to /mnt/user-data/outputs/fourier_residual_batched_demo.png")