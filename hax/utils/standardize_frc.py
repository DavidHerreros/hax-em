"""

Resample FRC curves onto a common reference frequency grid, so curves
computed on images with different box sizes (and/or different pixel sizes)
end up with the same fixed length and refer to the same physical
frequencies — required if you want a single downstream MLP to accept curves
from datasets that don't all share the same box size.

Uses zero-order-hold (step) interpolation by default: robust to the FRC
curve's natural noisiness, and doesn't invent smoothness the raw data
doesn't have.
"""

import numpy as np
from scipy.interpolate import interp1d


def standardize_frc_curve(frc_curve, freqs_px, ts, n_shells_ref, ts_ref=1.0, method="zero"):
    """
    Args:
        frc_curve: (B, n_shells_native) — FRC curve(s), from result.frc_curve.
        freqs_px: (n_shells_native,) — native frequency axis, cycles/pixel,
            from result.freqs (this dataset's own box size).
        ts: this dataset's pixel size, Angstrom/pixel.
        n_shells_ref: target number of frequency bins for the reference grid
            (e.g. the largest box size // 2 across your datasets, or a fixed
            constant you standardize everything to).
        ts_ref: reference pixel size defining the target Nyquist frequency
            (Angstrom/pixel); default 1.0 A/pixel.
        method: "zero" (zero-order hold / step interpolation, default) or
            "linear". Implemented via scipy.interpolate.interp1d, vectorized
            over the batch (no Python loop).

    Returns:
        frc_curve_std: (B, n_shells_ref) — resampled curve(s).
        freqs_ref: (n_shells_ref,) — the common target frequency axis,
            cycles/Angstrom.

    Frequencies in freqs_ref beyond this dataset's native max physical
    frequency (freqs_px.max() / ts) are set to 0 — there's no information
    there for this dataset, so extrapolating/repeating the last value would
    be misleading.
    """
    frc_curve = np.asarray(frc_curve)
    freqs_px = np.asarray(freqs_px)

    # this dataset's native frequency axis, converted to physical units (cycles/Angstrom)
    freqs_physical = freqs_px / ts

    # common target grid, in physical units
    nyquist_ref = 1.0 / (2.0 * ts_ref)
    freqs_ref = (np.arange(n_shells_ref) + 0.5) / n_shells_ref * nyquist_ref

    kind = "previous" if method == "zero" else method  # scipy calls zero-order-hold "previous"
    f = interp1d(
        freqs_physical, frc_curve, kind=kind, axis=1,
        bounds_error=False, fill_value=(frc_curve[:, 0], 0.0),
    )
    frc_curve_std = f(freqs_ref)

    return frc_curve_std, freqs_ref