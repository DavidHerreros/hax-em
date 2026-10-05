import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from xmipp_metadata.image_handler import ImageHandler
from hax.utils import *


def _rfft_freq_grids(grid_size):
    """Broadcastable (fz, fy, fx) in cycles/voxel, matching jnp.fft.rfftn axis order."""
    fz = jnp.fft.fftfreq(grid_size)[:, None, None]
    fy = jnp.fft.fftfreq(grid_size)[None, :, None]
    fx = jnp.fft.rfftfreq(grid_size)[None, None, :]
    return fz, fy, fx


def _shell_index(grid_size):
    """Integer Fourier-shell index of every rfft voxel, plus the number of shells."""
    fz, fy, fx = _rfft_freq_grids(grid_size)
    radius = jnp.sqrt(fz ** 2 + fy ** 2 + fx ** 2) * grid_size
    n_shells = grid_size // 2 + 1
    shell_idx = jnp.where(radius <= grid_size / 2.0,
                          jnp.round(radius),
                          n_shells).astype(jnp.int32)
    return shell_idx, n_shells


def hermitian_multiplicity(grid_size):
    """rfftn keeps half of Fourier space. Every voxel outside the kx = 0 and
    kx = Nyquist planes stands for itself *and* its Friedel mate, so it must count
    twice in any sum meant to equal a full-space sum (e.g. a Parseval-consistent loss)."""
    mult = 2.0 * jnp.ones((1, 1, grid_size // 2 + 1), dtype=jnp.float32)
    mult = mult.at[0, 0, 0].set(1.0)
    if grid_size % 2 == 0:
        mult = mult.at[0, 0, -1].set(1.0)
    return jnp.broadcast_to(mult, (grid_size, grid_size, grid_size // 2 + 1))


def _fibonacci_half_sphere(n_dirs):
    """Near-uniform directions on the upper hemisphere. Half is enough: the transform
    of a real map is Friedel-symmetric, so k and -k carry the same FSC."""
    i = np.arange(n_dirs, dtype=np.float64) + 0.5
    z = i / n_dirs
    r = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    dirs = np.stack([z, r * np.cos(phi), r * np.sin(phi)], axis=1)  # (z, y, x) order
    return jnp.asarray(dirs, dtype=jnp.float32)


def _sinc(f):
    """sin(pi f) / (pi f), safe at f = 0."""
    near_zero = jnp.abs(f) < 1e-8
    pf = jnp.pi * jnp.where(near_zero, 1.0, f)
    return jnp.where(near_zero, 1.0, jnp.sin(pf) / pf)


@partial(jax.jit, static_argnames=("n_shells",))
def _shell_fsc(ft1, ft2, shell_idx, mult, n_shells):
    """Isotropic (spherically averaged) FSC curve."""
    flat_shell = shell_idx.ravel()
    seg = lambda a: jax.ops.segment_sum(a.ravel(), flat_shell, num_segments=n_shells + 1)[:n_shells]
    num = seg(mult * (ft1.real * ft2.real + ft1.imag * ft2.imag))
    p1 = seg(mult * (ft1.real ** 2 + ft1.imag ** 2))
    p2 = seg(mult * (ft2.real ** 2 + ft2.imag ** 2))
    return num / jnp.sqrt(p1 * p2 + 1e-20)


def _cone_weight(cos_ang, cone_cos, cone_sigma, hard_cones):
    """Soft (Gaussian-in-angle) cone hard-truncated at the nominal half-angle, or, with
    hard_cones, AR-Decon's plain cone: every voxel inside the half-angle counts fully."""
    inside = cos_ang >= cone_cos
    if hard_cones:
        return inside.astype(jnp.float32)
    return jnp.exp(-0.5 * (jnp.arccos(cos_ang) / cone_sigma) ** 2) * inside


@partial(jax.jit, static_argnames=("n_shells", "grid_size", "hard_cones"))
def _directional_shell_fsc(ft1, ft2, dirs, shell_idx, mult, n_shells, grid_size,
                           cone_cos, cone_sigma, hard_cones=False):
    """One FSC curve per cone direction -> (n_dirs, n_shells).

    Scanned rather than vmapped: a (n_dirs, N, N, N//2+1) intermediate would not fit
    for any realistic box size."""
    fz, fy, fx = _rfft_freq_grids(grid_size)
    radius = jnp.sqrt(fz ** 2 + fy ** 2 + fx ** 2)
    safe_r = jnp.where(radius > 0, radius, 1.0)

    num = mult * (ft1.real * ft2.real + ft1.imag * ft2.imag)
    p1 = mult * (ft1.real ** 2 + ft1.imag ** 2)
    p2 = mult * (ft2.real ** 2 + ft2.imag ** 2)
    flat_shell = shell_idx.ravel()

    def body(carry, d):
        cos_ang = jnp.clip(jnp.abs(fz * d[0] + fy * d[1] + fx * d[2]) / safe_r, 0.0, 1.0)
        cw = _cone_weight(cos_ang, cone_cos, cone_sigma, hard_cones)
        cw = jnp.where(radius > 0, cw, 1.0)  # DC has no direction; it belongs to every cone
        seg = lambda a: jax.ops.segment_sum((cw * a).ravel(), flat_shell, num_segments=n_shells + 1)[:n_shells]

        num_sum = seg(num)
        w_sum = seg(cw * mult)

        fsc = num_sum / jnp.sqrt(seg(p1) * seg(p2) + 1e-20)
        fsc = jnp.where(w_sum < 0.1, 1.0, fsc)

        return carry, fsc

    _, curves = jax.lax.scan(body, None, dirs)
    return curves


@partial(jax.jit, static_argnames=("grid_size", "hard_cones"))
def _scatter_directional_curves(curves, dirs, shell_idx, grid_size, cone_cos, cone_sigma, hard_cones=False):
    """Cone-weighted angular interpolation of the per-direction curves back onto every
    rfft voxel. Same cone weights as the analysis pass, so the result inherits the
    same angular smoothing instead of showing cone boundaries."""
    fz, fy, fx = _rfft_freq_grids(grid_size)
    radius = jnp.sqrt(fz ** 2 + fy ** 2 + fx ** 2)
    safe_r = jnp.where(radius > 0, radius, 1.0)
    shape = (grid_size, grid_size, grid_size // 2 + 1)

    def body(carry, xs):
        acc, wsum = carry
        d, curve = xs
        cos_ang = jnp.clip(jnp.abs(fz * d[0] + fy * d[1] + fx * d[2]) / safe_r, 0.0, 1.0)
        cw = _cone_weight(cos_ang, cone_cos, cone_sigma, hard_cones)
        cw = jnp.where(radius > 0, cw, 1.0)

        padded_curve = jnp.append(curve, 0.0)
        shell_val = jnp.take(padded_curve, shell_idx)

        return (acc + cw * shell_val, wsum + cw), None

    init = (jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32))
    (acc, wsum), _ = jax.lax.scan(body, init, (dirs, curves))
    return acc / jnp.maximum(wsum, 1e-8)


def _phase_randomized_map(vol, shell_idx, start_shell, key):
    """Replace phases beyond `start_shell` with random ones, keeping the amplitudes.
    Correlation surviving this is mask correlation, not signal."""
    ft = jnp.fft.rfftn(vol)

    noise = jax.random.normal(key, vol.shape)
    noise_ft = jnp.fft.rfftn(noise)
    phases = jnp.angle(noise_ft)
    randomized = jnp.abs(ft) * jnp.exp(1j * phases)

    ft = jnp.where(shell_idx >= start_shell, randomized, ft)
    return jnp.fft.irfftn(ft, s=vol.shape)


def fsc_resolution(curve, sr, grid_size, threshold=0.143):
    """Resolution (A) at which an FSC curve first crosses `threshold`, linearly
    interpolated between shells. Returns np.inf if it never crosses."""
    curve = np.asarray(curve)
    below = np.nonzero(curve < threshold)[0]
    below = below[below > 0]
    if below.size == 0:
        return float(grid_size * sr / (len(curve) - 1))

    j = int(below[0])
    prev, cur = curve[j - 1], curve[j]
    frac = 0.0 if prev == cur else (prev - threshold) / (prev - cur)
    shell = (j - 1) + float(np.clip(frac, 0.0, 1.0))

    if shell <= 0.0:
        return np.inf
    return float(grid_size * sr / shell)


def compute_fsc_weight(half1, half2, mask=None, mode="fsc_ref", directional=True,
                       n_cones=100, cone_angle=20.0, phase_randomize=True,
                       randomize_from_fsc=0.8, seed=0, sr=1.0, hard_cones=False, verbose=True):
    """Per-Fourier-voxel confidence W(k_vec) in [0, 1] built from two half-maps.

    Args:
        half1, half2: (N, N, N) unmasked half-maps (arrays, or paths readable by
            ImageHandler). They must come from independently refined halves; halves
            that share a reference give an inflated FSC and W will over-sharpen.
        mask: optional soft/binary mask. Masking inflates the FSC through mask
            correlation, which is what `phase_randomize` corrects for.
        mode: "fsc_ref" -> sqrt(2 FSC / (1 + FSC)), the correlation of the *full* map
            with ground truth and the right multiplier for amplitude restoration;
            "fsc" -> the raw half-map FSC; "ones" -> disable weighting (ablation).
        directional: True computes a cone-resolved (3D) FSC and so also corrects
            anisotropic resolution; False gives a spherically averaged weight.
        n_cones / cone_angle: number of hemisphere directions and cone half-angle in
            degrees. Narrow cones are noisy; 20 deg with ~100 cones keeps the cones
            overlapping, which is what smooths the result.
        hard_cones: use AR-Decon's plain cones (every voxel inside the half-angle weighs 1)
            instead of the Gaussian-tapered ones. To reproduce AR-Decon's dFSC3d pass the
            raw mask, phase_randomize=False, cone_angle=20 and half its cone count (AR-Decon
            spreads its cones over the whole sphere, these cover a hemisphere).

    Returns:
        (weight, info) where weight is (N, N, N//2+1) float32 aligned with rfftn, and
        info holds the spherical FSC curves and the resolution estimate.
    """
    if isinstance(half1, str):
        half1 = ImageHandler(half1).getData()
    if isinstance(half2, str):
        half2 = ImageHandler(half2).getData()
    if isinstance(mask, str):
        mask = ImageHandler(mask).getData()

    half1 = jnp.asarray(np.squeeze(np.asarray(half1)), dtype=jnp.float32)
    half2 = jnp.asarray(np.squeeze(np.asarray(half2)), dtype=jnp.float32)
    if half1.shape != half2.shape:
        raise ValueError(f"Half-map shapes differ: {half1.shape} vs {half2.shape}")
    grid_size = half1.shape[0]

    shell_idx, n_shells = _shell_index(grid_size)
    mult = hermitian_multiplicity(grid_size)

    if mode == "ones":
        return jnp.ones((grid_size, grid_size, grid_size // 2 + 1), jnp.float32), {"resolution": np.inf}

    # Unmasked FSC: needed on its own to pick the phase-randomization shell.
    ft1_raw, ft2_raw = jnp.fft.rfftn(half1), jnp.fft.rfftn(half2)
    fsc_unmasked = _shell_fsc(ft1_raw, ft2_raw, shell_idx, mult, n_shells)

    if mask is not None:
        mask_j = jnp.asarray(np.squeeze(np.asarray(mask)), dtype=jnp.float32)
        m1, m2 = half1 * mask_j, half2 * mask_j
    else:
        m1, m2 = half1, half2

    ft1, ft2 = jnp.fft.rfftn(m1), jnp.fft.rfftn(m2)
    fsc_masked = _shell_fsc(ft1, ft2, shell_idx, mult, n_shells)

    # --- mask-correlation correction (Chen et al. / high-resolution noise substitution)
    cone_cos = float(np.cos(np.deg2rad(cone_angle)))
    cone_sigma = float(np.deg2rad(cone_angle) / 2.0)
    dirs = _fibonacci_half_sphere(n_cones) if directional else _fibonacci_half_sphere(1)

    start_shell = n_shells
    if mask is not None and phase_randomize:
        below = np.nonzero(np.asarray(fsc_unmasked) < randomize_from_fsc)[0]
        below = below[below > 0]
        if below.size > 0:
            start_shell = int(below[0])
            key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
            r1 = _phase_randomized_map(half1, shell_idx, start_shell, key1)
            r2 = _phase_randomized_map(half2, shell_idx, start_shell, key2)

            if directional:
                fsc_rand = _directional_shell_fsc(jnp.fft.rfftn(r1 * mask_j), jnp.fft.rfftn(r2 * mask_j),
                                                  dirs, shell_idx, mult, n_shells, grid_size, cone_cos, cone_sigma,
                                                  hard_cones)
            else:
                fsc_rand = _shell_fsc(jnp.fft.rfftn(r1 * mask_j), jnp.fft.rfftn(r2 * mask_j),
                                      shell_idx, mult, n_shells)
        else:
            fsc_rand = jnp.zeros((n_cones, n_shells) if directional else n_shells, jnp.float32)
    else:
        fsc_rand = jnp.zeros((n_cones, n_shells) if directional else n_shells, jnp.float32)

    shells = jnp.arange(n_shells)
    correct_from = start_shell + 2  # small offset: the randomization edge itself is unreliable

    def _correct(curve, rand_curve):
        corrected = (curve - rand_curve) / jnp.maximum(1.0 - rand_curve, 1e-3)
        corrected = jnp.where(shells >= correct_from, corrected, curve)
        return corrected, jnp.clip(corrected, 0.0, 0.999)

    # --- Final FSC ---
    if directional:
        raw_curves = _directional_shell_fsc(ft1, ft2, dirs, shell_idx, mult, n_shells,
                                            grid_size, cone_cos, cone_sigma, hard_cones)

        fsc_corrected_dir, curves_clipped = jax.vmap(_correct)(raw_curves, fsc_rand)
        fsc_corrected_spherical, _ = _correct(fsc_masked,
                                              jnp.mean(fsc_rand, axis=0) if fsc_rand.ndim == 2 else fsc_rand)
    else:
        fsc_corrected_spherical, curves_clipped = _correct(fsc_masked, fsc_rand)
        curves_clipped = curves_clipped[None, :]

    # --- FSC -> restoration weight
    if mode == "fsc_ref":
        curves_w = jnp.sqrt(2.0 * curves_clipped / (1.0 + curves_clipped))
    elif mode == "fsc":
        curves_w = curves_clipped
    else:
        raise ValueError(f"Unknown FSC weighting mode: {mode}")

    curves_w = curves_w.at[:, 0].set(1.0)  # DC is always fully trusted

    if directional:
        weight = _scatter_directional_curves(curves_w, dirs, shell_idx, grid_size,
                                             cone_cos, cone_sigma, hard_cones)
    else:
        weight = curves_w[0][shell_idx]
    weight = jnp.clip(weight, 0.0, 1.0)

    res = fsc_resolution(fsc_corrected_spherical, sr, grid_size)

    info = {"fsc_unmasked": np.asarray(fsc_unmasked),
            "fsc_masked": np.asarray(fsc_masked),
            "fsc_random": np.asarray(fsc_rand),
            "fsc_corrected": np.asarray(fsc_corrected_spherical),
            "resolution": res,
            "randomization_shell": start_shell,
            "n_shells": n_shells}

    if directional:
        info["fsc_directional"] = np.asarray(fsc_corrected_dir)
        info["directions"] = np.asarray(dirs)

    if verbose:
        print(f"{bcolors.OKCYAN}###### 3D FSC confidence map ######{bcolors.ENDC}")
        print(f"  box {grid_size}^3 @ {sr:.3f} A/px | "
              f"{'directional (' + str(n_cones) + ' cones, ' + str(cone_angle) + ' deg)' if directional else 'spherical'}")
        if mask is not None and phase_randomize and start_shell < n_shells:
            print(f"  phase randomization from shell {start_shell} "
                  f"({grid_size * sr / max(start_shell, 1):.2f} A) to undo mask correlation")
        print(f"  FSC = 0.143 at {res:.2f} A | mean weight {float(jnp.mean(weight)):.3f}")

    return weight, info