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
    """Near-uniform directions on the upper hemisphere."""
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


@partial(jax.jit, static_argnames=("n_shells",))
def _shell_crosspower(ft1, ft2, shell_idx, mult, n_shells):
    """Isotropic cross-power Re(F1 * F2*) per shell."""
    flat_shell = shell_idx.ravel()
    seg = lambda a: jax.ops.segment_sum(a.ravel(), flat_shell, num_segments=n_shells + 1)[:n_shells]
    num = seg(mult * (ft1.real * ft2.real + ft1.imag * ft2.imag))
    w_sum = seg(mult)
    return num / jnp.maximum(w_sum, 1e-8)


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
    """One FSC curve and cross-power curve per cone direction -> (n_dirs, n_shells)."""
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
        cross_p = num_sum / jnp.maximum(w_sum, 1e-8)

        return carry, (fsc, cross_p)

    _, (curves, cross_powers) = jax.lax.scan(body, None, dirs)
    return curves, cross_powers


@partial(jax.jit, static_argnames=("n_shells", "grid_size"))
def _fit_guinier_envelope(cross_powers, fsc_curves, n_shells, grid_size, sr, res_thresh=0.143):
    """Fits directional B-factor from Re(F1*F2*) between 10 A and cone resolution."""
    shells = jnp.arange(n_shells)
    s = shells / (grid_size * sr)
    s_sq = s ** 2
    j_min = jnp.maximum(1, jnp.ceil(0.1 * grid_size * sr).astype(jnp.int32))

    def fit_cone(p_c, f_c):
        # Last shell before the first 0.143 crossing: later shells above it are noise
        first_below = jnp.min(jnp.where((f_c < res_thresh) & (shells > 0), shells, n_shells))
        cutoff_shell = jnp.maximum(first_below - 1, j_min)
        valid = (shells >= j_min) & (shells <= cutoff_shell) & (p_c > 1e-12)
        w = valid.astype(jnp.float32) * f_c  # Weight regression by FSC confidence
        w_sum = jnp.sum(w)
        safe_w = jnp.maximum(w_sum, 1e-8)

        y = jnp.log(jnp.maximum(p_c, 1e-12))
        x_m = jnp.sum(w * s_sq) / safe_w
        y_m = jnp.sum(w * y) / safe_w

        var_x = jnp.sum(w * (s_sq - x_m) ** 2)
        cov_xy = jnp.sum(w * (s_sq - x_m) * (y - y_m))

        slope = cov_xy / jnp.maximum(var_x, 1e-12)
        b_dir = jnp.where(jnp.sum(valid) >= 3, jnp.clip(-2.0 * slope, 0.0, 400.0), 0.0)

        # Cap envelope decay at the cone resolution shell
        s_cap = jnp.minimum(s, cutoff_shell / (grid_size * sr))

        env = jnp.exp(- (b_dir / 4.0) * (s_cap ** 2))
        env = env.at[0].set(1.0)
        return env, b_dir

    return jax.vmap(fit_cone)(cross_powers, fsc_curves)


@partial(jax.jit, static_argnames=("grid_size", "hard_cones"))
def _scatter_directional_curves(curves, dirs, shell_idx, grid_size, cone_cos, cone_sigma, hard_cones=False):
    """Cone-weighted angular interpolation of the per-direction curves back onto every rfft voxel."""
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

        # Evita escal?n artificial en las esquinas del cubo de Fourier
        padded_curve = jnp.append(curve, curve[-1])
        shell_val = jnp.take(padded_curve, shell_idx)

        return (acc + cw * shell_val, wsum + cw), None

    init = (jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32))
    (acc, wsum), _ = jax.lax.scan(body, init, (dirs, curves))
    return acc / jnp.maximum(wsum, 1e-8)


def _smooth_radial_curves(curves):
    """Suavizado radial [0.25, 0.5, 0.25] para eliminar el ruido Monte Carlo de fsc_rand en conos estrechos."""
    kernel = jnp.array([0.25, 0.5, 0.25], dtype=jnp.float32)
    def smooth_1d(c):
        padded = jnp.pad(c, (1, 1), mode="edge")
        return jnp.convolve(padded, kernel, mode="valid")
    return jax.vmap(smooth_1d)(curves) if curves.ndim == 2 else smooth_1d(curves)


def _phase_randomized_map(vol, shell_idx, start_shell, key):
    """Replace phases beyond `start_shell` with random ones, keeping the amplitudes."""
    ft = jnp.fft.rfftn(vol)
    noise = jax.random.normal(key, vol.shape)
    noise_ft = jnp.fft.rfftn(noise)
    phases = jnp.angle(noise_ft)
    randomized = jnp.abs(ft) * jnp.exp(1j * phases)
    ft = jnp.where(shell_idx >= start_shell, randomized, ft)
    return jnp.fft.irfftn(ft, s=vol.shape)


def fsc_resolution(curve, sr, grid_size, threshold=0.143):
    """Resolution (A) at which an FSC curve first crosses `threshold`."""
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

    ft1_raw, ft2_raw = jnp.fft.rfftn(half1), jnp.fft.rfftn(half2)
    fsc_unmasked = _shell_fsc(ft1_raw, ft2_raw, shell_idx, mult, n_shells)

    if mask is not None:
        mask_j = jnp.asarray(np.squeeze(np.asarray(mask)), dtype=jnp.float32)
        m1, m2 = half1 * mask_j, half2 * mask_j
    else:
        m1, m2 = half1, half2

    ft1, ft2 = jnp.fft.rfftn(m1), jnp.fft.rfftn(m2)
    fsc_masked = _shell_fsc(ft1, ft2, shell_idx, mult, n_shells)
    cross_p_spherical = _shell_crosspower(ft1, ft2, shell_idx, mult, n_shells)

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
                fsc_rand, cross_p_rand = _directional_shell_fsc(jnp.fft.rfftn(r1 * mask_j), jnp.fft.rfftn(r2 * mask_j),
                                                                dirs, shell_idx, mult, n_shells, grid_size, cone_cos,
                                                                cone_sigma, hard_cones)
            else:
                fsc_rand = _shell_fsc(jnp.fft.rfftn(r1 * mask_j), jnp.fft.rfftn(r2 * mask_j),
                                      shell_idx, mult, n_shells)
                cross_p_rand = _shell_crosspower(jnp.fft.rfftn(r1 * mask_j), jnp.fft.rfftn(r2 * mask_j),
                                                 shell_idx, mult, n_shells)
            fsc_rand = _smooth_radial_curves(fsc_rand)
        else:
            fsc_rand = jnp.zeros((n_cones, n_shells) if directional else n_shells, jnp.float32)
            cross_p_rand = jnp.zeros_like(fsc_rand)
    else:
        fsc_rand = jnp.zeros((n_cones, n_shells) if directional else n_shells, jnp.float32)
        cross_p_rand = jnp.zeros_like(fsc_rand)

    shells = jnp.arange(n_shells)
    correct_from = start_shell + 2

    def _correct(curve, rand_curve):
        corrected = (curve - rand_curve) / jnp.maximum(1.0 - rand_curve, 1e-3)
        corrected = jnp.where(shells >= correct_from, corrected, curve)
        return corrected, jnp.clip(corrected, 0.0, 0.999)

    if directional:
        raw_curves, cross_p_curves = _directional_shell_fsc(ft1, ft2, dirs, shell_idx, mult, n_shells,
                                                            grid_size, cone_cos, cone_sigma, hard_cones)
        fsc_corrected_dir, curves_clipped = jax.vmap(_correct)(raw_curves, fsc_rand)
        fsc_corrected_spherical, _ = _correct(fsc_masked,
                                              jnp.mean(fsc_rand, axis=0) if fsc_rand.ndim == 2 else fsc_rand)
    else:
        fsc_corrected_spherical, curves_clipped = _correct(fsc_masked, fsc_rand)
        curves_clipped = curves_clipped[None, :]
        cross_p_curves = cross_p_spherical[None, :]

    # Cross-power left after phase randomization is mask correlation, not signal
    cross_p_curves = jnp.where(shells >= correct_from, cross_p_curves - jnp.atleast_2d(cross_p_rand), cross_p_curves)

    # C?lculo de curvas Wiener y Envolvente de Guinier
    curves_wiener = (2.0 * curves_clipped / (1.0 + curves_clipped)).at[:, 0].set(1.0)
    curves_env, b_factors = _fit_guinier_envelope(cross_p_curves, curves_clipped, n_shells, grid_size, sr)

    if mode == "fsc_ref":
        curves_w = jnp.sqrt(curves_wiener)
    elif mode == "wiener":
        curves_w = curves_wiener
    elif mode == "envelope":
        curves_w = curves_env
    elif mode == "guinier_wiener":
        curves_w = curves_env * curves_wiener
    elif mode == "fsc":
        curves_w = curves_clipped
    else:
        raise ValueError(f"Unknown FSC weighting mode: {mode}")

    curves_w = curves_w.at[:, 0].set(1.0)

    if directional:
        weight = _scatter_directional_curves(curves_w, dirs, shell_idx, grid_size,
                                             cone_cos, cone_sigma, hard_cones)
        snr_3d = _scatter_directional_curves(curves_wiener, dirs, shell_idx, grid_size,
                                             cone_cos, cone_sigma, hard_cones)
    else:
        weight = curves_w[0][shell_idx]
        snr_3d = curves_wiener[0][shell_idx]

    weight = jnp.clip(weight, 0.0, 1.0)
    snr_3d = jnp.clip(snr_3d, 0.0, 1.0)

    res = fsc_resolution(fsc_corrected_spherical, sr, grid_size)

    info = {"fsc_unmasked": np.asarray(fsc_unmasked),
            "fsc_masked": np.asarray(fsc_masked),
            "fsc_random": np.asarray(fsc_rand),
            "fsc_corrected": np.asarray(fsc_corrected_spherical),
            "resolution": res,
            "randomization_shell": start_shell,
            "n_shells": n_shells,
            "b_factors": np.asarray(b_factors),
            "snr_weight": snr_3d}

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
        print(f"  FSC = 0.143 at {res:.2f} A | mean B = {float(jnp.mean(b_factors)):.1f} A^2 | mean weight {float(jnp.mean(weight)):.3f}")

    return weight, info