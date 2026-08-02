import numpy as np
from jax import numpy as jnp


def ctf_freqs(shape, d=1.0, full=True):
    """
    :param shape: Shape tuple.
    :param d: Frequency spacing in inverse Å (1 / pixel size).
    :param full: When false, return only unique Fourier half-space for real data.
    """
    if full:
        xfrq = jnp.fft.fftfreq(shape[1], dtype=jnp.float32)
    else:
        xfrq = jnp.fft.rfftfreq(shape[1], dtype=jnp.float32)
    x, y = jnp.meshgrid(xfrq, jnp.fft.fftfreq(shape[0], dtype=jnp.float32))
    rho = jnp.sqrt(x ** 2. + y ** 2.)
    a = jnp.atan2(y, x)
    s = rho * d
    return s, a

def eval_ctf(s, a, def1, def2, angast=0., phase=0., kv=300., ac=0.1, cs=2.0, bf=0., lp=0.):
    """
    :param s: Precomputed frequency grid for CTF evaluation.
    :param a: Precomputed frequency grid angles.
    :param def1: 1st prinicipal underfocus distance (Å).
    :param def2: 2nd principal underfocus distance (Å).
    :param angast: Angle of astigmatism (deg) from x-axis to azimuth.
    :param phase: Phase shift (deg).
    :param kv:  Microscope acceleration potential (kV).
    :param ac:  Amplitude contrast in [0, 1.0].
    :param cs:  Spherical aberration (mm).
    :param bf:  B-factor, divided by 4 in exponential, lowpass positive.
    :param lp:  Hard low-pass filter (Å), should usually be Nyquist.
    """
    angast = jnp.deg2rad(angast)
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / jnp.sqrt(kv * (1. + kv * 0.978466e-6))
    def_avg = -(def1 + def2) * 0.5
    def_dev = -(def1 - def2) * 0.5
    k1 = np.pi / 2. * 2. * lamb
    k2 = np.pi / 2. * cs * lamb ** 3.
    k3 = np.sqrt(1. - ac ** 2.)
    k4 = bf / 4.  # B-factor, follows RELION convention.
    k5 = np.deg2rad(phase)  # Phase shift.
    if lp != 0:  # Hard low- or high-pass.
        s *= s <= (1. / lp)
    s_2 = s ** 2.
    s_4 = s_2 ** 2.
    dZ = def_avg[:, None, None] + def_dev[:, None, None] * (jnp.cos(2. * (a - angast[:, None, None])))
    gamma = (k1 * dZ * s_2) + (k2[:, None, None] * s_4) - k5
    ctf = -(k3 * jnp.sin(gamma) - ac * jnp.cos(gamma))
    if bf != 0:  # Enforce envelope.
        ctf *= jnp.exp(-k4 * s_2)
    return ctf

def computeCTF(defocusU, defocusV, defocusAngle, cs, kv, sr, img_shape, batch_size, applyCTF):
    if applyCTF:
        s, a = ctf_freqs([img_shape[0], img_shape[0]], 1 / sr)
        # ``eval_ctf`` introduces the particle axis through defocus/angle.  Keep
        # the invariant frequency grids two-dimensional and let broadcasting do
        # the work instead of materialising two batch-sized copies every step.
        ctf = eval_ctf(s, a, defocusU, defocusV, angast=defocusAngle, cs=cs, kv=kv)
        ctf = jnp.fft.fftshift(ctf[:, :, :img_shape[1]])
        return ctf
    else:
        return jnp.ones([batch_size, img_shape[0], img_shape[1]], dtype=jnp.float32)
