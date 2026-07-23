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

import os
import sys
from collections import deque
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import dm_pix

from .ctf import eval_ctf
from .euler import euler_matrix_batch
from .loggers import bcolors

__all__ = ["reconstruct_consensus_volume", "consensus_mask", "report_half_map_resolution",
           "build_knn_field_operator", "apply_knn_field",
           "reconstruct_motion_corrected_volume"]


def build_knn_field_operator(source_coords_vox, target_coords_vox, k=6, h=None):
    """A fixed convex-combination operator carrying a sparse field onto dense voxels.

    Used for motion models whose field is only defined on their own points and cannot be
    queried elsewhere (Zernike3Deep, or a HetSIREN without mass transport). Each target voxel
    takes a Gaussian-weighted average of its ``k`` nearest source points, with the weights
    NORMALISED TO SUM TO ONE.

    That normalisation is the whole point, and it is what the previous splat-blur-divide did
    not have. A convex combination of the source displacements is bounded by the largest of
    them, so no configuration of points can produce an output bigger than the field itself.
    The old path divided by a blurred coverage channel that could -- and did -- change sign,
    which is unbounded: it delivered displacements of tens of thousands of Angstrom.

    Built once on the host; per step it is a gather plus a contraction.
    """
    from scipy.spatial import cKDTree

    source = np.asarray(source_coords_vox, np.float32)
    target = np.asarray(target_coords_vox, np.float32)
    k = int(min(k, source.shape[0]))
    dist, idx = cKDTree(source).query(target, k=k, workers=-1)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]

    if h is None:
        # Typical spacing between source points, so the kernel is wide enough to interpolate
        # between them but not so wide that it averages a moving domain with a static one.
        h = float(np.median(dist[:, 0]))
    h = max(float(h), 1e-3)

    w = np.exp(-0.5 * (dist / h) ** 2)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return jnp.asarray(idx.astype(np.int32)), jnp.asarray(w.astype(np.float32))


@jax.jit
def apply_knn_field(field_source, idx, weights):
    """Carry ``(B, S, 3)`` source displacements onto ``(B, T, 3)`` target voxels."""
    return jnp.einsum("tk,btkc->btc", weights, field_source[:, idx, :])


def _slice_geometry(box, sr):
    """Centred frequency grids and the identity-pose central slice, in index units."""
    g = (np.fft.fftshift(np.fft.fftfreq(box)) * box).astype(np.float32)
    f0, f1 = np.meshgrid(g, g, indexing="ij")                     # f0 <-> row, f1 <-> col
    s = np.sqrt((f0 / box) ** 2 + (f1 / box) ** 2) / sr           # spatial frequency, 1/A
    a = np.arctan2(f0, f1)                                        # azimuth, for astigmatism
    k_rot = np.stack([f1, f0, np.zeros_like(f0)], 0)              # component order (x, y, z)
    return (jnp.asarray(s), jnp.asarray(a), jnp.asarray(k_rot),
            jnp.asarray(f0), jnp.asarray(f1))


@partial(jax.jit, static_argnums=(7,), donate_argnums=(0, 1))
def _insert_slices(num, den, images, rotations, shifts, ctf, k_rot, box, f0, f1):
    """Accumulate one batch of CTF-weighted central slices into the 3D transform.

    The accumulators are **donated**, so the scatter lands in place. Without that, XLA has to
    preserve the caller's buffers: every call allocates a fresh ``num`` (262 MB at box 320) and
    ``den`` (131 MB), copies the running totals in, and frees the old pair afterwards. The copy
    is wasted bandwidth, but the free is worse -- with preallocation disabled it is a
    ``cudaFree``, which *synchronizes the device*, so the GPU drains and idles once per batch
    instead of staying fed. Donation means the caller's ``num``/``den`` are invalid after the
    call, which is exactly how the streaming loop uses them (it rebinds the result).
    """
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

    # Only the slice itself is scattered here; its Friedel mate is added once at the end of
    # the pass by ``_hermitian_symmetrize``, which is the same thing for half the atomics.
    return scatter(num, den, k, data, weight)


def _hermitian_symmetrize(num, den):
    """Add the Friedel mate F(-k) = conj(F(k)) of everything accumulated so far.

    Inserting each slice's mate as it streams by (the obvious way) doubles the scatter work
    of every single particle, and the scatter is what this reconstruction spends its GPU time
    on. It is also unnecessary: the trilinear neighbourhood of ``-k`` is the exact mirror of
    the neighbourhood of ``+k``, with mirrored weights -- for a sample at ``pos = k + c``,
    ``floor(2c - pos) = 2c - floor(pos) - 1``, so the two corner sets mirror into each other
    and the interpolation weight ``1 - frac`` mirrors ``frac``. Mirroring the accumulators
    once at the end therefore reproduces the double insertion exactly, at the cost of one pass
    over the grid instead of doubling the cost of one pass over the *data*.

    ``roll(flip(A), 1)`` maps voxel ``i`` to its mate ``box - i``. The wrap at ``i = 0`` is
    not an accident: that plane is Nyquist, which under the DFT's periodicity is its own
    Friedel mate and must come out real -- and ``A[0] + conj(A[0])`` is exactly what makes it so.
    """
    def mirror(a):
        return jnp.roll(jnp.flip(a, axis=(0, 1, 2)), shift=(1, 1, 1), axis=(0, 1, 2))
    return num + jnp.conj(mirror(num)), den + mirror(den)


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


def _gray_scale(volume, reader, columns, sr, box, has_ctf, k_rot, f0, f1, s, a, n_probe=2000,
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
    n = min(int(n_probe), reader.n)
    if n < 8:
        return 1.0

    idx = np.linspace(0, reader.n - 1, n).astype(np.int64)
    volume_ft = jnp.asarray(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(np.asarray(volume)))))
    angles = np.asarray(columns["euler_angles"], np.float32)
    shifts = np.asarray(columns["shifts"], np.float32)
    kv = float(np.asarray(columns["ctfVoltage"]).ravel()[0]) if has_ctf else 0.0

    cross = 0.0
    energy = 0.0
    for start in range(0, n, batch_size):
        chunk = idx[start:start + batch_size]
        images = reader.read(chunk)

        ang = jnp.asarray(angles[chunk])
        rotations = euler_matrix_batch(ang[:, 0], ang[:, 1], ang[:, 2])
        if has_ctf:
            ctf = eval_ctf(s[None], a[None],
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


class _StackReader:
    """Where the particle images are read from: the original stack, or an SSD cache of it.

    The reconstruction touches every image exactly once, so on a fast disk reading the stack
    directly is optimal and the cache is pure overhead. On a slow one (an HDD sustains ~150
    MB/s, and this program has to move ``n * box^2 * 4`` bytes -- 410 GB for 1M particles at
    box 320) the read *is* the run, and it is worth paying once for a local float16 copy that
    every later pass, and every other hax program pointed at the same scratch folder, reads
    instead. That copy is exactly the ``images_mmap_grain`` array-record HetSIREN/MoDART already
    build, so it is shared rather than duplicated -- whoever runs first pays for it.

    Both back-ends expose the same two operations: read an arbitrary set of rows (for the
    gray-scale probe) and stream the whole set in order (for the insertion pass).
    """

    def __init__(self, md, scratch_dir=None):
        self.md = md
        self.n = len(md)
        self.source = None
        if scratch_dir is not None:
            from array_record.python.array_record_data_source import ArrayRecordDataSource
            from glob import glob
            shards = sorted(glob(os.path.join(scratch_dir, "dataset-*.arrayrecord")))
            if shards:
                self.source = ArrayRecordDataSource(
                    shards, reader_options={"index_storage_option": "in_memory"})

    @property
    def cached(self):
        return self.source is not None

    def read(self, idx):
        """Images for the metadata rows ``idx`` (arbitrary order), as float32."""
        if self.source is None:
            return np.asarray(self.md.getMetaDataImage(np.asarray(idx, np.int64)), np.float32)
        from hax.generators.generator_metadata import parse_and_decompress
        # __getitems__ (plural) is the batched read; __getitem__ takes a single key.
        records = self.source.__getitems__([int(i) for i in np.asarray(idx, np.int64)])
        return np.stack([parse_and_decompress(r)[0][..., 0] for r in records]).astype(np.float32)

    def _read_chunk(self, start, stop):
        """A contiguous run of rows -- plus the row ids, which the cache carries per record."""
        idx = np.arange(start, stop, dtype=np.int64)
        if self.source is None:
            return idx, np.asarray(self.md.getMetaDataImage(idx), np.float32)
        from hax.generators.generator_metadata import parse_and_decompress
        records = self.source.__getitems__([int(i) for i in idx])
        decoded = [parse_and_decompress(r) for r in records]
        # The record's own label is the metadata row it came from: trust it rather than the
        # record position, so a cache written in a shuffled order still lines up with the poses.
        labels = np.asarray([lab for _, lab in decoded], np.int64)
        images = np.stack([img[..., 0] for img, _ in decoded]).astype(np.float32)
        return labels, images


def _stream_chunks(reader, n, batch_size, threads):
    """Yield ``(labels, images)`` chunks, reading ahead on a thread pool.

    The read-ahead is deliberately *bounded*. Submitting every chunk up front and walking the
    resulting list of futures looks equivalent, but a ``Future`` owns its result until it is
    garbage collected, and the list keeps every future alive for the whole run -- so each
    chunk that has been read stays resident even after it has been inserted, and peak RAM
    grows to the size of the entire stack (400+ GB for 1M particles at box 320). There is also
    no back-pressure: the readers race ahead of the GPU as fast as the disk allows.

    Here at most ``2 * threads`` chunks are ever in flight, and each is dropped as soon as it
    has been consumed, so peak RAM tracks the window, not the particle count.
    """
    with ThreadPoolExecutor(max_workers=threads) as pool:
        starts = iter(range(0, n, batch_size))
        window = deque()

        def submit_next():
            start = next(starts, None)
            if start is not None:
                window.append(pool.submit(reader._read_chunk, start, min(start + batch_size, n)))

        for _ in range(2 * threads):
            submit_next()

        while window:
            future = window.popleft()
            labels, images = future.result()
            submit_next()          # refill only now, so the window stays bounded
            yield labels, images
            del labels, images     # drop the chunk before waiting on the next one


def reconstruct_consensus_volume(md, columns, sr, tau=0.05, batch_size=1024, threads=8,
                                 use_ctf=True, denoise=True, calibrate_gray_scale=True,
                                 scratch_dir=None, quiet=False):
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

    ``scratch_dir`` is an ``images_mmap_grain`` folder (see ``_StackReader``): when it holds a
    cached copy of the stack the images are streamed from there instead of from ``md``, which is
    what makes this bearable when the particles live on a spinning disk.

    Returns the volume; when ``denoise`` it also prints the measured resolution.
    Images are read in chunks on a thread pool while the GPU accumulates, so peak RAM
    tracks ``batch_size`` rather than the particle count.
    """
    n = len(md)
    box = int(md.getMetaDataImage(0).shape[0])
    reader = _StackReader(md, scratch_dir)

    s, a, k_rot, f0, f1 = _slice_geometry(box, sr)
    # Two independent half-sets: same total insertion work, but the FSC between them is what
    # tells us how far the data actually goes. Each half gets its OWN accumulator: sharing one
    # zeros array between them would be fine without donation and fatal with it (the first
    # insert would consume the buffer the second half still expects to read).
    num = [jnp.zeros((box,) * 3, jnp.complex64) for _ in range(2)]
    den = [jnp.zeros((box,) * 3, jnp.float32) for _ in range(2)]

    angles = np.asarray(columns["euler_angles"], np.float32)
    shifts = np.asarray(columns["shifts"], np.float32)
    has_ctf = use_ctf and "ctfDefocusU" in columns
    if has_ctf:
        kv = float(np.asarray(columns["ctfVoltage"]).ravel()[0])

    if not quiet:
        print(f"{bcolors.OKCYAN}\n###### Reconstructing consensus volume from {n} posed particles... ######{bcolors.ENDC}")

    n_chunks = (n + batch_size - 1) // batch_size
    for labels, images in tqdm(_stream_chunks(reader, n, batch_size, threads), total=n_chunks,
                               file=sys.stdout, ascii=" >=", colour="green", disable=quiet):
        # One host->device transfer for the whole chunk. The halves are then selected on the
        # device: `images[sel]` would fancy-index a fresh copy of half the chunk in
        # single-threaded numpy (100+ MB per half at box 320) while the GPU sits idle, and then
        # transfer each half separately.
        images_dev = jnp.asarray(images)

        for half in (0, 1):
            # Interleave the halves so both see the same pose and defocus distribution.
            # Particle `i` belongs to half `i % 2` -- keyed off the metadata row, so the split
            # is the same one no matter what order the images arrived in.
            pos = np.flatnonzero(labels % 2 == half)
            if pos.size == 0:
                continue
            idx = labels[pos]

            ang = jnp.asarray(angles[idx])
            rotations = euler_matrix_batch(ang[:, 0], ang[:, 1], ang[:, 2])

            if has_ctf:
                # eval_ctf indexes the per-particle parameters as [:, None, None], so the two
                # frequency grids broadcast from (1, box, box). Tiling them to (b, box, box)
                # first would materialize two copies per batch -- 840 MB at batch_size=1024,
                # box=320 -- for values that are identical across the batch.
                ctf = eval_ctf(s[None], a[None],
                               jnp.asarray(np.asarray(columns["ctfDefocusU"])[idx]),
                               jnp.asarray(np.asarray(columns["ctfDefocusV"])[idx]),
                               angast=jnp.asarray(np.asarray(columns["ctfDefocusAngle"])[idx]),
                               cs=jnp.asarray(np.asarray(columns["ctfSphericalAberration"])[idx]),
                               kv=kv)
            else:
                ctf = jnp.ones((idx.shape[0], box, box), jnp.float32)

            num[half], den[half] = _insert_slices(
                num[half], den[half], images_dev[jnp.asarray(pos)], rotations,
                jnp.asarray(shifts[idx]), ctf, k_rot, box, f0, f1)

    # The Friedel mates of every slice, added in one pass rather than during the streaming.
    num[0], den[0] = _hermitian_symmetrize(num[0], den[0])
    num[1], den[1] = _hermitian_symmetrize(num[1], den[1])

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
        scale = _gray_scale(volume, reader, columns, sr, box, has_ctf, k_rot, f0, f1, s, a)
        volume = volume * scale
        if not quiet:
            print(f"{bcolors.OKGREEN}Gray-scale calibrated to the input images (x{scale:.3f}); "
                  f"projecting the map reproduces their contrast.{bcolors.ENDC}")

    return np.asarray(volume, np.float32)


@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def _unwarp_images(images, points, displacements, weights, shifts, rotations,
                   box, warp_sigma, min_coverage, jacobian=False):
    """Undo each particle's modelled motion in its own image plane.

    The deformation lives in 3D, and a general 3D warp does not act on a central Fourier
    slice -- so it cannot be applied exactly during insertion. What *can* be applied exactly
    is its shadow: a point at canonical position ``c`` that the model displaces to ``c + d``
    moves, in the image, from ``(R c)_xy`` to ``(R c + R d)_xy``. Resampling the observed
    image at ``p + u(p)`` therefore pulls the density back to where the canonical structure
    would have put it, and the corrected image can then be inserted by the ordinary slice
    machinery with the ordinary CTF weighting.

    ``points`` ``(N, 3)`` are the canonical positions the field is sampled at (voxel units,
    box-centred; shared by every particle), ``displacements`` ``(B, N, 3)`` the modelled motion
    there, ``weights`` ``(N,)`` the consensus density at those points -- the warp should be set
    by where the mass is, not by empty space.

    The scattered 2D displacements are turned into a dense field by a normalised (Shepard)
    splat. Two properties make that safe here and are worth stating, because getting them
    wrong is exactly how the previous 3D version failed: the smoothing kernel is a real-space
    truncated Gaussian, so it is NON-NEGATIVE and the coverage denominator cannot change sign;
    and the projection is dense (~10^4 points over ~10^4 pixels, versus 10^4 over 2x10^6
    voxels in 3D), so almost every pixel has real support. Where it does not -- coverage below
    ``min_coverage`` of the mean -- the warp is set to zero rather than divided up, i.e. the
    image is left alone where the model has nothing to say.
    """
    b = images.shape[0]
    centre = box // 2

    # Canonical and displaced positions in the observed image frame, as (row, col).
    # The canonical points are shared across the batch; only the displacement is per-particle.
    p3 = jnp.einsum("bij,nj->bni", rotations, points)
    d3 = jnp.einsum("bij,bnj->bni", rotations, displacements)
    row = p3[..., 1] - shifts[:, 1, None] + centre
    col = p3[..., 0] - shifts[:, 0, None] + centre
    d_row, d_col = d3[..., 1], d3[..., 0]

    # Bilinear splat of (w * u) and of w onto the image grid.
    base_r, base_c = jnp.floor(row).astype(jnp.int32), jnp.floor(col).astype(jnp.int32)
    fr, fc = row - base_r, col - base_c
    acc = jnp.zeros((b, box, box, 3), jnp.float32)
    w = weights[None, :]

    def splat(acc, dr, dc):
        idx_r, idx_c = base_r + dr, base_c + dc
        wt = ((fr if dr else 1.0 - fr) * (fc if dc else 1.0 - fc)) * w
        inside = (idx_r >= 0) & (idx_r <= box - 1) & (idx_c >= 0) & (idx_c <= box - 1)
        wt = wt * inside
        ir, ic = jnp.clip(idx_r, 0, box - 1), jnp.clip(idx_c, 0, box - 1)
        payload = jnp.stack([wt * d_row, wt * d_col, wt], axis=-1)
        return jax.vmap(lambda a, r, c, v: a.at[r, c].add(v))(acc, ir, ic, payload)

    for dr in (0, 1):
        for dc in (0, 1):
            acc = splat(acc, dr, dc)

    acc = dm_pix.gaussian_blur(acc, sigma=warp_sigma,
                               kernel_size=int(2 * round(3 * warp_sigma) + 1))
    cover = acc[..., 2]
    floor = min_coverage * (jnp.mean(cover, axis=(1, 2), keepdims=True) + 1e-12)
    live = cover > floor
    inv = jnp.where(live, 1.0 / jnp.maximum(cover, 1e-12), 0.0)
    u_row, u_col = acc[..., 0] * inv, acc[..., 1] * inv

    # Resample the observed image at p + u(p).
    gr, gc = jnp.meshgrid(jnp.arange(box, dtype=jnp.float32),
                          jnp.arange(box, dtype=jnp.float32), indexing="ij")
    sample_r, sample_c = gr[None] + u_row, gc[None] + u_col
    corrected = jax.vmap(lambda im, r, c: jax.scipy.ndimage.map_coordinates(
        im, jnp.stack([r, c]), order=1, mode="nearest"))(images, sample_r, sample_c)

    if jacobian:
        # Resampling moves values; mass needs the Jacobian of the warp as well. Where the
        # warp compresses, the same material lands on fewer pixels, so the resampled value
        # has to be scaled up by the area ratio for the line integral to be preserved.
        # Second order for a near-isometric deformation -- measured at 1.2% median on the
        # ribosome, but 30% at p99, and it grows in proportion to the size of the warp.
        def d(a, axis):
            return 0.5 * (jnp.roll(a, -1, axis=axis) - jnp.roll(a, 1, axis=axis))
        det = ((1.0 + d(u_row, 1)) * (1.0 + d(u_col, 2))
               - d(u_row, 2) * d(u_col, 1))
        corrected = corrected * jnp.where(live, det, 1.0)

    # How far the images were actually moved, averaged over the pixels the model had
    # something to say about. This is the one number that says whether the run did
    # anything, and it must be reported: a model whose deformation has collapsed produces
    # a warp of a hundredth of a pixel, and the output is then the consensus map with a
    # different name -- which looks like a successful run in every other respect.
    live_u = jnp.sqrt(u_row ** 2 + u_col ** 2) * live
    mean_shift = jnp.sum(live_u) / (jnp.sum(live) + 1e-8)
    return corrected, mean_shift


@partial(jax.jit, static_argnums=(5,), donate_argnums=(0,))
def _insert_weights(den, rotations, ctf, k_rot, box, _unused=None):
    """Accumulate only the CTF power, for a numerator built elsewhere.

    ``_insert_slices`` builds numerator and denominator together, which is right when both
    come from the same slice insertion. The deformed backprojection builds its numerator in
    real space instead, so it needs this half on its own -- and the denominator is unchanged
    by the deformation, because it is a property of the CTF and the pose, not of the image.
    Leaving it undeformed is the same approximation DynaMight makes (its ``CTFy`` term is
    never resampled onto the deformed grid): the exact weight would be the deformed
    forward operator's normal matrix, which is not diagonal in Fourier space at all.
    """
    weight = ctf ** 2
    k = jnp.einsum("bji,jhw->bihw", rotations, k_rot)
    k = jnp.stack([k[:, 2], k[:, 1], k[:, 0]], 1)

    pos = k + box // 2
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
                den = den.at[z, y, x].add(weight * w)
    return den


def _grid_neighbours(mask_idx, box):
    """For each masked voxel, the positions of its +-1 neighbours WITHIN the masked list.

    Returned as ``(minus, plus, inv_step)``: index arrays ``(M, 3)`` into the masked voxel
    list and the reciprocal of the actual finite-difference step ``(M, 3)``. A neighbour that
    falls outside the mask is replaced by the voxel itself and the step halves accordingly,
    so the difference degrades to one-sided at the mask boundary rather than reaching for a
    voxel that carries no field.
    """
    m = np.asarray(mask_idx, np.int64)                      # flat indices into box^3
    lookup = np.full(box ** 3, -1, np.int64)
    lookup[m] = np.arange(m.size)
    strides = np.array([box * box, box, 1], np.int64)       # array order [z, y, x]
    zyx = np.stack([m // (box * box), (m // box) % box, m % box], 1)

    minus = np.zeros((m.size, 3), np.int64)
    plus = np.zeros((m.size, 3), np.int64)
    inv_step = np.zeros((m.size, 3), np.float32)
    self_pos = np.arange(m.size)
    for axis in range(3):                                   # axis 0=z, 1=y, 2=x
        # Clip before the gather: a voxel on the far face has m + stride past the end of the
        # lookup, and the np.where would still have evaluated the out-of-range index.
        safe = lambda f: lookup[np.clip(f, 0, box ** 3 - 1)]
        lo = np.where(zyx[:, axis] > 0, safe(m - strides[axis]), -1)
        hi = np.where(zyx[:, axis] < box - 1, safe(m + strides[axis]), -1)
        has_lo, has_hi = lo >= 0, hi >= 0
        minus[:, axis] = np.where(has_lo, lo, self_pos)
        plus[:, axis] = np.where(has_hi, hi, self_pos)
        span = has_lo.astype(np.float32) + has_hi.astype(np.float32)
        inv_step[:, axis] = np.where(span > 0, 1.0 / np.maximum(span, 1.0), 0.0)

    # The field components are ordered (x, y, z) but the grid axes above are (z, y, x), so
    # reverse the axis order once here rather than at every use.
    return (jnp.asarray(minus[:, ::-1].copy()), jnp.asarray(plus[:, ::-1].copy()),
            jnp.asarray(inv_step[:, ::-1].copy()))


@partial(jax.jit, static_argnums=(4, 5))
def _field_jacobian(disp, minus, plus, inv_step, limit=2.0, smooth_passes=2):
    """det(I + grad d) at every masked voxel, from the field on the masked voxels alone.

    This is the factor that turns a resampling into a mass-conserving one. Moving material
    with a deformation phi maps a density by rho_0(c) = rho_def(phi(c)) * det(grad phi), and
    reading the image without that determinant conserves *values* rather than *mass*: a
    region the deformation compresses comes back too bright and too small.

    The raw determinant is NOT usable as-is, and the reason is the field it differentiates.
    The displacement reaches these voxels through a k-NN interpolation, which is continuous
    but not smooth: the set of source points changes from one voxel to the next, so the field
    has kinks and a finite difference across one produces a spike that is an artefact of the
    interpolation, not a property of the deformation. Measured on the ribosome: median
    |det-1| = 0.022 (the real, small effect) but a tail reaching det = -2.69 to +2.88, with
    573 voxels at det < 0 -- an orientation flip, which multiplies the density by a NEGATIVE
    number and is physically impossible for an elastically regularised deformation. The
    visible cost was single-voxel speckle: peak roughness 1.81x the consensus with the
    Jacobian on, 1.00x with it off.

    So the determinant is clamped to [1/limit, limit] -- a deformation that halves or doubles
    local volume within one voxel is differentiation noise, not signal -- and then smoothed
    over the same 6-neighbour stencil, because the true determinant field varies on the
    deformation's length scale rather than per voxel.
    """
    g = (disp[:, plus, :] - disp[:, minus, :]) * inv_step[None, :, :, None]
    # g[b, m, axis, comp] = d(d_comp)/d(x_axis); the Jacobian wants [comp, axis].
    det = jnp.linalg.det(jnp.eye(3)[None, None] + jnp.swapaxes(g, -1, -2))
    det = jnp.clip(det, 1.0 / limit, limit)

    # Clamp first, so a wild voxel is bounded before it is spread over its neighbours.
    for _ in range(int(smooth_passes)):
        acc = det
        for axis in range(3):
            acc = acc + det[:, minus[:, axis]] + det[:, plus[:, axis]]
        det = acc / 7.0
    return det


@partial(jax.jit, static_argnums=(5,))
def _deformed_backproject(images, points, displacements, rotations, shifts, box, jac=None,
                          window=None):
    """The DIFFERENCE between a deformed and a straight backprojection, summed over a batch.

    The value belonging at canonical voxel ``c`` is read from wherever *that voxel's own*
    material landed, ``P(R(c + d(c)))``. Two voxels at different depths on one ray therefore
    read different pixels -- which is exactly what a single-valued 2D warp cannot do, and on
    the ribosome the part of the field that varies along a ray is ~95% the size of the part
    that does not. This is the correction DynaMight makes (Nat Methods 21, 1855-1862).

    Returned as a difference against the straight backprojection, rather than as the deformed
    backprojection itself, for two reasons. It is what makes restricting the work to the
    molecule legitimate: the difference is identically zero wherever the deformation is zero,
    whereas a truncated backprojection is not -- masking and the Fourier deconvolution that
    follows it do not commute, which is why DynaMight has to backproject the whole box. And
    it makes the mode reduce EXACTLY to the ordinary consensus when the model predicts no
    motion, so a null result is unmistakably null rather than a slightly different estimator.

    Note the direction: what is needed is the FORWARD field evaluated at CANONICAL points,
    which is what ``decode_field_at`` returns. DynaMight trains a separate inverse-deformation
    network for this step because its decoder only moves a sparse set of pseudo-atoms; a
    coordinate network that can be evaluated anywhere does not need one.
    """
    def project(c):
        p = jnp.einsum("bij,bnj->bni", rotations, c)
        return (p[..., 1] - shifts[:, 1, None] + box // 2,
                p[..., 0] - shifts[:, 0, None] + box // 2)

    # 'constant' with cval 0: a voxel whose material projects outside the frame was not
    # observed, so it must contribute nothing rather than borrow the nearest edge pixel.
    def sample(row, col):
        return jax.vmap(lambda im, r, c_: jax.scipy.ndimage.map_coordinates(
            im, jnp.stack([r, c_]), order=1, mode="constant", cval=0.0))(images, row, col)

    moved = sample(*project(points[None] + displacements))
    if jac is not None:
        moved = moved * jac
    rest = sample(*project(jnp.broadcast_to(points[None], displacements.shape)))

    # The window multiplies the DIFFERENCE, not the displacement. Windowing the displacement
    # instead makes the taper look like a real deformation to the Jacobian -- a field of
    # ~0.4 voxels faded over 2 voxels carries a divergence of ~0.2, so det(I + grad d) picks
    # up a spurious 20-50% amplitude ring exactly where the window acts. Applying it here
    # leaves the Jacobian describing the model's own field and still takes the correction
    # smoothly to zero.
    delta = moved - rest
    if window is not None:
        delta = delta * window
    return jnp.sum(delta, axis=0)


@jax.jit
def _ctf_correlate(images, ctf):
    """The real-space image whose straight backprojection equals the CTF-weighted slice.

    The Fourier path inserts ``CTF * F(image)``; its real-space equivalent is the inverse
    transform of that product, so the two numerators agree term by term and the two modes
    stay comparable.
    """
    ft = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(images, axes=(-2, -1))), axes=(-2, -1))
    return jnp.real(jnp.fft.fftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(ft * ctf, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1)))


def _points_mask(points, weights, box, radius, taper=0.0):
    """Voxels within ``radius`` of a consensus point that carries mass, and a soft window.

    The deformed backprojection costs one gather per voxel per particle, so it is run over
    the molecule rather than the whole box -- and the model has nothing to say outside its
    own point cloud anyway. Derived from the motion model's own consensus so the mode needs
    no reference map of its own.

    ``taper`` is the width, in voxels, over which the returned window falls from 1 to 0 as
    the mask boundary is approached. It is not cosmetic. The correction is a DIFFERENCE, so
    cutting it off at a hard boundary leaves a step of the correction's full size, and that
    step rings through the Fourier division that follows. Measured on the ribosome with no
    taper: 71.6% of the total squared difference between the corrected map and the consensus
    sat in a +-1.5 voxel shell at the mask edge (5.1% of the box by volume), where it was
    1.43x the size of the density itself, and the difference spectrum climbed with frequency
    exactly as edge ringing does. The interpolated field does not decay on its own -- its
    outermost band was the LARGEST of any (mean 0.48 voxels) -- so the window has to impose
    the decay. Raised cosine, so the window is C1 at both ends and adds no corner of its own.
    """
    from scipy.spatial import cKDTree

    live = np.asarray(points, np.float32)[np.asarray(weights, np.float32) > 0]
    g = np.arange(box, dtype=np.float32) - box // 2
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")            # volume order [z, y, x]
    grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)    # component order (x, y, z)
    dist, _ = cKDTree(live).query(grid, k=1, workers=-1)
    keep = np.flatnonzero(dist <= float(radius))

    taper = float(max(taper, 0.0))
    if taper <= 0.0:
        window = np.ones(keep.size, np.float32)
    else:
        t = np.clip((float(radius) - dist[keep]) / taper, 0.0, 1.0)
        window = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)
    return keep.astype(np.int64), grid[keep], window


def reconstruct_motion_corrected_volume(md, columns, sr, motion_model, tau=0.05,
                                        batch_size=512, field_batch=32, threads=8,
                                        use_ctf=True, denoise=True, calibrate_gray_scale=True,
                                        scratch_dir=None, warp_sigma=2.0, min_coverage=0.05,
                                        correction="image", jacobian=False, mask_radius=5.0, mask_taper=2.0,
                                        quiet=False):
    """Motion-corrected reconstruction by Fourier gridding, in one streaming pass.

    Same estimator as ``reconstruct_consensus_volume`` -- CTF-weighted central-slice insertion
    with a Wiener quotient -- with each particle's modelled motion removed from its image
    first (see ``_unwarp_images``). ``motion_model`` is a trained HetSIREN with mass transport;
    its encoder supplies the per-particle latent and its decoder the displacement field.

    This exists as an alternative to MoDART's iterative fit because the fit is the wrong
    estimator for this problem, not merely a slow one. Fitting ~10^5 free voxel amplitudes by
    Adam on an image-space residual leaves every direction the forward operator is blind to
    completely unconstrained -- and Adam gives an unconstrained direction a full-size step, so
    those directions fill with noise rather than staying at their initial value. Gridding has
    no free parameters at all: each Fourier voxel is a CTF-weighted average of the data that
    touched it, shells nobody measured stay empty instead of filling with noise, and the whole
    thing costs one pass over the images instead of tens of thousands of gradient steps.

    Returns ``(volume, extras)`` where ``extras`` holds the two half maps, the FSC and the
    measured resolution. Half maps are always produced: without them there is no way to tell
    a reconstruction from its own input.
    """
    n = len(md)
    box = int(md.getMetaDataImage(0).shape[0])
    reader = _StackReader(md, scratch_dir)

    s, a, k_rot, f0, f1 = _slice_geometry(box, sr)
    num = [jnp.zeros((box,) * 3, jnp.complex64) for _ in range(2)]
    den = [jnp.zeros((box,) * 3, jnp.float32) for _ in range(2)]

    angles = np.asarray(columns["euler_angles"], np.float32)
    shifts_all = np.asarray(columns["shifts"], np.float32)
    has_ctf = use_ctf and "ctfDefocusU" in columns
    if has_ctf:
        kv = float(np.asarray(columns["ctfVoltage"]).ravel()[0])

    # The field is sampled at the decoder's own points, which is where it is defined; the
    # projection to 2D is what puts it on the image grid, so no 3D interpolation is needed.
    from flax import nnx
    graphdef_motion, state_motion = nnx.split(motion_model)
    dvd = motion_model.delta_volume_decoder

    # The model's geometry is in ITS box, which need not be the reconstruction box (a motion
    # model is often trained on downsampled particles). Both the rest positions and the
    # displacements are lengths, so both scale by the same ratio.
    box_scale = box / float(motion_model.xsize)
    c0, w0 = dvd.coords, dvd.reference_values
    rest_points = np.asarray(dvd.scale * c0[0], np.float32) * box_scale
    point_weights = jnp.asarray(np.maximum(np.asarray(w0[0], np.float32), 0.0))

    @jax.jit
    def decode_field(graphdef, state, images):
        model = nnx.merge(graphdef, state)
        if images.shape[1] != model.xsize:
            images = jax.image.resize(images, (images.shape[0], model.xsize, model.xsize, 1),
                                      method="bilinear")
        field, _ = model.decode_field(images)
        # normalized -> model voxels -> reconstruction voxels
        return field * (0.5 * model.xsize) * box_scale

    if correction not in ("image", "backprojection"):
        raise UserWarning(f"unknown correction mode '{correction}' "
                          f"(expected 'image' or 'backprojection')")

    deformed = correction == "backprojection"
    if deformed:
        # The canonical voxels the rays are bent onto, and the operator that carries the
        # field from the decoder's own points to them. The operator is built ONCE: within a
        # batch only the field changes, so per particle this is a gather and a contraction.
        # (DynaMight instead evaluates its network on the full grid and amortises that by
        # binning the latent space into tiles, which quantises the deformation; interpolating
        # a per-particle field costs less here and keeps every particle its own.)
        mask_idx, mask_points, mask_window = _points_mask(
            rest_points, np.asarray(point_weights), box, mask_radius, mask_taper)
        if mask_idx.size == 0:
            raise UserWarning("the motion model's consensus selected no voxels; "
                              "raise mask_radius")
        knn_idx, knn_w = build_knn_field_operator(rest_points, mask_points)
        mask_points_dev = jnp.asarray(mask_points)

        # The correction has to vanish where there is no material to move. A distance-based
        # window cannot do that -- the field is pure extrapolation in the space between and
        # around the points, and measured on the ribosome the correction there was LARGER
        # than on the density itself (rms 0.012-0.014 at 3-5 voxels out, against 0.008 on the
        # molecule, where the map's own rms is 3x higher). Widening the mask and tapering it
        # only moved that ring outwards. So the window is the DENSITY: the consensus
        # occupancy carried onto the same voxels by the same operator, normalised so the body
        # of the molecule sits at 1 and empty space at 0. It has no boundary to ring at.
        occ = np.asarray(apply_knn_field(
            jnp.asarray(point_weights)[None, :, None], knn_idx, knn_w))[0, :, 0]
        occ_ref = float(np.percentile(occ, 90))
        density_window = np.clip(occ / max(occ_ref, 1e-12), 0.0, 1.0).astype(np.float32)
        # ...times the distance taper, which costs nothing and guarantees an exact zero at
        # the mask boundary however the density happens to behave there.
        window_dev = jnp.asarray(density_window * mask_window)[None, :]
        acc = [jnp.zeros((mask_idx.size,), jnp.float32) for _ in range(2)]
        nb_minus, nb_plus, nb_inv = (_grid_neighbours(mask_idx, box) if jacobian
                                     else (None, None, None))
        if not quiet:
            print(f"{bcolors.OKGREEN}Deformed backprojection over {mask_idx.size} voxels "
                  f"({100.0 * mask_idx.size / box ** 3:.1f}% of the box), "
                  f"weighted by the consensus density ({100.0 * float(np.mean(density_window > 0.5)):.0f}% "
                  f"of them above half weight).{bcolors.ENDC}")

    if not quiet:
        print(f"{bcolors.OKCYAN}\n###### Motion-corrected reconstruction from {n} posed "
              f"particles... ######{bcolors.ENDC}")

    n_chunks = (n + batch_size - 1) // batch_size
    shift_sum, shift_n = 0.0, 0
    for labels, images in tqdm(_stream_chunks(reader, n, batch_size, threads), total=n_chunks,
                               file=sys.stdout, ascii=" >=", colour="green", disable=quiet):
        images_dev = jnp.asarray(images)

        # The field decode holds (B, n_points, hidden) activations, so it runs in smaller
        # sub-batches than the disk reads, which want to be large.
        for start in range(0, labels.shape[0], field_batch):
            sl = slice(start, min(start + field_batch, labels.shape[0]))
            idx = labels[sl]
            imgs = images_dev[sl]

            ang = jnp.asarray(angles[idx])
            rotations = euler_matrix_batch(ang[:, 0], ang[:, 1], ang[:, 2])
            shifts = jnp.asarray(shifts_all[idx])

            field = decode_field(graphdef_motion, state_motion, imgs[..., None])
            if not deformed:
                corrected, mean_shift = _unwarp_images(
                    imgs, jnp.asarray(rest_points), field, point_weights,
                    shifts, rotations, box, float(warp_sigma), float(min_coverage),
                    bool(jacobian))
                shift_sum += float(mean_shift) * idx.shape[0]
                shift_n += idx.shape[0]

            if has_ctf:
                ctf = eval_ctf(s[None], a[None],
                               jnp.asarray(np.asarray(columns["ctfDefocusU"])[idx]),
                               jnp.asarray(np.asarray(columns["ctfDefocusV"])[idx]),
                               angast=jnp.asarray(np.asarray(columns["ctfDefocusAngle"])[idx]),
                               cs=jnp.asarray(np.asarray(columns["ctfSphericalAberration"])[idx]),
                               kv=kv)
            else:
                ctf = jnp.ones((idx.shape[0], box, box), jnp.float32)

            if deformed:
                # Windowed BEFORE the Jacobian, so the determinant describes the deformation
                # that is actually applied rather than one that is then silently truncated.
                disp = apply_knn_field(field, knn_idx, knn_w)
                jac = (_field_jacobian(disp, nb_minus, nb_plus, nb_inv) if jacobian else None)
                images_ctf = _ctf_correlate(imgs, ctf) if has_ctf else imgs
                # Mean in-plane size of the correction actually applied, for the same
                # readout the image mode gives -- measured on the projected displacement.
                p_can = jnp.einsum("bij,nj->bni", rotations, mask_points_dev)
                p_def = jnp.einsum("bij,bnj->bni", rotations, mask_points_dev[None] + disp)
                # Weighted by the same window the correction carries, so the number reports
                # the shift that was APPLIED rather than the raw field -- most of which lives
                # in empty space the window suppresses.
                mag = jnp.linalg.norm((p_def - p_can)[..., :2], axis=-1)
                shift_sum += float(jnp.sum(mag * window_dev) / jnp.sum(
                    jnp.broadcast_to(window_dev, mag.shape))) * idx.shape[0]
                shift_n += idx.shape[0]

            # Same parity split as the consensus, so the two FSC curves are comparable.
            for half in (0, 1):
                pos = np.flatnonzero(idx % 2 == half)
                if pos.size == 0:
                    continue
                p = jnp.asarray(pos)
                if deformed:
                    # The straight part goes through the ordinary slice insertion; only the
                    # difference the deformation makes is accumulated in real space.
                    num[half], den[half] = _insert_slices(
                        num[half], den[half], imgs[p], rotations[p], shifts[p], ctf[p],
                        k_rot, box, f0, f1)
                    acc[half] = acc[half] + _deformed_backproject(
                        images_ctf[p], mask_points_dev, disp[p], rotations[p], shifts[p],
                        box, None if jac is None else jac[p], window_dev)
                else:
                    num[half], den[half] = _insert_slices(
                        num[half], den[half], corrected[p], rotations[p], shifts[p], ctf[p],
                        k_rot, box, f0, f1)

    applied_shift = shift_sum / max(shift_n, 1)
    if not quiet:
        # A correction much smaller than a resolution element cannot sharpen anything, so
        # say so here rather than let a null result be read off a map comparison later.
        # The threshold is a quarter of a pixel: at that size the warp is below the
        # interpolation error of the resampling that applies it.
        note = ""
        if applied_shift < 0.25:
            note = (f"\n{bcolors.WARNING}  This is far below one pixel, so the corrected map "
                    f"will be indistinguishable from the uncorrected one. What limits this is "
                    f"the model's deformation, not the reconstruction -- check 'Decoder "
                    f"deformation (A)' on the run that trained it.{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Applied motion correction: {applied_shift:.3f} px "
              f"({applied_shift * sr:.2f} A) mean in-plane shift over the modelled area."
              f"{bcolors.ENDC}{note}")

    num[0], den[0] = _hermitian_symmetrize(num[0], den[0])
    num[1], den[1] = _hermitian_symmetrize(num[1], den[1])

    if deformed:
        # Add the real-space correction. It needs no Friedel completion of its own -- the
        # transform of a real volume is Hermitian by construction -- so it is added after
        # the slice accumulators have been symmetrised, not before.
        # ...in the SAME UNITS as the slice numerator it is added to. A real-space
        # backprojection deposits its image along the whole ray, so its transform is the
        # central slice times the ray length: FT(broadcast of I along z) = box * I_hat(kx,ky)
        # at kz = 0. The slice insertion places I_hat with weight 1. Measured ratios of
        # 16.4 / 33.4 / 68.4 at box 16 / 32 / 64 confirm the factor is box. Without this the
        # correction enters ~box times oversized, swamps the numerator, and the gray-scale
        # calibration then quietly rescales the whole map to compensate (x0.044 instead of
        # x1.237 on the ribosome) -- a map that looks plausible and is entirely wrong.
        def to_ft(a):
            vol = np.zeros(box ** 3, np.float32)
            vol[mask_idx] = np.asarray(a, np.float32)
            return jnp.asarray(np.fft.fftshift(np.fft.fftn(
                np.fft.ifftshift(vol.reshape(box, box, box))))) / box
        num = [num[0] + to_ft(acc[0]), num[1] + to_ft(acc[1])]

    half_a = _invert(num[0], den[0], tau, box)
    half_b = _invert(num[1], den[1], tau, box)
    volume = _invert(num[0] + num[1], den[0] + den[1], tau, box)

    shells = _shell_index(box)
    n_shells = box // 2 + 1
    fsc = _half_map_fsc(half_a, half_b, shells, n_shells)
    resolution = _resolution(fsc, box, sr)

    if denoise:
        curve = _fsc_filter(fsc)
        v_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(volume)))
        volume = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(v_ft * curve[shells]))))
        if not quiet:
            print(f"{bcolors.OKGREEN}Half-map FSC = 0.143 at {resolution:.1f} A "
                  f"(Nyquist {2.0 * sr:.1f} A); the map is filtered to that limit.{bcolors.ENDC}")

    if calibrate_gray_scale:
        scale = _gray_scale(volume, reader, columns, sr, box, has_ctf, k_rot, f0, f1, s, a)
        volume = volume * scale
        half_a, half_b = half_a * scale, half_b * scale
        if not quiet:
            print(f"{bcolors.OKGREEN}Gray-scale calibrated to the input images (x{scale:.3f})."
                  f"{bcolors.ENDC}")

    extras = {"half_a": np.asarray(half_a, np.float32), "half_b": np.asarray(half_b, np.float32),
              "fsc": fsc, "resolution": resolution, "applied_shift_px": applied_shift}
    return np.asarray(volume, np.float32), extras


def report_half_map_resolution(vol_a, vol_b, sr, label="map", reference=None, mask=None,
                               threshold=0.143):
    """Print (and return) the half-map FSC resolution of a pair of independent half maps.

    ``reference``, when given, is a map the result should be compared against -- typically the
    consensus the reconstruction started from. Its own resolution is *not* recomputed here
    (that needs its half maps); what is reported instead is the shell correlation between the
    combined map and the reference, which is what says whether the two differ at all and
    where. Without a readout like this a reconstruction cannot be told from its own input.

    Returns ``(fsc, resolution)``.
    """
    vol_a = np.asarray(vol_a, np.float32)
    vol_b = np.asarray(vol_b, np.float32)
    if mask is not None:
        m = np.asarray(mask, np.float32)
        vol_a, vol_b = vol_a * m, vol_b * m

    box = vol_a.shape[0]
    shells = _shell_index(box)
    n_shells = box // 2 + 1
    fsc = _half_map_fsc(vol_a, vol_b, shells, n_shells)
    resolution = _resolution(fsc, box, sr, threshold)

    nyquist = 2.0 * sr
    note = ""
    if resolution <= nyquist * 1.001:
        note = (f"  {bcolors.WARNING}(this is the Nyquist limit of the box -- the sampling, not "
                f"the data, is what stops it here; re-extract finer to measure past it)"
                f"{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}{label} half-map FSC = {threshold} at {resolution:.1f} A "
          f"(Nyquist {nyquist:.1f} A).{bcolors.ENDC}{note}")

    if reference is not None:
        ref = np.asarray(reference, np.float32)
        if mask is not None:
            ref = ref * np.asarray(mask, np.float32)
        combined = 0.5 * (vol_a + vol_b)
        cross = _half_map_fsc(combined, ref, shells, n_shells)
        # State the similarity itself, not the shell where it crosses a threshold. Reporting
        # only the crossing is actively misleading in the case that matters most: when the
        # two maps are IDENTICAL the correlation never falls to 0.5, the crossing is pinned
        # at Nyquist, and "shell correlation 0.5 at <Nyquist>" then reads as poor agreement
        # when it means the exact opposite. The mean correlation cannot be misread that way.
        # Shell k holds resolution box*sr/k, so the measured band runs out to k = box*sr/res.
        k_max = int(min(n_shells - 1, max(2, round(box * sr / max(resolution, 1e-6)))))
        band = cross[1:k_max + 1]
        mean_corr = float(np.mean(band)) if band.size else float("nan")
        agree = _resolution(cross, box, sr, 0.5)
        if mean_corr > 0.99:
            verdict = (f"{bcolors.WARNING}  The two maps are the same to within 1% -- this run "
                       f"changed nothing.{bcolors.ENDC}")
        elif mean_corr > 0.95:
            verdict = f"{bcolors.WARNING}  Only a marginal difference.{bcolors.ENDC}"
        else:
            verdict = f"  They diverge below {agree:.1f} A, which is where the change is."
        print(f"{bcolors.OKGREEN}{label} vs reference: mean shell correlation {mean_corr:.4f} "
              f"to {resolution:.1f} A (1.0 = identical).{bcolors.ENDC}\n{verdict}")

    return fsc, resolution


def consensus_mask(volume, threshold=0.02, dilate=2, keep_largest=True):
    """A binary mask of the protein region of a reconstructed consensus volume.

    ``threshold`` is a fraction of the volume's maximum, applied after a light blur so the
    mask is connected rather than speckled; ``dilate`` grows it by that many voxels so the
    deformation has somewhere to move into.

    ``keep_largest`` keeps only the single largest connected region. A gridding
    reconstruction leaves faint fragments of mass away from the molecule -- most visibly in
    the corners and along the edges of the box, where the CTF power is low and the
    interpolation is least constrained. They sit above the intensity threshold, so thresholding
    alone cannot remove them; but they are *disconnected* from the protein, which is one
    contiguous blob, so a connected-component filter can. This runs before the dilation, so
    growing the mask cannot reconnect a fragment that was just discarded.
    """
    from scipy.ndimage import gaussian_filter, binary_dilation, label

    smooth = gaussian_filter(np.asarray(volume, np.float32), 1.5)
    mask = smooth > threshold * smooth.max()

    if keep_largest and mask.any():
        # 6-connectivity (faces only): a fragment touching the protein at a single corner is
        # not really attached, so it should not keep the fragment alive.
        labels, n_components = label(mask)
        if n_components > 1:
            counts = np.bincount(labels.ravel())
            counts[0] = 0                       # background label
            mask = labels == counts.argmax()

    if dilate and dilate > 0:
        mask = binary_dilation(mask, iterations=int(dilate))
    return mask.astype(np.float32)
