#!/usr/bin/env python
"""
Synthetic phantom generator for the Hax test-suite.

This module builds, *from scratch and without any external data*, a small CryoEM
dataset exhibiting **continuous + compositional conformational heterogeneity**,
together with the metadata that the Hax programs expect (per-particle angular
assignment, in-plane shifts and CTF parameters).

Design goals
------------
* The phantom is a point cloud (a set of 3D Gaussian "atoms").  A scalar latent
  ``t in [0, 1]`` deforms it:
      - **continuous** heterogeneity: one blob slides continuously along X with t.
      - **compositional** heterogeneity: a second blob's occupancy (weight) grows
        with t, i.e. it "appears" for part of the population.
* Every particle gets a *different* random pose (Xmipp Euler angles), a *different*
  in-plane shift and a *different* CTF (defocus / astigmatism).
* Projections are generated with the very same geometric convention Hax uses
  internally (``euler_matrix_batch`` + the PhysDecoder splatting convention), and
  the CTF is applied with Hax's own ``computeCTF`` / ``ctfFilter`` so the data is
  physically consistent with what the networks model.
* ``apply_ctf=False`` produces **CTF-free** images (used to exercise the ``None``
  CTF mode), while ``apply_ctf=True`` corrupts the projections with the CTF (used
  by the ``apply`` / ``wiener`` / ``precorrect`` modes).

Nothing here imports or modifies any Hax *program* module: it only reuses the
public numeric helpers (CTF, Euler, splatting), exactly as a user-written script
would.
"""

import os
from functools import partial
import numpy as np

import jax
from jax import numpy as jnp
import dm_pix

from hax.utils import computeCTF, ctfFilter, euler_matrix_batch

from xmipp_metadata.image_handler import ImageHandler
from xmipp_metadata.metadata import XmippMetaData


# --------------------------------------------------------------------------- #
#  Conformation model (the "ground-truth" heterogeneity)                       #
# --------------------------------------------------------------------------- #
# Characteristic radius of the molecule, as a fraction of the box.  Large enough
# that the auto-generated protein mask (snake+Otsu, used by fit_volume /
# local_reconstruction with the default n_init=2500) comfortably exceeds 2500
# voxels, while still leaving margin to the box edge after arbitrary rotations.
_RADIUS_FRACTION = 0.22


def _base_atoms(box):
    """Build the static part of the phantom: an elongated body made of blobs.

    Coordinates are expressed in **voxel units centred at the origin** (range
    roughly ``[-box/2, box/2]``), matching the centred coordinates used by the
    Hax PhysDecoder.
    """
    r = _RADIUS_FRACTION * box  # characteristic radius of the molecule
    coords = []
    weights = []

    # Central elongated body (a string of blobs along Z)
    for z in np.linspace(-r, r, 9):
        coords.append([0.0, 0.0, z])
        weights.append(1.0)

    # Lateral lobes (static) to give the projections structure / body
    for y in (-0.6 * r, 0.6 * r):
        for x in (-0.4 * r, 0.4 * r):
            coords.append([x, y, 0.0])
            weights.append(0.9)

    return np.asarray(coords, dtype=np.float32), np.asarray(weights, dtype=np.float32)


def conformation(t, box, n_jitter=60, seed=0, compositional=True):
    """Return ``(coords, weights)`` of the phantom for latent ``t in [0, 1]``.

    * The *mobile* blob slides along +X as ``t`` grows   -> continuous motion
      (a mass-conserving deformation).
    * When ``compositional=True`` a second lobe's occupancy ramps with ``t``
      -> compositional change (mass appears/disappears).  Set it to ``False`` for
      methods that model *only* continuous deformation (e.g. Zernike3Deep): the
      lobe then has a constant occupancy and the sole source of variability is
      the continuous motion.

    A fixed cloud of small jitter atoms is added so that the resulting volume is
    not perfectly sparse (closer to a real density map).
    """
    r = _RADIUS_FRACTION * box
    base_c, base_w = _base_atoms(box)

    # Continuous heterogeneity: mobile blob translating along X.
    mobile_c = np.array([[-r + 2.0 * r * t, 0.0, 0.0]], dtype=np.float32)
    mobile_w = np.array([1.2], dtype=np.float32)

    # Second lobe: occupancy ramps with t (compositional) or is constant (motion-only).
    comp_c = np.array([[0.0, 0.0, 0.9 * r]], dtype=np.float32)
    comp_w = np.array([1.4 * float(t) if compositional else 0.7], dtype=np.float32)

    # Deterministic jitter cloud (same atoms for every particle, only pose differs)
    rng = np.random.default_rng(seed)
    jit_c = (rng.standard_normal((n_jitter, 3)).astype(np.float32)) * (0.35 * r)
    jit_w = 0.3 * np.ones((n_jitter,), dtype=np.float32)

    coords = np.concatenate([base_c, mobile_c, comp_c, jit_c], axis=0)
    weights = np.concatenate([base_w, mobile_w, comp_w, jit_w], axis=0)
    return coords, weights


# --------------------------------------------------------------------------- #
#  Forward projection (Hax geometric convention) + CTF                         #
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("box",))
def _project_batch(coords_b, weights_b, angles, shifts, box):
    """Project a batch of point clouds to 2D images.

    Replicates the PhysDecoder convention:
        rotated = coords @ R^T ; (x, y) kept ; row<-y, col<-x ; shift subtracted.
    ``box`` is a static (Python int) argument so it can size the output grid.
    """
    R = euler_matrix_batch(angles[:, 0], angles[:, 1], angles[:, 2])     # (B,3,3)
    p = jnp.einsum("bnj,bij->bni", coords_b, R)                          # coords @ R^T

    half = 0.5 * box
    col = p[..., 0] - shifts[:, None, 0] + half
    row = p[..., 1] - shifts[:, None, 1] + half

    ic = jnp.clip(jnp.round(col).astype(jnp.int32), 0, box - 1)
    ir = jnp.clip(jnp.round(row).astype(jnp.int32), 0, box - 1)

    def scatter_one(ir_i, ic_i, w_i):
        img = jnp.zeros((box, box), dtype=jnp.float32)
        return img.at[ir_i, ic_i].add(w_i)

    images = jax.vmap(scatter_one)(ir, ic, weights_b)

    # Smooth the splatted points into blobs (gaussian, sigma ~ 1.2 px)
    images = dm_pix.gaussian_blur(images[..., None], sigma=1.2, kernel_size=9)[..., 0]
    return images


def _sample_ctf_params(n, rng):
    """Return realistic random per-particle CTF parameters."""
    defocusU = rng.uniform(6000.0, 22000.0, size=n).astype(np.float32)      # Angstrom
    astig = rng.uniform(-800.0, 800.0, size=n).astype(np.float32)
    defocusV = (defocusU + astig).astype(np.float32)
    defocusAngle = rng.uniform(0.0, 180.0, size=n).astype(np.float32)
    cs = np.full(n, 2.7, dtype=np.float32)                                  # mm
    kv = np.full(n, 300.0, dtype=np.float32)                                # kV
    return defocusU, defocusV, defocusAngle, cs, kv


# --------------------------------------------------------------------------- #
#  Dataset builder                                                             #
# --------------------------------------------------------------------------- #
def generate_dataset(n_particles=256, box=48, sr=2.0, apply_ctf=True, noise=0.03,
                     n_atom_jitter=60, seed=0, compositional=True):
    """Generate images + per-particle metadata fields.

    Returns a dict with image stack (N,box,box) and all metadata arrays, plus the
    ground-truth latent ``t`` and a reference (consensus) volume / mask.

    ``compositional=False`` yields *continuous-only* heterogeneity (motion without
    mass change), as required by deformation-only methods such as Zernike3Deep.
    """
    rng = np.random.default_rng(seed)

    # Ground-truth latent: continuous in [0, 1].
    t = rng.uniform(0.0, 1.0, size=n_particles).astype(np.float32)

    # Per-particle conformations
    coords_list, weights_list = [], []
    for ti in t:
        c, w = conformation(float(ti), box, n_jitter=n_atom_jitter, seed=seed,
                            compositional=compositional)
        coords_list.append(c)
        weights_list.append(w)
    coords_b = jnp.asarray(np.stack(coords_list, axis=0))      # (N, Natoms, 3)
    weights_b = jnp.asarray(np.stack(weights_list, axis=0))    # (N, Natoms)

    # Random poses (Xmipp Euler angles, degrees) and in-plane shifts (px)
    angles = np.stack([
        rng.uniform(0.0, 360.0, size=n_particles),
        rng.uniform(0.0, 180.0, size=n_particles),
        rng.uniform(0.0, 360.0, size=n_particles),
    ], axis=1).astype(np.float32)
    shifts = rng.uniform(-0.08 * box, 0.08 * box, size=(n_particles, 2)).astype(np.float32)

    # CTF parameters (always defined in the metadata; only *applied* to the images
    # when apply_ctf=True so that the "None" mode truly sees CTF-free images).
    defocusU, defocusV, defocusAngle, cs, kv = _sample_ctf_params(n_particles, rng)

    # ---- Forward project (in chunks to bound GPU memory) ----
    images = []
    chunk = 64
    for s in range(0, n_particles, chunk):
        e = min(s + chunk, n_particles)
        clean = _project_batch(coords_b[s:e], weights_b[s:e],
                               jnp.asarray(angles[s:e]), jnp.asarray(shifts[s:e]), box=int(box))
        if apply_ctf:
            ctf = computeCTF(jnp.asarray(defocusU[s:e]), jnp.asarray(defocusV[s:e]),
                             jnp.asarray(defocusAngle[s:e]), jnp.asarray(cs[s:e]),
                             float(kv[s]), sr, [2 * box, int(2 * 0.5 * box + 1)],
                             e - s, True)
            clean = ctfFilter(clean, ctf, pad_factor=2)
        images.append(np.asarray(clean))
    images = np.concatenate(images, axis=0).astype(np.float32)

    # Per-image normalisation + additive Gaussian noise
    images = (images - images.mean(axis=(1, 2), keepdims=True))
    std = images.std(axis=(1, 2), keepdims=True)
    images = images / np.where(std > 0, std, 1.0)
    images = images + noise * rng.standard_normal(images.shape).astype(np.float32)

    # ---- Reference (consensus) volume + mask from the mean conformation ----
    # Rendered as a *filled* density (analytic Gaussian sum) rather than a sparse
    # point splat, so the derived mask covers a sizeable, connected protein-like
    # region (needed e.g. by fit_volume / local_reconstruction which sample
    # thousands of points inside the mask).
    c_ref, w_ref = conformation(0.5, box, n_jitter=n_atom_jitter, seed=seed,
                                compositional=compositional)
    ref_vol = _render_volume(c_ref, w_ref, box, sigma=0.08 * box)
    mask = (ref_vol > 0.04 * ref_vol.max()).astype(np.float32)

    return {
        "images": images,
        "angles": angles,
        "shifts": shifts,
        "ctfDefocusU": defocusU,
        "ctfDefocusV": defocusV,
        "ctfDefocusAngle": defocusAngle,
        "ctfSphericalAberration": cs,
        "ctfVoltage": kv,
        "t": t,
        "ref_vol": ref_vol.astype(np.float32),
        "mask": mask,
        "box": box,
        "sr": sr,
        "apply_ctf": apply_ctf,
    }


def _render_volume(coords, weights, box, sigma):
    """Render a *filled* 3D density as an analytic sum of Gaussians.

    ``coords`` are centred voxel coordinates (range ~[-box/2, box/2]); the volume
    is indexed ``[z, y, x]``.  Looping over the (few) atoms keeps memory low.
    """
    ax = np.arange(box, dtype=np.float32) - 0.5 * box           # centred axis
    Z, Y, X = np.meshgrid(ax, ax, ax, indexing="ij")
    vol = np.zeros((box, box, box), dtype=np.float32)
    inv = 1.0 / (2.0 * sigma * sigma)
    for (cx, cy, cz), w in zip(coords, weights):
        vol += w * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) * inv)
    return vol


def write_dataset(out_dir, data, prefix="particles"):
    """Write the dataset to disk as an .mrcs stack + .xmd metadata.

    Returns a dict of paths: ``md`` (.xmd), ``vol`` (.mrc), ``mask`` (.mrc).
    """
    os.makedirs(out_dir, exist_ok=True)
    stack_path = os.path.join(out_dir, prefix + ".mrcs")
    md_path = os.path.join(out_dir, prefix + ".xmd")
    vol_path = os.path.join(out_dir, prefix + "_reference.mrc")
    mask_path = os.path.join(out_dir, prefix + "_mask.mrc")

    # Image stack
    ImageHandler().write(data["images"], stack_path, overwrite=True, sr=data["sr"])

    # Metadata built straight from the stack, injecting angles/shifts/CTF columns
    md = XmippMetaData(
        stack_path,
        angles=data["angles"],
        shifts=data["shifts"],
        ctfDefocusU=data["ctfDefocusU"],
        ctfDefocusV=data["ctfDefocusV"],
        ctfDefocusAngle=data["ctfDefocusAngle"],
        ctfSphericalAberration=data["ctfSphericalAberration"],
        ctfVoltage=data["ctfVoltage"],
    )
    md.write(md_path)

    # Reference volume + mask
    ImageHandler().write(data["ref_vol"], vol_path, overwrite=True, sr=data["sr"])
    ImageHandler().write(data["mask"], mask_path, overwrite=True, sr=data["sr"])

    return {"md": md_path, "vol": vol_path, "mask": mask_path, "stack": stack_path}
