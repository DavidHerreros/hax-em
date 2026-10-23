#!/usr/bin/env python
"""
MoDART symmetry check, run as a subprocess scenario of ``test_modart.py``.

The CLI scenarios only see that ``--symmetry_group`` writes a map. This script checks
the geometry itself with MoDART's own functions, for c2, c3 and d2:

* poses: for a map with the group's symmetry, every symmetry copy of a pose must give
  the same projection as the pose itself;
* motion: with a non-symmetric field on the model points, copy g must send voxel u to
  R (S_g u + f(S_g u)), the field being carried from the points as in a c1 run.
"""

import sys

import numpy as np
import jax.numpy as jnp

from hax.programs.modart import (batch_geometry, knn_field_chunk, project_maps, splat_deapodization,
                                 symmetry_copy_fields, symmetry_field_operator, warped_detector_coords)
from hax.utils import build_knn_field_operator
from hax.utils.symmetry_groups import symmetry_matrices

GREEN, RED, END = "\033[92m", "\033[91m", "\033[0m"
BOX, B = 24, 4

rng = np.random.default_rng(0)
angles = np.stack([rng.uniform(0, 360, B), rng.uniform(20, 160, B), rng.uniform(0, 360, B)], 1).astype(np.float32)
md = {"euler_angles": jnp.asarray(angles), "shifts": jnp.zeros((B, 2), jnp.float32)}
labels, eye, zeros = jnp.arange(B), jnp.tile(jnp.eye(3, dtype=jnp.float32), (B, 1, 1)), jnp.zeros((B, 2), jnp.float32)
vox = (np.argwhere(np.ones((BOX,) * 3))[:, ::-1] - 0.5 * BOX).astype(np.float32)
poses, pose_shifts, _ = batch_geometry(md, labels, eye, zeros, jnp.eye(3)[None], 1.0, BOX, "None")

failed = False
for group in ("c2", "c3", "d2"):
    sym = jnp.asarray(symmetry_matrices(group), jnp.float32)
    G = sym.shape[0]
    rotations, shifts, _ = batch_geometry(md, labels, eye, zeros, sym, 1.0, BOX, "None")

    # Poses: a symmetric map (blobs at symmetry-related positions) projects identically from every copy
    centres = rng.uniform(-6, 6, (12, 3)).astype(np.float32)
    centres = np.concatenate([centres @ np.asarray(s).T for s in sym])
    volume = np.exp(-0.5 * np.sum((vox[:, None] - centres[None]) ** 2, -1)).sum(1)
    no_field = jnp.zeros((vox.shape[0], 1), jnp.int32), jnp.zeros((vox.shape[0], 1))
    proj = np.asarray(project_maps(jnp.asarray(volume[None], jnp.float32), jnp.zeros(G * B, jnp.int32), jnp.asarray(vox), *no_field,
                                   None, rotations, shifts, None, splat_deapodization(BOX), BOX, vox.shape[0], "None")).reshape(G, B, -1)
    cc = min(np.corrcoef(proj[0, b], proj[g, b])[0, 1] for g in range(1, G) for b in range(B))

    # Motion: copy g must land voxel u at R (S_g u + f(S_g u))
    points = rng.uniform(-10, 10, (300, 3)).astype(np.float32)
    field = jnp.asarray(np.sin(points[None, :, [1, 2, 0]] / 3.0 + rng.uniform(0, 6, (B, 1, 3))), jnp.float32)
    knn_idx, knn_w = symmetry_field_operator([points], vox, sym)
    got = warped_detector_coords(jnp.asarray(vox), knn_field_chunk(symmetry_copy_fields(field, sym), knn_idx, knn_w),
                                 rotations, shifts, BOX)
    err = 0.0
    for g in range(G):
        su = vox @ np.asarray(sym[g]).T
        moved = su[None] + knn_field_chunk(field, *build_knn_field_operator(points, su))
        expected = warped_detector_coords(jnp.zeros_like(jnp.asarray(vox)), moved, poses, pose_shifts, BOX)
        err = max(err, float(jnp.abs(got[g * B:(g + 1) * B] - expected).max()))

    ok = cc > 0.99 and err < 1e-3
    failed |= not ok
    print(f"  {GREEN + 'OK  ' if ok else RED + 'FAIL'}{END} {group}: projection cc between copies {cc:.5f} (> 0.99), "
          f"warped-voxel error {err:.2e} px (< 1e-3)", flush=True)

sys.exit(1 if failed else 0)
