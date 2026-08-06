import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import mrcfile
from tqdm import tqdm
from functools import partial
from jax import vmap, jit, lax
from skimage.transform import rescale
from xmipp_metadata.image_handler import ImageHandler

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
EPS = 1e-6

def generate_uniform_rotations(num_rots=4000):
    num_s2 = int(jnp.round(num_rots ** (2 / 3)))
    num_s1 = int(jnp.round(num_rots ** (1 / 3)))

    indices_s2 = jnp.arange(num_s2, dtype=jnp.float32)
    indices_s1 = jnp.arange(num_s1, dtype=jnp.float32)

    phi_golden = jnp.pi * (3.0 - jnp.sqrt(5.0))
    z = 1.0 - (indices_s2 / float(num_s2 - 1)) * 2.0
    radius_s2 = jnp.sqrt(1.0 - z * z)
    theta = phi_golden * indices_s2

    x = radius_s2 * jnp.cos(theta)
    y = radius_s2 * jnp.sin(theta)

    psi = 2.0 * jnp.pi * indices_s1 / float(num_s1)
    x_grid, psi_grid = jnp.meshgrid(x, psi, indexing='ij')
    y_grid, _ = jnp.meshgrid(y, psi, indexing='ij')
    z_grid, _ = jnp.meshgrid(z, psi, indexing='ij')

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()
    psi_flat = psi_grid.flatten()

    sin_half_theta = jnp.sqrt(jnp.maximum((1.0 - z_flat) / 2.0, 0.0))
    cos_half_theta = jnp.sqrt(jnp.maximum((1.0 + z_flat) / 2.0, 0.0))

    phi_angle = jnp.arctan2(y_flat, x_flat)

    qw = cos_half_theta * jnp.cos(psi_flat / 2.0)
    qz = cos_half_theta * jnp.sin(psi_flat / 2.0)
    qx = sin_half_theta * jnp.cos(phi_angle + psi_flat / 2.0)
    qy = sin_half_theta * jnp.sin(phi_angle + psi_flat / 2.0)

    quats = jnp.stack([qw, qx, qy, qz], axis=1)

    rots = jax.vmap(quaternion_to_matrix)(quats)
    rots = rots.at[0].set(jnp.eye(3))

    return rots

def save_mrc(data, apix, path, centering=False):
    data_array = np.array(data, dtype=np.float32)
    nx, ny, nz = data_array.shape
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data_array)
        mrc.voxel_size = apix

        if not centering:
            mrc.header.origin.x = -(nx / 2.0) * apix
            mrc.header.origin.y = -(ny / 2.0) * apix
            mrc.header.origin.z = -(nz / 2.0) * apix


@jit
def quaternion_to_matrix(q):
    q = q / (jnp.linalg.norm(q) + EPS)
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ], dtype=jnp.float32)

@jit
def matrix_to_quaternion(R):
    tr = jnp.trace(R)

    def trace_positive():
        S = jnp.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
        return jnp.array([w, x, y, z])

    def branch_negative():
        def cond_1():
            S = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
            return jnp.array([w, x, y, z])

        def cond_2():
            S = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
            return jnp.array([w, x, y, z])

        def cond_3():
            S = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
            return jnp.array([w, x, y, z])

        return lax.cond((R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]), cond_1,
                        lambda: lax.cond(R[1, 1] > R[2, 2], cond_2, cond_3))

    return lax.cond(tr > 0, trace_positive, branch_negative)

@partial(jit, static_argnames=("apix",))
def precompute_target_gradients(vol, apix):
    nx, ny, nz = vol.shape
    fft_t = jnp.fft.rfftn(vol)

    freq_x = jnp.fft.fftfreq(nx, d=apix)
    freq_y = jnp.fft.fftfreq(ny, d=apix)
    freq_z = jnp.fft.rfftfreq(nz, d=apix)
    kx, ky, kz = jnp.meshgrid(freq_x, freq_y, freq_z, indexing='ij')

    fft_t_dx = 1j * kx * fft_t
    fft_t_dy = 1j * ky * fft_t
    fft_t_dz = 1j * kz * fft_t

    return fft_t_dx, fft_t_dy, fft_t_dz

@partial(jit, static_argnames=("grid_shape", "apix_s",))
def vectorial_rots_shifts(probe, fft_t_dx, fft_t_dy, fft_t_dz, grid_shape, apix_s):
    nx, ny, nz = grid_shape
    freq_x = jnp.fft.fftfreq(nx, d=apix_s)
    freq_y = jnp.fft.fftfreq(ny, d=apix_s)
    freq_z = jnp.fft.rfftfreq(nz, d=apix_s)
    kx, ky, kz = jnp.meshgrid(freq_x, freq_y, freq_z, indexing='ij')

    fft_probe = jnp.fft.rfftn(probe)

    fft_p_dx = 1j * kx * fft_probe
    fft_p_dy = 1j * ky * fft_probe
    fft_p_dz = 1j * kz * fft_probe

    cc_x = fft_t_dx * jnp.conj(fft_p_dx)
    cc_y = fft_t_dy * jnp.conj(fft_p_dy)
    cc_z = fft_t_dz * jnp.conj(fft_p_dz)

    cc_grid = jnp.fft.irfftn(cc_x + cc_y + cc_z, s=grid_shape)

    flat_idx = jnp.argmax(cc_grid)
    max_cc = cc_grid.flatten()[flat_idx]

    x, y, z = jnp.unravel_index(flat_idx, grid_shape)
    dx = jnp.where(x > nx // 2, x - nx, x)
    dy = jnp.where(y > ny // 2, y - ny, y)
    dz = jnp.where(z > nz // 2, z - nz, z)

    shift_ang = jnp.array([dx, dy, dz]) * apix_s
    return max_cc, shift_ang


class SparseGaussianRasterizer(eqx.Module):
    grid_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)
    sigma: jnp.ndarray
    mesh: jnp.ndarray

    def __init__(self, grid_shape, voxel_size, sigma_vec, kernel_width=7):
        self.grid_shape = grid_shape
        self.voxel_size = voxel_size
        self.sigma = jnp.maximum(sigma_vec, voxel_size * 0.5)

        r = jnp.arange(-kernel_width // 2, kernel_width // 2 + 1)
        self.mesh = jnp.meshgrid(r, r, r, indexing='ij')

    @eqx.filter_jit
    def __call__(self, coords, weights=1.0):
        coords_vox = coords / self.voxel_size
        coords_vox_int = jax.lax.stop_gradient(jnp.round(coords_vox).astype(jnp.int32))

        target_x = coords_vox_int[:, 0:1] + self.mesh[0].flatten()[None, :]
        target_y = coords_vox_int[:, 1:2] + self.mesh[1].flatten()[None, :]
        target_z = coords_vox_int[:, 2:3] + self.mesh[2].flatten()[None, :]

        diff_x = target_x.astype(jnp.float32) - coords_vox[:, 0:1]
        diff_y = target_y.astype(jnp.float32) - coords_vox[:, 1:2]
        diff_z = target_z.astype(jnp.float32) - coords_vox[:, 2:3]
        dist_sq = diff_x ** 2 + diff_y ** 2 + diff_z ** 2

        var_vox = (self.sigma / self.voxel_size) ** 2 + (1/12)
        pw = weights[:, None] if hasattr(weights, 'ndim') and weights.ndim == 1 else weights

        amplitude = 1.0 / ((2.0 * jnp.pi * var_vox) ** 1.5 + EPS)
        densities = amplitude * jnp.exp(-dist_sq / (2.0 * var_vox + EPS)) * pw

        nx, ny, nz = self.grid_shape
        out_of_bounds = (target_x < 0) | (target_x >= nx) | \
                        (target_y < 0) | (target_y >= ny) | \
                        (target_z < 0) | (target_z >= nz) | \
                        (densities < 1e-4)

        target_idx_flat = target_x * (ny * nz) + target_y * nz + target_z

        safe_densities = jnp.where(out_of_bounds, 0.0, densities)
        safe_idx = (target_idx_flat % (nx * ny * nz)).astype(jnp.int32)

        grid_flat = jnp.zeros(nx * ny * nz, dtype=jnp.float32)
        grid_flat = grid_flat.at[safe_idx.reshape(-1)].add(safe_densities.reshape(-1))
        return grid_flat.reshape(self.grid_shape)

class ProteinTopology:
    # Electronic Scattering Approximation for Kirkland Cross-Section.
    electron_scattering = {
        'H': 0.52, 'C': 2.50, 'N': 2.85, 'O': 3.20,
        'P': 5.80, 'S': 6.10, 'SE': 10.2,
        'MG': 4.30, 'CA': 7.50, 'ZN': 11.0, 'FE': 9.20,
        'NA': 4.00, 'K': 6.80, 'CL': 6.50, 'I': 18.5
    }
    def __init__(self, pdb_path):
        self.pdb_path = pdb_path
        if self.pdb_path.lower().endswith('.cif'):
            all_atoms, weights_list = self._parse_cif(self.pdb_path)
        else:
            all_atoms, weights_list = self._parse_pdb(self.pdb_path)

        self.atom_weights = jnp.array(weights_list, dtype=jnp.float32)
        self.com = jnp.sum(jnp.array(all_atoms) * self.atom_weights[:, None], axis=0) / (jnp.sum(self.atom_weights) + EPS)
        self.all_coords = jnp.array(jnp.array(all_atoms) - self.com)

    def _parse_pdb(self, pdb_path):
        all_atoms, weights = [], []
        protein_ca_coords = []

        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    is_hetatm = line.startswith("HETATM")
                    try:
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        coord = np.array([x, y, z])

                        atom_name = line[12:16].strip()
                        elem = line[76:78].strip().upper()
                        if not elem:
                            elem = atom_name[0]

                        weight = self.electron_scattering.get(elem, 2.5)

                        if not is_hetatm:
                            if atom_name == 'CA' or atom_name == 'P':
                                protein_ca_coords.append(coord)
                        else:
                            if len(protein_ca_coords) > 0:
                                ca_array = np.array(protein_ca_coords)
                                dists_sq = np.sum((ca_array - coord) ** 2, axis=1)
                                if np.min(dists_sq) >= 15.0 ** 2:
                                    weight = 0.0
                            else:
                                weight = 0.0

                        all_atoms.append(coord)
                        weights.append(weight)

                    except ValueError:
                        continue
        return all_atoms, weights

    def _parse_cif(self, path):
        all_atoms, weights = [], []
        protein_ca_coords = []

        in_atom_site_header = False
        in_atom_site_data = False
        col_idx = {}

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue

                if line == "loop_":
                    in_atom_site_header = False
                    in_atom_site_data = False
                    col_idx = {}
                    continue

                if line.startswith("_atom_site."):
                    in_atom_site_header = True
                    col_name = line
                    col_idx[col_name] = len(col_idx)
                    continue

                if in_atom_site_header and not line.startswith("_"):
                    in_atom_site_header = False
                    if "_atom_site.Cartn_x" in col_idx:
                        in_atom_site_data = True
                        idx_group = col_idx.get("_atom_site.group_PDB", -1)
                        idx_atom = col_idx.get("_atom_site.label_atom_id", col_idx.get("_atom_site.auth_atom_id", -1))
                        idx_x = col_idx["_atom_site.Cartn_x"]
                        idx_y = col_idx["_atom_site.Cartn_y"]
                        idx_z = col_idx["_atom_site.Cartn_z"]
                        idx_elem = col_idx.get("_atom_site.type_symbol", -1)

                if in_atom_site_data:
                    if line.startswith("_") or line == "loop_":
                        in_atom_site_data = False
                        continue

                    parts = line.split()
                    try:
                        group_type = parts[idx_group] if idx_group != -1 else "ATOM"
                        if group_type not in ("ATOM", "HETATM"):
                            continue

                        is_hetatm = (group_type == "HETATM")
                        x, y, z = float(parts[idx_x]), float(parts[idx_y]), float(parts[idx_z])
                        coord = np.array([x, y, z])

                        atom_name = parts[idx_atom].strip('"\'') if idx_atom != -1 else "UNK"
                        elem = parts[idx_elem].upper() if idx_elem != -1 else "C"
                        elem = ''.join([c for c in elem if c.isalpha()])

                        weight = self.electron_scattering.get(elem, 2.5)

                        if not is_hetatm:
                            if atom_name == 'CA' or atom_name == 'P':
                                protein_ca_coords.append(coord)
                        else:
                            if len(protein_ca_coords) > 0:
                                ca_array = np.array(protein_ca_coords)
                                dists_sq = np.sum((ca_array - coord) ** 2, axis=1)
                                if np.min(dists_sq) >= 15.0 ** 2:
                                    weight = 0.0
                            else:
                                weight = 0.0

                        all_atoms.append(coord)
                        weights.append(weight)

                    except (IndexError, ValueError):
                        continue

        return all_atoms, weights

class RigidTransformation(eqx.Module):
    global_shift: jnp.ndarray
    global_rot_quat: jnp.ndarray

    def __init__(self, init_quat=None, init_shift=None):
        self.global_shift = init_shift if init_shift is not None else jnp.zeros(3)
        self.global_rot_quat = init_quat if init_quat is not None else jnp.array([1.0, 0.0, 0.0, 0.0])

    def __call__(self):
        g_q = self.global_rot_quat / (jnp.linalg.norm(self.global_rot_quat) + EPS)
        return g_q, self.global_shift

class RigidCoords(eqx.Module):
    all_coords: jnp.ndarray
    com: jnp.ndarray

    def __init__(self, topo: ProteinTopology):
        self.all_coords = jax.device_put(topo.all_coords)
        self.com = jax.device_put(topo.com)

    def __call__(self, global_q, global_t):
        R_global = quaternion_to_matrix(global_q)
        all_coords_final = jnp.dot(self.all_coords, R_global.T) + global_t
        return all_coords_final

class RigidEngine:
    def __init__(self, topo, vol, apix, out_dir):
        self.topo = topo
        self.apix = apix
        self.out_dir = out_dir

        self.vol_raw = vol
        self.vol_shape = self.vol_raw.shape

        self.transformation = RigidTransformation()
        self.coords = RigidCoords(topo)
        self.rasterizer = SparseGaussianRasterizer(self.vol_shape, self.apix, sigma_vec=self.apix, kernel_width=5)

    def get_transformed_coords(self, transformation):
        g_q, g_s = transformation()
        return self.coords(g_q, g_s)

    def global_grid_search(self, vol_s_orig, vol_s_flip, downsample_factor=4, n_rots=4000, batch_size=100):
        center_proj = jnp.array(self.vol_shape) * self.apix / 2.0
        apix_s = self.apix * downsample_factor

        fft_dx_o, fft_dy_o, fft_dz_o = precompute_target_gradients(vol_s_orig, apix_s)
        fft_dx_f, fft_dy_f, fft_dz_f = precompute_target_gradients(vol_s_flip, apix_s)

        raster_s = SparseGaussianRasterizer(vol_s_orig.shape, apix_s, sigma_vec=apix_s * 1.5, kernel_width=5)
        rots = generate_uniform_rotations(n_rots)

        @jit
        def process_batch(rot_batch, coords, center, weights):
            coords_rot = jnp.einsum('bij,nj->bni', rot_batch, coords) + center
            sim_vol = vmap(lambda c: raster_s(c, weights))(coords_rot)

            sim_centered = sim_vol - jnp.mean(sim_vol)
            sim_var = jnp.sum(sim_centered ** 2) + EPS
            probe = sim_centered / jnp.sqrt(sim_var)

            res_o = vmap(lambda v: vectorial_rots_shifts(v, fft_dx_o, fft_dy_o, fft_dz_o, vol_s_orig.shape, apix_s))(probe)
            res_f = vmap(lambda v: vectorial_rots_shifts(v, fft_dx_f, fft_dy_f, fft_dz_f, vol_s_orig.shape, apix_s))(probe)
            return res_o, res_f

        n_rots = rots.shape[0]
        all_cc_o = jnp.zeros(n_rots, dtype=jnp.float32)
        all_sh_o = jnp.zeros((n_rots, 3), dtype=jnp.float32)
        all_cc_f = jnp.zeros(n_rots, dtype=jnp.float32)
        all_sh_f = jnp.zeros((n_rots, 3), dtype=jnp.float32)

        n_batches = int(jnp.ceil(n_rots / batch_size))
        pbar = tqdm(range(n_batches), desc=f"Exhaustive Search")
        for i in pbar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_rots)
            b_rots = rots[i * batch_size: (i + 1) * batch_size]
            (cc_o, sh_o), (cc_f, sh_f) = process_batch(
                b_rots, self.topo.all_coords, center_proj, self.topo.atom_weights
            )
            all_cc_o = all_cc_o.at[start_idx:end_idx].set(cc_o)
            all_sh_o = all_sh_o.at[start_idx:end_idx].set(sh_o)
            all_cc_f = all_cc_f.at[start_idx:end_idx].set(cc_f)
            all_sh_f = all_sh_f.at[start_idx:end_idx].set(sh_f)

            idx_best_o = jnp.argmax(cc_o)
            idx_best_f = jnp.argmax(cc_f)
            pbar.set_postfix(CC_o=f"{float(cc_o[idx_best_o]):.3e}", CC_f=f"{float(cc_f[idx_best_f]):.3e}")

        top_k = min(10, n_rots)
        top_cc_o, top_idx_o = lax.top_k(all_cc_o, top_k)
        top_cc_f, top_idx_f = lax.top_k(all_cc_f, top_k)

        @eqx.filter_jit
        def score_pose_native(rot_m, shift_v, target_vol):
            coords = jnp.dot(self.topo.all_coords, rot_m.T) + center_proj + shift_v
            sim_vol = self.rasterizer(coords, self.topo.atom_weights)
            sim_centered = sim_vol - jnp.mean(sim_vol)
            sim_norm = sim_centered / (jnp.sqrt(jnp.sum(sim_centered ** 2)) + EPS)
            return jnp.sum(sim_norm * target_vol)

        scores_o = jax.vmap(lambda r, s: score_pose_native(r, s, vol_s_orig))(rots[top_idx_o], all_sh_o[top_idx_o])
        scores_f = jax.vmap(lambda r, s: score_pose_native(r, s, vol_s_flip))(rots[top_idx_f], all_sh_f[top_idx_f])
        best_idx_o, best_idx_f = jnp.argmax(scores_o), jnp.argmax(scores_f)

        if scores_f[best_idx_f] > scores_o[best_idx_o]:
            print(f">>> FLIPPED Volume Selected (High-Res CC: {float(scores_f[best_idx_f]):.4f})")
            self.vol_raw = jnp.flip(self.vol_raw, axis=0)
            final_rot, final_shift = rots[top_idx_f][best_idx_f], all_sh_f[top_idx_f][best_idx_f]
            self.cc_global = float(scores_f[best_idx_f])
            final_vol_target = vol_s_flip
        else:
            print(f">>> ORIGINAL Volume Selected (High-Res CC: {float(scores_o[best_idx_o]):.4f})")
            final_rot, final_shift = rots[top_idx_o][best_idx_o], all_sh_o[top_idx_o][best_idx_o]
            self.cc_global = float(scores_o[best_idx_o])
            final_vol_target = vol_s_orig

        final_global_shift = final_shift + center_proj
        self.transformation = eqx.tree_at(lambda m: m.global_rot_quat, self.transformation, matrix_to_quaternion(final_rot))
        self.transformation = eqx.tree_at(lambda m: m.global_shift, self.transformation, final_global_shift)

        final_coords_raw = self.get_transformed_coords(self.transformation)
        sim_native = self.rasterizer(final_coords_raw, self.topo.atom_weights)
        sim_centered = sim_native - jnp.mean(sim_native)
        sim_target = sim_centered / (jnp.sqrt(jnp.sum(sim_centered ** 2)) + EPS)
        native_cc = jnp.sum(sim_target * final_vol_target)
        print(f">>> Final Native CC: {native_cc:.4f}\n")

    def optimize_rigid_pose(self, epochs=1000, is_aligned=False):
        diff, static = eqx.partition(self.transformation, eqx.is_inexact_array)
        optim = optax.adamw(learning_rate=0.01)
        opt_state = optim.init(diff)

        @eqx.filter_jit
        def step(diff, static, state, vol_target, rasterizer, weights, current_sigma):
            def loss_fn(d):
                transformation = eqx.combine(d, static)
                coords_transformed = self.get_transformed_coords(transformation)

                dyn_rasterizer = eqx.tree_at(lambda r: r.sigma, rasterizer, jnp.maximum(current_sigma, self.apix * 0.5))
                sim = dyn_rasterizer(coords_transformed, weights)

                sim_centered = sim - jnp.mean(sim)
                var_sim = jnp.sqrt(jnp.sum(sim_centered ** 2)) + EPS
                sim_target = sim_centered / var_sim

                cc_mask = jnp.sum(sim_target * vol_target)
                return 1.0 - cc_mask, cc_mask

            (loss, cc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(diff)
            updates, new_state = optim.update(grads, state, diff)
            new_diff = eqx.apply_updates(diff, updates)
            return new_diff, new_state, loss, cc

        vol_centered = self.vol_raw - jnp.mean(self.vol_raw)
        vol_target = vol_centered / (jnp.sqrt(jnp.sum(vol_centered ** 2)) + EPS)

        pbar = tqdm(range(epochs), desc='Local Refinement')
        ccs = []
        for i in pbar:
            progress = i / epochs
            if not is_aligned:
                current_sigma = jnp.array(self.apix * (8.0 - 7.0 * progress))
            else:
                current_sigma = jnp.array(self.apix * 1.0)
            diff, opt_state, loss, cc = step(diff, static, opt_state, vol_target, self.rasterizer, self.topo.atom_weights, current_sigma)
            ccs.append(cc)
            pbar.set_postfix(Loss=f"{loss:.4f}", CC=f"{cc:.4f}", Sigma=f"{current_sigma:.4f}")

        self.transformation = eqx.combine(diff, static)
        print(quaternion_to_matrix(self.transformation.global_rot_quat), self.transformation.global_shift)

    def save(self, output_name, is_aligned=False):
        print(f"Saving results to {self.out_dir}...")
        all_final_coords_orig = self.get_transformed_coords(self.transformation)
        if not is_aligned:
            all_final_coords = all_final_coords_orig - jnp.array(self.vol_shape) * self.apix / 2.0
        else:
            all_final_coords = all_final_coords_orig - self.topo.com

        ext = os.path.splitext(self.topo.pdb_path)[1].lower()

        if ext == '.cif':
            out_file = os.path.join(self.out_dir, f"{output_name}_fitted.cif")
            self._write_cif(self.topo.pdb_path, out_file, np.array(all_final_coords))
        else:
            out_file = os.path.join(self.out_dir, f"{output_name}_fitted.pdb")
            self._write_pdb(self.topo.pdb_path, out_file, np.array(all_final_coords))

        sim_vol = self.rasterizer(jnp.array(all_final_coords_orig), self.topo.atom_weights)
        save_mrc(sim_vol.T, self.apix, os.path.join(self.out_dir, f"{output_name}_sim.mrc"))
        save_mrc(self.vol_raw.T, self.apix, os.path.join(self.out_dir, f"{output_name}_input_norm.mrc"), centering=False)

    def _write_pdb(self, in_path, out_path, coords):
        with open(in_path, 'r') as f_in, open(out_path, 'w') as f_out:
            atom_idx = 0
            for line in f_in:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    if atom_idx < len(coords):
                        x, y, z = coords[atom_idx]
                        f_out.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                        atom_idx += 1
                    else:
                        f_out.write(line)
                else:
                    f_out.write(line)

    def _write_cif(self, in_path, out_path, coords):
        in_atom_site_header = False
        in_atom_site_data = False
        col_idx = {}
        idx_x = idx_y = idx_z = idx_group = -1
        atom_idx = 0

        with open(in_path, 'r') as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                line_stripped = line.strip()

                if line_stripped == "loop_":
                    in_atom_site_header = False
                    in_atom_site_data = False
                    col_idx = {}
                    f_out.write(line)
                    continue

                if line_stripped.startswith("_atom_site."):
                    in_atom_site_header = True
                    col_idx[line_stripped] = len(col_idx)
                    f_out.write(line)
                    continue

                if in_atom_site_header and not line_stripped.startswith("_"):
                    in_atom_site_header = False
                    if "_atom_site.Cartn_x" in col_idx:
                        in_atom_site_data = True
                        idx_group = col_idx.get("_atom_site.group_PDB", -1)
                        idx_x = col_idx["_atom_site.Cartn_x"]
                        idx_y = col_idx["_atom_site.Cartn_y"]
                        idx_z = col_idx["_atom_site.Cartn_z"]

                if in_atom_site_data:
                    if line_stripped.startswith("_") or line_stripped == "loop_":
                        in_atom_site_data = False
                        f_out.write(line)
                        continue

                    if not line_stripped or line_stripped.startswith('#'):
                        f_out.write(line)
                        continue

                    parts = line_stripped.split()
                    if len(parts) > max(idx_x, idx_y, idx_z):
                        if idx_group == -1 or parts[idx_group] in ("ATOM", "HETATM"):
                            if atom_idx < len(coords):
                                x, y, z = coords[atom_idx]
                                parts[idx_x] = f"{x:.3f}"
                                parts[idx_y] = f"{y:.3f}"
                                parts[idx_z] = f"{z:.3f}"
                                f_out.write(" ".join(parts) + "\n")
                                atom_idx += 1
                                continue

                f_out.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--vol", type=str, required=True)
    parser.add_argument("--sr", type=float, default=1.0)
    parser.add_argument("--downfactor", type=float, default=1.0)
    parser.add_argument("--is_aligned", action="store_true")
    parser.add_argument("--out_dir", type=str, default="rigid_output")
    args, _ = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)

    topo = ProteinTopology(args.pdb)
    with mrcfile.open(args.vol, permissive=True) as mrc:
        vol = jnp.array(mrc.data.T)
        vol = jnp.array((vol - jnp.min(vol)) / (jnp.max(vol) - jnp.min(vol) + EPS))
        apix = float(mrc.voxel_size.x) if args.sr == 1.0 else args.sr

    engine = RigidEngine(topo, vol, apix, args.out_dir)

    vol_s_orig = rescale(vol, 1 / args.downfactor, anti_aliasing=True, preserve_range=True)
    vol_centered = vol_s_orig - jnp.mean(vol_s_orig)
    vol_centered = vol_centered / jnp.sqrt(jnp.sum(vol_centered ** 2) + EPS)
    vol_s_flip = jnp.flip(vol_centered, axis=0)
    engine.global_grid_search(vol_centered, vol_s_flip, downsample_factor=args.downfactor, n_rots=1000, batch_size=10)

    if not args.is_aligned:
        engine.save("final_rigid", is_aligned=args.is_aligned)
        engine.optimize_rigid_pose(epochs=1000, is_aligned=args.is_aligned)
        engine.save("final_rigid_refined", is_aligned=args.is_aligned)
    else:
        engine.save("final_rigid", is_aligned=args.is_aligned)


if __name__ == "__main__":
    main()