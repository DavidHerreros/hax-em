import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import mrcfile
from tqdm import tqdm
from functools import partial
from jax import vmap, jit, lax

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/fittax_xla"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "float32")
EPS = 1e-6


def fourier_downsample(vol, downfactor):
    nx, ny, nz = vol.shape
    new_shape = (int(nx / downfactor), int(ny / downfactor), int(nz / downfactor))
    nnx, nny, nnz = new_shape

    ft = jnp.fft.fftn(vol)
    ft_shifted = jnp.fft.fftshift(ft)

    cx, cy, cz = nx // 2, ny // 2, nz // 2
    hx, hy, hz = nnx // 2, nny // 2, nnz // 2

    cropped_ft = ft_shifted[cx - hx: cx + hx + (nnx % 2),
    cy - hy: cy + hy + (nny % 2),
    cz - hz: cz + hz + (nnz % 2)]

    down_vol = jnp.real(jnp.fft.ifftn(jnp.fft.ifftshift(cropped_ft)))
    return down_vol * (np.prod(new_shape) / np.prod(vol.shape))

def generate_uniform_rotations(num_rots=4000):
    phi = jnp.sqrt(2.0)
    psi = 1.533751168755204288118041
    s = jnp.arange(num_rots, dtype=jnp.float32) + 0.5
    t = s / num_rots
    r, R = jnp.sqrt(t), jnp.sqrt(1.0 - t)
    a, b = 2.0 * jnp.pi * s / phi, 2.0 * jnp.pi * s / psi
    quats = jnp.stack([r * jnp.sin(a), r * jnp.cos(a), R * jnp.sin(b), R * jnp.cos(b)], axis=-1)
    rots = jax.vmap(quaternion_to_matrix)(quats)
    rots = rots.at[0].set(jnp.eye(3))
    return rots

def rfft_grids(shape, apix):
    nx, ny, nz = shape
    freq_x = jnp.fft.fftfreq(nx, d=apix)
    freq_y = jnp.fft.fftfreq(ny, d=apix)
    freq_z = jnp.fft.rfftfreq(nz, d=apix)
    return jnp.meshgrid(freq_x, freq_y, freq_z, indexing='ij')

def _hermitian_weights(shape):
    nx, ny, nz = shape
    w = jnp.ones((nx, ny, nz // 2 + 1), dtype=jnp.float32) * 2.0
    w = w.at[..., 0].set(1.0)
    if nz % 2 == 0:
        w = w.at[..., -1].set(1.0)
    return w

def angle_between_matrices(R1, R2):
    R1_np, R2_np = np.asarray(R1), np.asarray(R2)
    cos_theta = (np.trace(R1_np.T @ R2_np) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

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
        return jnp.array([0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S])

    def branch_negative():
        def cond_1():
            S = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            return jnp.array([(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S])

        def cond_2():
            S = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            return jnp.array([(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S])

        def cond_3():
            S = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            return jnp.array([(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S])

        return lax.cond((R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]), cond_1,
                        lambda: lax.cond(R[1, 1] > R[2, 2], cond_2, cond_3))

    return lax.cond(tr > 0, trace_positive, branch_negative)

@partial(jit, static_argnames=("apix",))
def precompute_target_vectorial(vol, apix, sigma_s):
    shape = vol.shape
    kx, ky, kz = rfft_grids(shape, apix)
    k2 = kx ** 2 + ky ** 2 + kz ** 2

    env = jnp.exp(-2.0 * jnp.pi ** 2 * k2 * sigma_s ** 2)
    W = 4.0 * jnp.pi ** 2 * k2 * env ** 2
    ft = jnp.fft.rfftn(vol)
    tf = ft * W

    tpi = 2.0j * jnp.pi
    g2 = sum(jnp.fft.irfftn(tpi * k * ft, s=shape) ** 2 for k in (kx, ky, kz))
    fgt2 = jnp.fft.rfftn(g2)

    return tf, fgt2, W

@partial(jit, static_argnames=("grid_shape", "apix_s", "top_k", "nms_win"))
def vectorial_search_batch(probe, lo, hi, tf, fgt2, W, herm, grid_shape, apix_s, top_k=10, nms_win=5):
    nx, ny, nz = grid_shape
    N = float(np.prod(grid_shape))

    fp = jnp.fft.rfftn(probe)
    e_p = jnp.sum(herm * W * jnp.abs(fp) ** 2) / N + EPS

    mask = (probe > 1e-4 * jnp.max(probe)).astype(jnp.float32)
    fm = jnp.conj(jnp.fft.rfftn(mask))

    e_t = jnp.fft.irfftn(fgt2 * fm, s=grid_shape)
    cc = jnp.fft.irfftn(tf * jnp.conj(fp), s=grid_shape)

    cc_grid = jnp.where(e_t > 1e-4 * jnp.max(e_t),
                        cc / (jnp.sqrt(jnp.maximum(e_t, 0.0) * e_p) + EPS),
                        -jnp.inf)

    sx = jnp.where(jnp.arange(nx) > nx // 2, jnp.arange(nx) - nx, jnp.arange(nx)) * apix_s
    sy = jnp.where(jnp.arange(ny) > ny // 2, jnp.arange(ny) - ny, jnp.arange(ny)) * apix_s
    sz = jnp.where(jnp.arange(nz) > nz // 2, jnp.arange(nz) - nz, jnp.arange(nz)) * apix_s
    in_x = (sx >= -lo[0]) & (sx < nx * apix_s - hi[0])
    in_y = (sy >= -lo[1]) & (sy < ny * apix_s - hi[1])
    in_z = (sz >= -lo[2]) & (sz < nz * apix_s - hi[2])
    cc_grid = jnp.where(in_x[:, None, None] & in_y[None, :, None] & in_z[None, None, :], cc_grid, -jnp.inf)

    padded = jnp.pad(cc_grid, nms_win // 2, mode="wrap")
    pooled = lax.reduce_window(padded, -jnp.inf, lax.max, (nms_win,) * 3, (1, 1, 1), "VALID")
    vals, flat = lax.top_k(jnp.where(cc_grid >= pooled, cc_grid, -jnp.inf).reshape(-1), top_k)

    x, y, z = jnp.unravel_index(flat, grid_shape)
    shift_ang = jnp.stack([
        jnp.where(x > nx // 2, x - nx, x),
        jnp.where(y > ny // 2, y - ny, y),
        jnp.where(z > nz // 2, z - nz, z)
    ], axis=-1) * apix_s

    return vals, shift_ang

@jit
def expm_so3(omega):
    th2 = jnp.sum(omega ** 2)
    th = jnp.sqrt(th2 + 1e-12)
    K = jnp.array([[0.0, -omega[2], omega[1]],
                   [omega[2], 0.0, -omega[0]],
                   [-omega[1], omega[0], 0.0]])
    return jnp.eye(3) + (jnp.sin(th) / th) * K + ((1.0 - jnp.cos(th)) / (th2 + 1e-12)) * (K @ K)

def bfgs_maximise(value, value_and_grad, first_step, max_iter=60, tol=1e-3):
    x = np.zeros(6)
    f, g = value_and_grad(x)
    H = np.eye(6)
    for it in range(max_iter):
        p = g / (np.linalg.norm(g) + 1e-30) * first_step if it == 0 else H @ g

        for step in 0.5 ** np.arange(20):
            if value(x + step * p) > f + 1e-4 * step * (g @ p):
                break
        else:
            break

        f_new, g_new = value_and_grad(x + step * p)
        s, y = step * p, g - g_new
        if s @ y > 1e-12:
            rho = 1.0 / (s @ y)
            H = (np.eye(6) - rho * np.outer(s, y)) @ H @ (np.eye(6) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, f, g = x + s, f_new, g_new
        if np.linalg.norm(s) < tol:
            break
    return x


class SparseGaussianRasterizer(eqx.Module):
    grid_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)
    sigma: jnp.ndarray
    mesh: jnp.ndarray

    def __init__(self, grid_shape, voxel_size, sigma_s, kernel_width=7):
        self.grid_shape = grid_shape
        self.voxel_size = voxel_size
        self.sigma = jnp.maximum(sigma_s, voxel_size * 0.5)
        h = kernel_width // 2
        r = jnp.arange(-h, h + 1)
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

        sqrt2_sig = (self.sigma / self.voxel_size) * jnp.sqrt(2.0)

        def phi(diff):
            return 0.5 * (jax.lax.erf((diff + 0.5) / sqrt2_sig) - jax.lax.erf((diff - 0.5) / sqrt2_sig))

        densities = phi(diff_x) * phi(diff_y) * phi(diff_z) * (
            weights[:, None] if hasattr(weights, 'ndim') else weights)
        nx, ny, nz = self.grid_shape
        out_of_bounds = (target_x < 0) | (target_x >= nx) | (target_y < 0) | (target_y >= ny) | (target_z < 0) | (
                target_z >= nz) | (densities < 1e-5)

        target_idx_flat = target_x * (ny * nz) + target_y * nz + target_z
        safe_idx = (target_idx_flat % (nx * ny * nz)).astype(jnp.int32)
        grid_flat = jnp.zeros(nx * ny * nz, dtype=jnp.float32).at[safe_idx.reshape(-1)].add(
            jnp.where(out_of_bounds, 0.0, densities).reshape(-1))
        return grid_flat.reshape(self.grid_shape)

class ProteinTopology:
    electron_scattering = {
        'H': 0.52, 'C': 2.50, 'N': 2.85, 'O': 3.20,
        'P': 5.80, 'S': 6.10, 'SE': 10.2,
        'MG': 4.30, 'CA': 7.50, 'ZN': 11.0, 'FE': 9.20,
        'NA': 4.00, 'K': 6.80, 'CL': 6.50, 'I': 18.5,
        'UNK': 2.50
    }

    atomic_masses = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'P': 30.974, 'S': 32.065, 'SE': 78.960,
        'MG': 24.305, 'CA': 40.078, 'ZN': 65.380, 'FE': 55.845,
        'NA': 22.990, 'K': 39.098, 'CL': 35.450, 'I': 126.904,
        'UNK': 12.011
    }

    protein_bb = ['N', 'CA', 'C', 'O', 'OXT']
    nucleic_bb = ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "C3'", "O3'"]

    ATOM_NAME_TO_ELEMENT = {
        'CA': 'C', 'CB': 'C', 'CG': 'C', 'CG1': 'C', 'CG2': 'C',
        'CD': 'C', 'CD1': 'C', 'CD2': 'C', 'CE': 'C', 'CE1': 'C',
        'CE2': 'C', 'CE3': 'C', 'CZ': 'C', 'CZ2': 'C', 'CZ3': 'C', 'CH2': 'C',
        "C1'": 'C', "C2'": 'C', "C3'": 'C', "C4'": 'C', "C5'": 'C',
        'N': 'N', 'ND1': 'N', 'ND2': 'N', 'NE': 'N', 'NE1': 'N',
        'NE2': 'N', 'NZ': 'N', 'NH1': 'N', 'NH2': 'N',
        'O': 'O', 'OXT': 'O', 'OG': 'O', 'OG1': 'O', 'OD1': 'O',
        'OD2': 'O', 'OE1': 'O', 'OE2': 'O', 'OH': 'O',
        'P': 'P', 'SD': 'S', 'SG': 'S', 'SE': 'SE'
    }

    def __init__(self, pdb_path):
        self.pdb_path = pdb_path
        if self.pdb_path.lower().endswith('.cif'):
            all_atoms, weights_list, mass_list = self._parse_cif(self.pdb_path)
        else:
            all_atoms, weights_list, mass_list = self._parse_pdb(self.pdb_path)

        if len(all_atoms) == 0:
            raise ValueError(f"No valid atoms found in {self.pdb_path}.")

        self.atom_weights = jnp.array(weights_list, dtype=jnp.float32)
        self.mass_kda = sum(mass_list) / 1000.0

        sum_weights = jnp.sum(self.atom_weights) + EPS
        self.com = jnp.sum(jnp.array(all_atoms) * self.atom_weights[:, None], axis=0) / sum_weights
        self.all_coords = jnp.array(jnp.array(all_atoms) - self.com)

    @staticmethod
    def _zero_distant_hetatm_weights(all_atoms, weights, masses, is_hetatm_flags, protein_ca_coords):
        hetatm_idx = [i for i, is_het in enumerate(is_hetatm_flags) if is_het]
        if not hetatm_idx:
            return
        if not protein_ca_coords:
            for i in hetatm_idx:
                weights[i] = 0.0
                masses[i] = 0.0
            return
        ca_array = np.array(protein_ca_coords)
        het_coords = np.array([all_atoms[i] for i in hetatm_idx])
        dists_sq = np.sum((het_coords[:, None, :] - ca_array[None, :, :]) ** 2, axis=-1)
        min_dists_sq = np.min(dists_sq, axis=1)
        for k, i in enumerate(hetatm_idx):
            if min_dists_sq[k] >= 15.0 ** 2:
                weights[i] = 0.0
                masses[i] = 0.0

    def _parse_pdb(self, pdb_path):
        all_atoms, weights, masses, is_hetatm_flags = [], [], [], []
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
                        if not elem or elem == "":
                            elem = self.ATOM_NAME_TO_ELEMENT.get(atom_name)

                        if not elem:
                            clean_name = ''.join([c for c in atom_name if c.isalpha()])
                            elem = clean_name[0] if clean_name else 'C'

                        weight = self.electron_scattering.get(elem, self.electron_scattering['UNK'])
                        mass = self.atomic_masses.get(elem, self.atomic_masses['UNK'])

                        if not is_hetatm and (atom_name == 'CA' or atom_name == 'P'):
                            protein_ca_coords.append(coord)

                        all_atoms.append(coord)
                        weights.append(weight)
                        masses.append(mass)
                        is_hetatm_flags.append(is_hetatm)

                    except ValueError:
                        continue

        self._zero_distant_hetatm_weights(all_atoms, weights, masses, is_hetatm_flags, protein_ca_coords)
        return all_atoms, weights, masses

    def _parse_cif(self, path):
        all_atoms, weights, masses, is_hetatm_flags = [], [], [], []
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

                        elem = ""
                        if idx_elem != -1 and idx_elem < len(parts):
                            elem = parts[idx_elem].strip().upper()
                            elem = ''.join([c for c in elem if c.isalpha()])

                        if not elem or elem == "":
                            elem = self.ATOM_NAME_TO_ELEMENT.get(atom_name)

                        if not elem:
                            clean_name = ''.join([c for c in atom_name if c.isalpha()])
                            elem = clean_name[0] if clean_name else 'C'

                        weight = self.electron_scattering.get(elem, self.electron_scattering['UNK'])
                        mass = self.atomic_masses.get(elem, self.atomic_masses['UNK'])

                        if not is_hetatm and (atom_name == 'CA' or atom_name == 'P'):
                            protein_ca_coords.append(coord)

                        all_atoms.append(coord)
                        weights.append(weight)
                        masses.append(mass)
                        is_hetatm_flags.append(is_hetatm)

                    except (IndexError, ValueError):
                        continue

        self._zero_distant_hetatm_weights(all_atoms, weights, masses, is_hetatm_flags, protein_ca_coords)
        return all_atoms, weights, masses

class RigidTransformation(eqx.Module):
    global_shift: jnp.ndarray
    global_rot_quat: jnp.ndarray

    def __init__(self, init_quat=None, init_shift=None):
        self.global_shift = init_shift if init_shift is not None else jnp.zeros(3)
        self.global_rot_quat = init_quat if init_quat is not None else jnp.array([1.0, 0.0, 0.0, 0.0])

    def __call__(self):
        g_q = self.global_rot_quat / (jnp.linalg.norm(self.global_rot_quat) + EPS)
        return g_q, self.global_shift

class RigidEngine:
    def __init__(self, topo, vol, apix, out_dir):
        self.topo = topo
        self.apix = apix
        self.out_dir = out_dir
        self.vol_raw = vol
        self.vol_shape = self.vol_raw.shape
        self.center_proj = jnp.array(vol.shape) * self.apix / 2.0
        self.transformation = RigidTransformation()
        self.rasterizer = SparseGaussianRasterizer(vol.shape, self.apix, sigma_s=self.apix, kernel_width=5)
        self.best_candidates = []
        self.best_hand = "original"

    def get_transformed_coords(self, transformation):
        g_q, g_s = transformation()
        R_global = quaternion_to_matrix(g_q)
        return jnp.dot(self.topo.all_coords, R_global.T) + g_s

    @eqx.filter_jit
    def compute_scalar_cc(self, rot_m, shift_v, target_vol):
        sim_vol = self.rasterizer(jnp.dot(self.topo.all_coords, rot_m.T) + self.center_proj + shift_v,
                                  self.topo.atom_weights)
        mask = (sim_vol > EPS).astype(sim_vol.dtype)

        num = jnp.sum(sim_vol * target_vol)
        den = jnp.sqrt(jnp.sum(sim_vol ** 2)) * jnp.sqrt(jnp.sum(mask * (target_vol ** 2))) + EPS

        return num / den

    def global_grid_search(self, vol_s_orig, vol_s_flip=None, topk_nms=5, n_rots=4000, batch_size=100, check_flip=False):
        apix_s_x = self.apix * (self.vol_shape[0] / vol_s_orig.shape[0])
        apix_s_y = self.apix * (self.vol_shape[1] / vol_s_orig.shape[1])
        apix_s_z = self.apix * (self.vol_shape[2] / vol_s_orig.shape[2])
        apix_s = float(jnp.mean(jnp.array([apix_s_x, apix_s_y, apix_s_z])))
        sigma_s = apix_s * 0.5

        raster_s = SparseGaussianRasterizer(vol_s_orig.shape, apix_s, sigma_s, kernel_width=5)
        rots = generate_uniform_rotations(n_rots)
        herm = _hermitian_weights(vol_s_orig.shape)

        @jit
        def process_batch(rot_batch, maps):
            coords_rot = jnp.einsum('bij,nj->bni', rot_batch, self.topo.all_coords) + self.center_proj
            probe = vmap(lambda c: raster_s(c, self.topo.atom_weights))(coords_rot)
            lo, hi = coords_rot.min(axis=1), coords_rot.max(axis=1)
            return vmap(lambda v, l, h: vectorial_search_batch(v, l, h, *maps, herm, vol_s_orig.shape, apix_s,
                                                               top_k=topk_nms))(probe, lo, hi)

        hands_to_search = [("original", vol_s_orig)]
        if check_flip and vol_s_flip is not None:
            hands_to_search.append(("flipped", vol_s_flip))

        self.best_candidates = []
        for hand, vol_target in hands_to_search:
            maps = precompute_target_vectorial(vol_target, apix_s, sigma_s)
            all_cc = jnp.zeros((n_rots, topk_nms), dtype=jnp.float32)
            all_sh = jnp.zeros((n_rots, topk_nms, 3), dtype=jnp.float32)

            pbar = tqdm(range(0, n_rots, batch_size), desc=f"Global Vectorial's Search ({hand})")
            for i in pbar:
                end_idx = min(i + batch_size, n_rots)
                cc_b, sh_b = process_batch(rots[i:end_idx], maps)
                all_cc = all_cc.at[i:end_idx].set(cc_b)
                all_sh = all_sh.at[i:end_idx].set(sh_b)
                pbar.set_postfix(Max_CC=f"{float(jnp.max(cc_b)):.3f}")

            flat_cc = np.array(all_cc).flatten()
            all_sh_np = np.array(all_sh)
            rots_np = np.array(rots)

            order = jnp.argsort(flat_cc)[::-1]
            cands = []
            for idx in order:
                if np.isnan(flat_cc[idx]) or np.isinf(flat_cc[idx]): continue
                idx = int(idx)
                rot = rots_np[idx // topk_nms]
                shift = all_sh_np[idx // topk_nms, idx % topk_nms]

                is_distinct = True
                for c in cands:
                    if angle_between_matrices(rot, c["rot"]) <= 15.0 and np.linalg.norm(shift - c["shift"]) <= 5.0:
                        is_distinct = False
                        break

                if is_distinct:
                    cands.append({
                        "hand": hand,
                        "rot": jnp.array(rot),
                        "shift": jnp.array(shift),
                        "cc": float(flat_cc[idx])
                    })
                    if len(cands) >= topk_nms:
                        break
            self.best_candidates.extend(cands)

        vol_raw_flip = jnp.flip(self.vol_raw, axis=0) if check_flip else None
        for cand in self.best_candidates:
            target = self.vol_raw if cand["hand"] == "original" else vol_raw_flip
            cand["raw_scalar_cc"] = float(self.compute_scalar_cc(cand["rot"], cand["shift"], target))

        self.best_candidates.sort(key=lambda x: x["cc"], reverse=True)
        best_cand = self.best_candidates[0]

        self.transformation = eqx.tree_at(lambda m: m.global_rot_quat, self.transformation,
                                          matrix_to_quaternion(best_cand["rot"]))
        self.transformation = eqx.tree_at(lambda m: m.global_shift, self.transformation,
                                          best_cand["shift"] + self.center_proj)
        self.best_hand = best_cand["hand"]

        gap_deg = float(np.degrees(3.8 * (n_rots ** (-1.0 / 3.0))))
        print(f"\n>>> GLOBAL ALIGNMENT COMPLETE ({best_cand['hand'].upper()})")
        print(f"Grid Discretization: Max Angular Gap ~{gap_deg:.1f} deg")
        print(f"Raw Scalar CC: {best_cand['raw_scalar_cc']:.4f} | Vectorial's CC: {best_cand['cc']:.4e}\n")

    def torque_polishing(self, vol_s_orig, vol_s_flip=None, n_polish=5, polish_iters=60):
        apix_s_x = self.apix * (self.vol_shape[0] / vol_s_orig.shape[0])
        apix_s_y = self.apix * (self.vol_shape[1] / vol_s_orig.shape[1])
        apix_s_z = self.apix * (self.vol_shape[2] / vol_s_orig.shape[2])
        apix_s = float(jnp.mean(jnp.array([apix_s_x, apix_s_y, apix_s_z])))
        sigma_s = apix_s * 0.5

        raster_s = SparseGaussianRasterizer(vol_s_orig.shape, apix_s, sigma_s, kernel_width=5)

        hand = self.best_hand
        vol_target = vol_s_orig if hand == "original" else vol_s_flip
        tf, fgt2, W = precompute_target_vectorial(vol_target, apix_s, sigma_s)
        herm = _hermitian_weights(vol_s_orig.shape)
        grad_t2 = jnp.fft.irfftn(fgt2, s=vol_s_orig.shape)
        n_vox = float(np.prod(vol_s_orig.shape))

        coords, weights = self.topo.all_coords, self.topo.atom_weights
        rg = float(jnp.sqrt(jnp.sum(weights[:, None] * coords ** 2) / jnp.sum(weights)))

        box = jnp.array(vol_s_orig.shape) * apix_s
        def vectorial_cc(x, rot, shift, e_t=None):
            R = expm_so3(x[:3] / rg) @ rot
            xyz = coords @ R.T + self.center_proj + shift + x[3:]
            probe = raster_s(xyz, weights)
            fp = jnp.fft.rfftn(probe)
            cc = jnp.sum(herm * jnp.real(jnp.conj(fp) * tf)) / n_vox
            e_p = jnp.sum(herm * W * jnp.abs(fp) ** 2) / n_vox
            if e_t is None:
                e_t = jnp.sum((probe > 1e-4 * jnp.max(probe)) * grad_t2)
            out = jnp.any((xyz < 0) | (xyz >= box))
            return jnp.where(out, -1.0, cc / jnp.sqrt(e_p * e_t + 1e-30)), e_t

        cc_fn = jit(vectorial_cc)
        cc_and_grad_fn = jit(jax.value_and_grad(vectorial_cc, has_aux=True))

        polished_cands = []
        cands_hand = [c for c in self.best_candidates if c["hand"] == hand][:n_polish]

        for cand in tqdm(cands_hand, desc=f"Torque Polishing ({hand})"):
            rot = jnp.array(cand["rot"])
            shift = jnp.array(cand["shift"])
            R_start = rot
            _, e_t = cc_fn(jnp.zeros(6), rot, shift)

            def value(x):
                return float(cc_fn(jnp.asarray(x, jnp.float32), rot, shift, e_t)[0])

            def value_and_grad(x):
                (cc, _), grad = cc_and_grad_fn(jnp.asarray(x, jnp.float32), rot, shift, e_t)
                return float(cc), np.asarray(grad, np.float64)

            x = bfgs_maximise(value, value_and_grad, first_step=sigma_s, max_iter=polish_iters)

            rot = expm_so3(jnp.asarray(x[:3] / rg, jnp.float32)) @ rot
            shift = shift + jnp.asarray(x[3:], jnp.float32)
            cc, _ = cc_fn(jnp.zeros(6), rot, shift)

            cand["rot"], cand["shift"], cand["cc_polished"] = rot, shift, float(cc)
            cand["turned_deg"] = angle_between_matrices(R_start, rot)
            polished_cands.append(cand)

        vol_raw_flip = jnp.flip(self.vol_raw, axis=0) if hand == "flipped" else None
        for cand in polished_cands:
            target = self.vol_raw if cand["hand"] == "original" else vol_raw_flip
            cand["final_scalar_cc"] = float(self.compute_scalar_cc(cand["rot"], cand["shift"], target))

        polished_cands.sort(key=lambda x: x["cc_polished"], reverse=True)
        best_cand = polished_cands[0]

        self.transformation = eqx.tree_at(lambda m: m.global_rot_quat, self.transformation,
                                          matrix_to_quaternion(best_cand["rot"]))
        self.transformation = eqx.tree_at(lambda m: m.global_shift, self.transformation,
                                          best_cand["shift"] + self.center_proj)
        self.best_hand = best_cand["hand"]

        print(f"\n>>> LOCAL REFINEMENT COMPLETE ({best_cand['hand'].upper()})")
        print(f"Corrected Kinematic Deflection: {best_cand['turned_deg']:.1f} deg")
        print(f"Final Scalar CC: {best_cand['final_scalar_cc']:.4f} | Vectorial CC: {best_cand['cc_polished']:.4e}\n")

    def save(self, output_name, is_aligned=False):
        print(f"Saving results to {self.out_dir}/{output_name}...")
        all_final_coords_orig = self.get_transformed_coords(self.transformation)
        if not is_aligned:
            all_final_coords = all_final_coords_orig - jnp.array(self.vol_shape) * self.apix / 2.0
        else:
            all_final_coords = all_final_coords_orig

        ext = os.path.splitext(self.topo.pdb_path)[1].lower()
        if ext == '.cif':
            out_file = os.path.join(self.out_dir, f"{output_name}_fitted.cif")
            self._write_cif(self.topo.pdb_path, out_file, np.array(all_final_coords))
        else:
            out_file = os.path.join(self.out_dir, f"{output_name}_fitted.pdb")
            self._write_pdb(self.topo.pdb_path, out_file, np.array(all_final_coords))

        out_vol = jnp.flip(self.vol_raw, axis=0) if self.best_hand == "flipped" else self.vol_raw

        sim_vol = self.rasterizer(all_final_coords_orig, self.topo.atom_weights)
        save_mrc(sim_vol.T, self.apix, os.path.join(self.out_dir, f"{output_name}_sim.mrc"))
        save_mrc(out_vol.T, self.apix, os.path.join(self.out_dir, f"{output_name}_input_norm.mrc"),
                 centering=False)

    def _write_pdb(self, in_path, out_path, coords):
        with open(in_path, 'r') as f_in, open(out_path, 'w') as f_out:
            atom_idx = 0
            for line in f_in:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        float(line[30:38])
                        float(line[38:46])
                        float(line[46:54])

                        if atom_idx < len(coords):
                            x, y, z = coords[atom_idx]
                            f_out.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                            atom_idx += 1
                        else:
                            f_out.write(line)
                    except ValueError:
                        f_out.write(line)
                else:
                    f_out.write(line)

    def _write_cif(self, in_path, out_path, coords):
        in_atom_site_header = False
        in_atom_site_data = False
        col_idx = {}
        idx_x = idx_y = idx_z = -1
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
                        try:
                            float(parts[idx_x])
                            float(parts[idx_y])
                            float(parts[idx_z])

                            if atom_idx < len(coords):
                                x, y, z = coords[atom_idx]
                                parts[idx_x] = f"{x:.3f}"
                                parts[idx_y] = f"{y:.3f}"
                                parts[idx_z] = f"{z:.3f}"
                                f_out.write(" ".join(parts) + "\n")
                                atom_idx += 1
                                continue
                        except (IndexError, ValueError):
                            f_out.write(line)
                            continue

                f_out.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--vol", type=str, required=True)
    parser.add_argument("--sr", type=float, default=1.0)
    parser.add_argument("--check_flip", action="store_true")
    parser.add_argument("--is_aligned", action="store_true")
    parser.add_argument("--topk_nms", type=int, default=10)
    parser.add_argument("--n_rots", type=int, default=4000)
    parser.add_argument("--n_polish", type=int, default=5)
    parser.add_argument("--polish_iters", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="rigid_output")
    args, _ = parser.parse_known_args()

    if args.n_polish > args.topk_nms:
        print(
            f"[WARNING] n_polish ({args.n_polish}) cannot be greater than topk_nms ({args.topk_nms}). Adjusting topk_nms to {args.n_polish}.")
        args.topk_nms = args.n_polish

    os.makedirs(args.out_dir, exist_ok=True)
    topo = ProteinTopology(args.pdb)

    with mrcfile.open(args.vol, permissive=True) as mrc:
        vol_orig = jnp.array(mrc.data).T
        apix = float(mrc.voxel_size.x) if args.sr == 1.0 else args.sr

    downfactor_bio = 2.0 * (topo.mass_kda / 50.0) ** (1 / 3)
    downfactor_comp = (np.prod(vol_orig.shape) / 4_000_000.0) ** (1 / 3)
    downfactor = float(np.max([downfactor_bio, downfactor_comp]))
    downfactor = float(np.clip(downfactor, 2.0, 4.0))
    print(f"[INFO] Auto-Configured Downsample Factor: {downfactor:.2f}")

    engine = RigidEngine(topo, vol_orig, apix, args.out_dir)
    if not args.is_aligned:
        vol_s_orig = fourier_downsample(vol_orig, downfactor)
        vol_s_flip = jnp.flip(vol_s_orig, axis=0) if args.check_flip else None

        engine.global_grid_search(vol_s_orig, vol_s_flip, topk_nms=args.topk_nms, n_rots=args.n_rots,
                                  batch_size=args.batch_size, check_flip=args.check_flip)
        engine.save("final_rigid", is_aligned=args.is_aligned)

        engine.torque_polishing(vol_s_orig, vol_s_flip, n_polish=args.n_polish, polish_iters=args.polish_iters)
        engine.save("final_rigid_refined", is_aligned=args.is_aligned)
    else:
        engine.save("final_rigid", is_aligned=args.is_aligned)


if __name__ == "__main__":
    main()
