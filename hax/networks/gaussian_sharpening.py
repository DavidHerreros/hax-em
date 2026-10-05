import os, sys
import numpy as np
import jax
import optax
import jax.numpy as jnp
from flax import nnx
from functools import partial
from tqdm import tqdm
from xmipp_metadata.image_handler import ImageHandler
from cuml.neighbors.nearest_neighbors import NearestNeighbors
from hax.utils import *
from hax.utils.fsc import hermitian_multiplicity
from hax.programs.gaussian_volume_fitting import GaussianSplatModel

def estimate_init_gauss(volume, mask, max_gaussians):
    coords = np.argwhere((mask > 0.0))
    densities = volume[coords[:, 0], coords[:, 1], coords[:, 2]]
    p_weights = np.clip(densities, 0.0, None)

    sum_weights = np.sum(p_weights)
    if sum_weights < 1e-6:
        p_weights = np.ones_like(p_weights) / len(p_weights)
    else:
        p_weights /= sum_weights

    n_active = len(coords)
    target_k = n_active // 8
    k_init = min(target_k, max_gaussians)
    indices = np.random.choice(n_active, size=k_init, replace=False, p=p_weights)
    sampled_coords = coords[indices]

    weights = volume[sampled_coords[:, 0], sampled_coords[:, 1], sampled_coords[:, 2]]
    weights = weights * (sum_weights / max(float(np.sum(weights)), 1e-12))

    factor = 0.5 * volume.shape[0]
    means_norm = (np.stack([sampled_coords[:, 2], sampled_coords[:, 1], sampled_coords[:, 0]], axis=1).astype(
        np.float32) - factor) / factor

    nbrs = NearestNeighbors(n_neighbors=2).fit(means_norm)
    distances, _ = nbrs.kneighbors(means_norm)
    optimal_sigma = np.mean(distances[:, 1]) * factor

    return {"means": means_norm, "weights": weights}, float(optimal_sigma)

def sharp_volume(target_vol, mask=None, fourier_weight=None, ratio_grad=None, ratio_lap=None, ratio_tv=None,
               ratio_spars=None, ratio_bound=None, ratio_sigma=None, l1_weight=0.2,
               n_iterations=1000, max_iterations=5000, learning_rate=0.01, max_gaussians=50000):
    # Grid size
    grid_size = target_vol.shape[0]

    inside = np.asarray(mask) > 0 if mask is not None else np.ones(target_vol.shape, bool)
    intensity_scale = float(np.percentile(np.asarray(target_vol)[inside], 99))
    if not intensity_scale > 0:
        intensity_scale = float(np.max(np.abs(np.asarray(target_vol)))) or 1.0
    target_vol = target_vol / intensity_scale

    if mask is not None:
        manual_init, optimal_sigma = estimate_init_gauss(target_vol, mask, max_gaussians)
        active_mask = jnp.array(mask, dtype=jnp.float32)

        # Init Model
        rngs = nnx.Rngs(42)
        model = GaussianSplatModel(manual_init=manual_init, sigma=optimal_sigma, grid_size=grid_size, rngs=rngs)
    else:
        # Init Model
        rngs = nnx.Rngs(42)
        model = GaussianSplatModel(n_init=1000, sigma=1.0, grid_size=grid_size, rngs=rngs)
        active_mask = jnp.zeros_like(target_vol, dtype=jnp.float32)

    # Init Optimizer (nnx.Optimizer automatically tracks model params)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)
    k_history = []

    print(f"\n{bcolors.OKCYAN}###### Starting Adaptive Fit on {grid_size}^3 volume... ######{bcolors.ENDC}")

    graphdef, state = nnx.split((model, optimizer))
    pbar = tqdm(range(max_iterations), desc="Fitting volume", file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

    target_max = jnp.max(target_vol)
    ratios_dict = {
        "grad": ratio_grad,
        "lap": ratio_lap,
        "tv": ratio_tv,
        "spars": ratio_spars,
        "bound": ratio_bound,
        "sigma": ratio_sigma
    }
    loss_weights = {k: jnp.array(1.0, dtype=jnp.float32) for k in ratios_dict.keys()}

    def recalibrate_weights(current_recon, raw_vals):
        new_weights = {}
        EPS = 1e-8
        for key_name, ratio in ratios_dict.items():
            weight_fixed = 1.0 if ratio is None else ratio

            raw_v = raw_vals[key_name]
            if key_name in ['bound', 'sigma']:
                safe_val = raw_v if raw_v > EPS else 0.1
                new_weights[key_name] = jnp.array(current_recon * weight_fixed / safe_val, dtype=jnp.float32)
            else:
                safe_val = raw_v if raw_v > EPS else EPS
                new_weights[key_name] = jnp.array(current_recon * weight_fixed / safe_val, dtype=jnp.float32)

        return new_weights

    if fourier_weight is None:
        fourier_weight = jnp.ones((grid_size, grid_size, grid_size // 2 + 1), dtype=jnp.float32)

    key = jax.random.PRNGKey(42)
    with pbar:
        for i in range(max_iterations):
            is_densification_step = (i > 0 and i % n_iterations == 0)

            if is_densification_step:
                key, subkey = jax.random.split(key)
                progress = i / max_iterations
                current_lr = learning_rate * (0.05 ** progress)

                model, optimizer = nnx.merge(graphdef, state)
                optimizer = adapt_merge_gaussians(model, grads, subkey, target_max=target_max, max_gaussians=max_gaussians,
                                            lr=current_lr)
                graphdef, state = nnx.split((model, optimizer))

            if i == 0 or is_densification_step:
                _, aux, _, _ = sharpening_step_volume(
                    graphdef, state, target_vol, active_mask, loss_weights,
                    l1_weight=l1_weight, fourier_weight=fourier_weight, update=False
                )

                current_recon_loss = float(aux[1])
                raw_vals = {
                    "grad": float(aux[8]),
                    "lap": float(aux[9]),
                    "tv": float(aux[10]),
                    "spars": float(aux[11]),
                    "bound": float(aux[12]),
                    "sigma": float(aux[13])
                }
                loss_weights = recalibrate_weights(current_recon_loss, raw_vals)

            loss_val, aux, grads, state = sharpening_step_volume(
                graphdef, state, target_vol, active_mask, loss_weights,
                l1_weight=l1_weight, fourier_weight=fourier_weight, update=True
            )

            model, _ = nnx.merge(graphdef, state)
            current_k = model.means.get_value().shape[0]
            k_history.append(current_k)

            s = float(jax.nn.softplus(model.sigma_param.get_value())[0])

            pbar.set_postfix_str(f"| TotLoss: {float(loss_val):.5f} | K: {current_k:04d} | sig: {s:.3f} "
                                 f"| REC: {aux[1]:.5f} | GRA: {aux[2]:.5f} | LAP: {aux[3]:.5f} | TV: {aux[4]:.5f} | SPARS: {aux[5]:.5f} "
                                 f"| BOUND: {aux[6]:.5f} | SIGMA: {aux[7]:.5f}")
            pbar.update(1)

    model, _ = nnx.merge(graphdef, state)

    # FINAL PRUNING
    means = model.means.get_value()
    weights = model.weights.get_value()
    actual_weights = nnx.relu(weights)

    weight_floor = jnp.max(actual_weights) * 0.005
    keep_mask = actual_weights > weight_floor

    filtered_means = means[keep_mask]
    filtered_weights = weights[keep_mask]

    final_means, final_weights = filtered_means, filtered_weights
    final_weights = final_weights * intensity_scale

    model.means = nnx.Param(final_means)
    model.weights = nnx.Param(final_weights)
    model.update_config()

    return model, k_history

def adapt_merge_gaussians(model, grads, key, target_max, max_gaussians=50000, lr=None, optimizer=None):
    means = model.means.get_value()
    weights_param = model.weights.get_value()
    actual_weights = nnx.relu(weights_param)

    prune_threshold = jnp.max(actual_weights) * 0.005
    keep_mask = actual_weights > prune_threshold

    means = means[keep_mask]
    actual_weights = actual_weights[keep_mask]
    grad_means = grads.means.get_value()[keep_mask]

    d_max_px = 0.5
    d_max = d_max_px * (2.0 / model.grid_size)
    voxel_coords = jnp.round(means / d_max).astype(jnp.int32)
    voxel_coords = voxel_coords - jnp.min(voxel_coords, axis=0)
    grid_size_hash = jnp.max(voxel_coords, axis=0) + 1

    voxel_ids_grid = voxel_coords[:, 0] + \
                     voxel_coords[:, 1] * grid_size_hash[0] + \
                     voxel_coords[:, 2] * (grid_size_hash[0] * grid_size_hash[1])

    grad_norms_pre = jnp.linalg.norm(grad_means, axis=-1)
    is_stable = grad_norms_pre < jnp.mean(grad_norms_pre)

    safe_ids = (grid_size_hash[0] * grid_size_hash[1] * grid_size_hash[2]) + jnp.arange(means.shape[0])
    voxel_ids = jnp.where(is_stable, voxel_ids_grid, safe_ids)

    unique_ids, inverse_indices = jnp.unique(voxel_ids, return_inverse=True)

    weighted_means = means * actual_weights[:, None]
    sum_weighted_means = jax.ops.segment_sum(weighted_means, inverse_indices, num_segments=unique_ids.shape[0])
    merged_weights_actual = jax.ops.segment_sum(actual_weights, inverse_indices, num_segments=unique_ids.shape[0])

    safe_merged_weights = jnp.where(merged_weights_actual > 0, merged_weights_actual, 1.0)
    merged_means = sum_weighted_means / safe_merged_weights[:, None]

    merged_grad_means = jax.ops.segment_sum(grad_means, inverse_indices, num_segments=unique_ids.shape[0])
    merged_weights_param = merged_weights_actual

    grad_norms = jnp.linalg.norm(merged_grad_means, axis=-1)
    normalized_grads = grad_norms / (target_max + 1e-8)
    print("grad", grad_norms, normalized_grads)

    base_split_mask = normalized_grads > 0.0002
    is_noise_chasing = normalized_grads > 0.010
    sigma_vox = jax.nn.softplus(model.sigma_param.get_value())[0]
    peak_density = merged_weights_param / ((2.0 * jnp.pi) ** 1.5 * sigma_vox ** 3)
    is_density_exploding = peak_density > (target_max * 1.5)
    overfit_mask = is_noise_chasing | is_density_exploding
    split_mask = base_split_mask & overfit_mask

    budget = max(int(max_gaussians) - int(merged_means.shape[0]), 0)
    if int(jnp.sum(split_mask)) > budget:
        split_score = jnp.where(split_mask, normalized_grads, -jnp.inf)
        best = jnp.argsort(-split_score)[:budget]
        split_mask = jnp.zeros_like(split_mask).at[best].set(True) & split_mask

    do_not_touch_mask = ~split_mask
    new_means_list = [merged_means[do_not_touch_mask]]
    new_weights_list = [merged_weights_param[do_not_touch_mask]]

    n_split = jnp.sum(split_mask)
    if n_split > 0:
        s_means = merged_means[split_mask]
        s_weights = merged_weights_param[split_mask]

        noise = jax.random.normal(key, s_means.shape) * (1.0 / model.grid_size)

        new_means_list.extend([s_means - noise, s_means + noise])
        new_weights_list.extend([s_weights * 0.5, s_weights * 0.5])

    final_means = jnp.concatenate(new_means_list, axis=0)
    final_actual_weights = jnp.concatenate(new_weights_list, axis=0)

    current_k = final_means.shape[0]
    if current_k > max_gaussians:
        top_indices = jnp.argsort(-final_actual_weights)[:max_gaussians]
        final_means = final_means[top_indices]
        final_actual_weights = final_actual_weights[top_indices]

    model.means = nnx.Param(final_means)
    model.weights = nnx.Param(final_actual_weights)

    if optimizer is not None:
        new_optimizer = nnx.Optimizer(model, optimizer.tx, wrt=nnx.Param)
    elif lr is not None:
        new_optimizer = nnx.Optimizer(model, optax.adamw(lr), wrt=nnx.Param)

    return new_optimizer

# Define Loss Function for NNX
@partial(jax.jit, static_argnames=("update",))
def sharpening_step_volume(graphdef, state, target_vol, mask, loss_weights, l1_weight, fourier_weight=None, update=True):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, target_vol, mask):
        recon = model()
        N_voxels = recon.size
        active_voxels = jnp.maximum(1.0, jnp.sum(mask))

        recon_masked = recon * mask
        if fourier_weight is not None:
            recon_ft = jnp.fft.rfftn(recon_masked)
            target_ft = jnp.fft.rfftn(target_vol)
            recon_blur_ft = recon_ft * fourier_weight
            diff_ft = recon_blur_ft - target_ft
            diff = jnp.fft.irfftn(diff_ft, s=recon.shape)

            d = recon.shape[0]
            fz = jnp.fft.fftfreq(d)[:, None, None]
            fy = jnp.fft.fftfreq(d)[None, :, None]
            fx = jnp.fft.rfftfreq(d)[None, None, :]
            k_sq = (fz ** 2 + fy ** 2 + fx ** 2) * (2.0 * jnp.pi) ** 2

            mult = hermitian_multiplicity(d)
            power = mult * (diff_ft.real ** 2 + diff_ft.imag ** 2)
            snr_weight = fourier_weight ** 2

            norm_factor_ft = active_voxels * N_voxels
            raw_grad = jnp.sum(snr_weight * k_sq * power) / norm_factor_ft
            raw_lap = jnp.sum(snr_weight * (k_sq ** 2) * power) / norm_factor_ft
        else:
            diff = recon_masked - target_vol
            raw_grad, raw_lap = 0.0, 0.0

        l1_loss = jnp.sum(jnp.abs(diff)) / active_voxels
        l2_loss = jnp.sum(diff ** 2) / active_voxels
        raw_recon_loss = l1_weight * l1_loss + (1 - l1_weight) * l2_loss
        recon_loss = raw_recon_loss

        diff_x = recon_masked[1:, :, :] - recon_masked[:-1, :, :]
        diff_y = recon_masked[:, 1:, :] - recon_masked[:, :-1, :]
        diff_z = recon_masked[:, :, 1:] - recon_masked[:, :, :-1]
        raw_tv = (jnp.sum(jnp.abs(diff_x)) + jnp.sum(jnp.abs(diff_y)) + jnp.sum(jnp.abs(diff_z))) / active_voxels
        tv_loss = loss_weights['tv'] * raw_tv

        actual_weights = nnx.relu(model.weights.get_value())
        raw_spars = jnp.sum(actual_weights) / active_voxels
        sparsity_loss = loss_weights['spars'] * raw_spars

        violation = nnx.relu(jnp.abs(model.means.get_value()) - 0.9)
        raw_boundary = jnp.mean(violation ** 2.)
        boundary_loss = loss_weights['bound'] * raw_boundary

        sigma = jax.nn.softplus(model.sigma_param.get_value())
        sigma_out_of_bounds = nnx.relu(sigma - 2.0) ** 2 + nnx.relu(0.2 - sigma) ** 2
        raw_sigma = jnp.sum(sigma_out_of_bounds)
        sigma_loss = loss_weights['sigma'] * raw_sigma

        grad_loss = loss_weights['grad'] * raw_grad
        laplacian_loss = loss_weights['lap'] * raw_lap

        total_loss = (recon_loss + boundary_loss + sigma_loss + grad_loss + laplacian_loss + tv_loss + sparsity_loss)

        return total_loss, (raw_recon_loss, recon_loss, grad_loss, laplacian_loss, tv_loss, sparsity_loss, boundary_loss, sigma_loss,
                            raw_grad, raw_lap, raw_tv, raw_spars, raw_boundary, raw_sigma)

    (loss_val, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, target_vol, mask)

    if update:
        optimizer.update(model, grads)
        state = nnx.state((model, optimizer))

    return loss_val, aux, grads, state

def save_fourier_3dfsc(fourier_weight, grid_size, output_path="fourier_psf.mrc"):
    print(f"{bcolors.OKBLUE}Saving Centered 3D-FSC Spectrum...{bcolors.ENDC}")

    delta = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    delta[0, 0, 0] = 1.0

    delta_ft = np.fft.rfftn(delta)
    filtered_delta_ft = delta_ft * np.array(fourier_weight)
    psf = np.fft.irfftn(filtered_delta_ft, s=(grid_size, grid_size, grid_size))
    full_spectrum = np.abs(np.fft.fftn(psf))
    centered_spectrum = np.fft.fftshift(full_spectrum)
    ImageHandler().write(centered_spectrum.astype(np.float32), output_path, overwrite=True)


def main():
    import argparse
    from scipy.ndimage import gaussian_filter
    from hax.generators import MetaDataGenerator
    from hax.utils.fsc import compute_fsc_weight

    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=str, required=True)
    parser.add_argument("--vol", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--half_map_1", type=str, required=True)
    parser.add_argument("--half_map_2", type=str, required=True)
    parser.add_argument("--sr", type=float, required=True)
    parser.add_argument("--n_jobs", required=False, type=int, default=1)
    parser.add_argument("--ratio_grad", required=False, type=float)
    parser.add_argument("--ratio_lap", required=False, type=float)
    parser.add_argument("--ratio_tv", required=False, type=float)
    parser.add_argument("--ratio_spars", required=False, type=float)
    parser.add_argument("--ratio_bound", required=False, type=float)
    parser.add_argument("--ratio_sigma", required=False, type=float)
    parser.add_argument("--l1_weight", required=False, type=float, default=0.8)
    parser.add_argument("--max_iterations", required=False, type=int, default=100000,
                        help='Total number of iterations to compute in gaussian fitting.')
    parser.add_argument("--max_gaussians", required=False, type=int, default=20000,
                        help='Maximum number of gaussians allowed in gaussian fitting when network tends to overfit.')
    parser.add_argument("--load_images_to_ram", action='store_true',
                        help=f"If provided, images will be loaded to RAM. This is recommended if you want the best performance "
                             f"and your dataset fits in your RAM memory. If this flag is not provided, images will be memory mapped. "
                             f"When this happens, the program will trade disk space for performance. Thus, during the execution "
                             f"additional disk space will be used and the performance will be slightly lower compared to loading "
                             f"the images to RAM. Disk usage will be back to normal once the execution has finished.")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, new metadata...)")
    parser.add_argument("--ssd_scratch_folder", required=False, type=str,
                        help=f"When the parameter {bcolors.UNDERLINE}load_images_to_ram{bcolors.ENDC} is not provided, "
                             f"we strongly recommend to provide here a path to a folder in a SSD disk to read faster the data. "
                             f"If not given, the data will be loaded from the default disk.")
    args, _ = parser.parse_known_args()

    vol = ImageHandler(args.vol).getData()
    mask_data = ImageHandler(args.mask).getData()

    mask_fsc = gaussian_filter(mask_data, sigma=2.0)
    fourier_weight, _ = compute_fsc_weight(half1=args.half_map_1, half2=args.half_map_2, mask=mask_fsc, mode='fsc',
                                               n_cones=250, cone_angle=20.0, phase_randomize=True, hard_cones=True,
                                               sr=args.sr)
    save_fourier_3dfsc(fourier_weight, mask_data.shape[0], "3dfsc.mrc")

    model, _ = sharp_volume(vol * mask_data, mask=mask_data, fourier_weight=fourier_weight, ratio_lap=args.ratio_lap,
                          ratio_grad=args.ratio_grad, ratio_tv=args.ratio_tv, ratio_spars=args.ratio_spars,
                          ratio_bound=args.ratio_bound, ratio_sigma=args.ratio_sigma, l1_weight=args.l1_weight,
                          max_iterations=args.max_iterations, learning_rate=0.01, max_gaussians=args.max_gaussians)

    vol_splatted = np.array(model())
    ImageHandler().write(vol_splatted, os.path.join(args.output_path, "consensus_volume.mrc"), overwrite=True, sr=args.sr)
    vol_deltas = np.array(model(place_deltas=True))
    ImageHandler().write(vol_deltas, os.path.join(args.output_path, "consensus_volume_deltas.mrc"), overwrite=True, sr=args.sr)

if __name__ == "__main__":
    main()