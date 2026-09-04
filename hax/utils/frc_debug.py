"""
frc_debug.py

For a few particles, compare the FRC curve of the true (aligned)
projection against a slightly perturbed (misaligned) one, plotted side by
side for visual inspection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def compare_frc_small_perturbation(
    vol, mask, md_columns, x_size, n_shells, args,
    data_loader_iter, rngs, n_particles,
    md_extraction, volumeProjection, compute_fourier_residual,
    standardize_frc_curve, generate_misalignment,
    alpha_min_deg=2.0, alpha_max_deg=10.0,
    shift_min_px=0.5, shift_max_frac=None,
    output_path=None,
):
    if shift_max_frac is None:
        shift_max_frac = 1.0 / x_size

    (x, index) = next(data_loader_iter)
    euler_angles, shifts, ctf = md_extraction(md_columns, index, vol, args)

    # Aligned
    proj_al = volumeProjection(vol=vol, mask=mask, euler_angles=euler_angles, shifts=shifts, ctf=ctf)
    result_al = compute_fourier_residual(proj_al, x, n_shells=n_shells)
    aligned_frc, freqs_ref = standardize_frc_curve(result_al.frc_curve, result_al.freqs, ts=args.sr, n_shells_ref=n_shells)

    # Small perturbation
    rngs, euler_angles_noisy, shifts_noisy = generate_misalignment(
        rngs, euler_angles, shifts, box_size=x_size,
        alpha_min_deg=alpha_min_deg, alpha_max_deg=alpha_max_deg,
        shift_min_px=shift_min_px, shift_max_frac=shift_max_frac)
    proj_mis = volumeProjection(vol=vol, mask=mask, euler_angles=euler_angles_noisy, shifts=shifts_noisy, ctf=ctf)
    result_mis = compute_fourier_residual(proj_mis, x, n_shells=n_shells)
    misaligned_frc = standardize_frc_curve(result_mis.frc_curve, result_mis.freqs, ts=args.sr, n_shells_ref=n_shells)[0]

    # Plot: n_particles subplots, one aligned/misaligned pair each
    fig, axes = plt.subplots(1, n_particles, figsize=(4 * n_particles, 4))
    if n_particles == 1:
        axes = [axes]

    for k in range(n_particles):
        axes[k].plot(freqs_ref, np.array(aligned_frc[k]), label="Aligned")
        axes[k].plot(freqs_ref, np.array(misaligned_frc[k]), label="Misaligned")
        axes[k].set_title(f"id {int(index[k])}")
        axes[k].set_ylim(-0.2, 1.05)
        axes[k].set_xlabel("Freq (cycles/Å)")
    axes[0].set_ylabel("FRC")
    axes[0].legend()

    plt.tight_layout()

    if output_path is not None:
        out_file = os.path.join(output_path, "frc_perturbation.png")
        plt.savefig(out_file, dpi=150)
        print(f"Salvato: {out_file}")

    return rngs, fig