import numpy as np
import matplotlib.pyplot as plt


def plot_angular_distribution(euler_angles):
    fig, ax = plt.subplots(figsize=(9, 9))
    x = euler_angles[:, 1] * np.cos(euler_angles[:, 0])
    y = euler_angles[:, 1] * np.sin(euler_angles[:, 0])
    hb = ax.hexbin(x, y, gridsize=30, cmap='viridis', mincnt=1)
    cb = fig.colorbar(hb, ax=ax, label='Counts')
    cb.ax.yaxis.set_tick_params(color='white')  # Ensure tick marks are white
    cb.outline.set_edgecolor('white')  # Ensure colorbar border is white
    cb.set_label('Counts', color='white')  # Ensure colorbar label is white
    ax.set_title("Angular distribution density", fontsize=16)
    ax.set_aspect('equal')
    ax.axis('off')
    max_r = np.pi
    circle_radii = [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    circle_labels = ['45°', '90° (Equator)', '135°', '180° (Down)']
    for r, label in zip(circle_radii, circle_labels):
        circle = plt.Circle((0, 0), r, color='lightgray', fill=False, linestyle='--', alpha=0.6, linewidth=2.0)
        ax.add_patch(circle)
        ax.text(0, r + 0.2, label, color='white', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0, max_r * 1.25, "Rot = 90° (Up)", ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    ax.text(0, -max_r * 1.25, "Rot = -90° (Down)", ha='center', va='center', color='white', fontsize=11,
            fontweight='bold')
    ax.text(max_r * 1.25, 0, "Rot = 0°", ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    ax.text(-max_r * 1.25, 0, "Rot = 180°", ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    ax.set_xlim(-max_r * 1.45, max_r * 1.45)
    ax.set_ylim(-max_r * 1.45, max_r * 1.45)
    plt.tight_layout()
    return fig, ax