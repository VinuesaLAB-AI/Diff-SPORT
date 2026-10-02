"""
Phase 1 Evaluation: Plot MSE vs number of sensors from random baseline results.

Usage:
    python phase1_evaluate.py [--input results/phase1_random_baseline.npz]
                              [--out_dir results/]
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from configs.plot_config import basic_plt_setup
from libs.cond_gen_plif import plot_mask_PLIF
from utils import get_sensor_pixel_indices, create_random_sensor_mask

basic_plt_setup()


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Evaluate random baseline")
    parser.add_argument("--input", type=str, default="results/phase1_random_baseline/data/phase1_random_baseline.npz",
                        help="Path to baseline NPZ")
    parser.add_argument("--out_dir", type=str, default="results/phase1_random_baseline/plots",
                        help="Output directory for plots")
    parser.add_argument("--plot_masks", action="store_true",
                        help="Plot representative sensor masks (base + first/mid/last sweep points)")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = SCRIPT_DIR / input_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = SCRIPT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = np.load(str(input_path), allow_pickle=True)
    n_sensors_list = results["n_sensors_list"]
    mean_mse_matrix = results["mean_mse_matrix"]  
    total_pixels = int(results["total_pixels"])
    n_pixels = int(results["n_pixels"])

    # Normalized-space MSE 
    has_norm = "mean_mse_norm_matrix" in results
    if has_norm:
        mean_mse_norm_matrix = results["mean_mse_norm_matrix"]

    # Percentage of total pixels
    pct_list = n_sensors_list / total_pixels * 100

    def plot_mse_curve(n_sensors, mean, std, ylabel, title_suffix, out_path, log_scale=False):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.errorbar(n_sensors, mean, yerr=std,
                     fmt='o-', capsize=4, capthick=1.5, linewidth=2,
                     markersize=6, color='steelblue', label='Random baseline')
        ax1.set_xlabel("Number of sensor pixels")
        ax1.set_ylabel(ylabel)
        if log_scale:
            ax1.set_yscale("log")
        ax1.set_title(f"Random Baseline: {title_suffix} (base mask n_pixels={n_pixels})")
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both' if log_scale else 'major')

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        tick_positions = ax1.get_xticks()
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f"{x / total_pixels * 100:.1f}%" for x in tick_positions])
        ax2.set_xlabel("% of total pixels")

        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    # --- Physical-space MSE ---
    mean_mse = np.mean(mean_mse_matrix, axis=1)
    std_mse = np.std(mean_mse_matrix, axis=1)

    plot_mse_curve(n_sensors_list, mean_mse, std_mse,
                   "MSE (physical)", "MSE vs Sensors",
                   out_dir / "phase1_mse_vs_sensors.pdf")
    plot_mse_curve(n_sensors_list, mean_mse, std_mse,
                   "MSE (physical, log)", "MSE vs Sensors",
                   out_dir / "phase1_mse_vs_sensors_log.pdf", log_scale=True)

    # --- Normalized-space MSE ---
    if has_norm:
        mean_mse_norm = np.mean(mean_mse_norm_matrix, axis=1)
        std_mse_norm = np.std(mean_mse_norm_matrix, axis=1)

        plot_mse_curve(n_sensors_list, mean_mse_norm, std_mse_norm,
                       "MSE (normalized)", "MSE (norm space) vs Sensors",
                       out_dir / "phase1_mse_norm_vs_sensors.pdf")
        plot_mse_curve(n_sensors_list, mean_mse_norm, std_mse_norm,
                       "MSE (normalized, log)", "MSE (norm space) vs Sensors",
                       out_dir / "phase1_mse_norm_vs_sensors_log.pdf", log_scale=True)
    else:
        print("Note: normalized-space MSE not found in results (older run). Skipping norm plots.")

    # ----- Print summary table -----
    if has_norm:
        print(f"\n{'Sensors':>8} {'%':>7} {'MSE_phys':>12} {'Std_phys':>12} {'MSE_norm':>12} {'Std_norm':>12}")
        print("-" * 70)
        for i in range(len(n_sensors_list)):
            print(f"{n_sensors_list[i]:8d} {pct_list[i]:6.2f}% {mean_mse[i]:12.6e} {std_mse[i]:12.6e}"
                  f" {mean_mse_norm[i]:12.6e} {std_mse_norm[i]:12.6e}")
    else:
        print(f"\n{'Sensors':>8} {'%':>7} {'Mean MSE':>12} {'Std MSE':>12}")
        print("-" * 42)
        for i in range(len(n_sensors_list)):
            print(f"{n_sensors_list[i]:8d} {pct_list[i]:6.2f}% {mean_mse[i]:12.6e} {std_mse[i]:12.6e}")

    # ----- Plot representative masks (optional) -----
    if args.plot_masks:
        base_mask = results["base_mask"]          # (1, H, W)
        obstacle_mask = results["obstacle_mask"]  # (H, W) boolean
        x_coords = results["x_coords"]
        y_coords = results["y_coords"]
        mask_seeds = results["mask_seeds"]
        H, W = obstacle_mask.shape

        sensor_indices = get_sensor_pixel_indices(
            torch.from_numpy(base_mask), obstacle_mask
        )

        # Base mask
        plot_mask_PLIF(
            mask_tensor=torch.from_numpy(base_mask),
            obstacle_mask=obstacle_mask,
            x_coords=x_coords, y_coords=y_coords,
            mask_type=f"base_mask_n{n_pixels}",
            out_pdf=False, out_png=True, out_show=False,
        )
        src = SCRIPT_DIR / f"mask-PLIF-base_mask_n{n_pixels}.png"
        dst = out_dir / f"phase1_base_mask_n{n_pixels}.png"
        if src.exists():
            src.rename(dst)
            print(f"Saved: {dst}")

        # Representative sweep points: first, middle, last
        repr_indices = [0, len(n_sensors_list) // 2, len(n_sensors_list) - 1]
        seed_to_plot = int(mask_seeds[0])

        for idx in repr_indices:
            ns = int(n_sensors_list[idx])
            sensor_mask = create_random_sensor_mask(
                sensor_indices, ns, (H, W), seed=seed_to_plot
            )
            mask_label = f"random_{ns}sensors_seed{seed_to_plot}"

            plot_mask_PLIF(
                mask_tensor=sensor_mask,
                obstacle_mask=obstacle_mask,
                x_coords=x_coords, y_coords=y_coords,
                mask_type=mask_label,
                out_pdf=False, out_png=True, out_show=False,
            )
            src = SCRIPT_DIR / f"mask-PLIF-{mask_label}.png"
            dst = out_dir / f"phase1_mask_{ns}sensors_seed{seed_to_plot}.png"
            if src.exists():
                src.rename(dst)
                print(f"Saved: {dst}")

        print("\nMask plots complete.")


if __name__ == "__main__":
    main()
