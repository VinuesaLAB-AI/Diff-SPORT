"""
Phase 2b: Aggregate SHAP values across snapshots and create threshold masks.

Loads 19 per-snapshot SHAP .npz files from Phase 2a, computes mean importance
per sensor pixel, and sweeps thresholds to produce reduced sensor masks for
Phase 2c conditional generation.

No GPU required — runs on login node.

Usage:
    python phase2_aggregate_threshold.py --n_pixels 2 --n_snapshots 19 \
        --strategy mean --thresholds 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Resolve repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from libs.shap_eval import (
    generate_all_shap_mats,
    create_shap_masks,
    mean_shap_vals,
    normalize_shap_values,
)
from configs.plot_config import basic_plt_setup
basic_plt_setup()


PHASE2_SHAP_DIR = Path(__file__).resolve().parent / "results" / "phase2_shap"
DATA_DIR = PHASE2_SHAP_DIR / "data"
PLOTS_DIR = PHASE2_SHAP_DIR / "plots"


def load_plif2d_shap_values(results_dir, n_pixels, block_size, n_snapshots, ncoalitions, start, stop, step):
    """
    Load per-snapshot SHAP .npz files and concatenate.

    Returns
    -------
    all_shaps : np.ndarray, (n_features, n_snapshots), absolute SHAP values
    seg_tensor : np.ndarray, (1, H, W)
    sensor_indices : np.ndarray, (n_features,)
    x_coords, y_coords : np.ndarray
    """
    all_shaps = []
    seg_tensor = sensor_indices = x_coords = y_coords = None

    for i in range(n_snapshots):
        fname = f"shap-plif2d-npix{n_pixels}-block{block_size}-snap{i}-ncoali{ncoalitions}-start{start}-stop{stop}-step{step}.npz"
        fpath = results_dir / fname
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping")
            continue

        data = np.load(fpath)
        shaps_col = np.abs(data["shaps"].reshape(-1, 1))  # (n_features, 1)
        all_shaps.append(shaps_col)

        if seg_tensor is None:
            seg_tensor = data["seg_tensor"]
            sensor_indices = data["sensor_indices"]
            x_coords = data["x_coords"]
            y_coords = data["y_coords"]

    if not all_shaps:
        raise RuntimeError("No SHAP files loaded.")

    all_shaps = np.concatenate(all_shaps, axis=1)  # (n_features, n_snapshots)
    print(f"Loaded {all_shaps.shape[1]} snapshots, {all_shaps.shape[0]} features")
    return all_shaps, seg_tensor, sensor_indices, x_coords, y_coords


def plot_importance_heatmap(mean_importance, x_coords, y_coords, save_path):
    """Plot spatial heatmap of mean SHAP importance."""
    fig, ax = plt.subplots(figsize=(12, 4))
    data_2d = mean_importance.squeeze()  # (H, W)
    im = ax.pcolormesh(x_coords, y_coords, data_2d.T, cmap="inferno", shading="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mean SHAP Importance")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="Normalized importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved importance heatmap: {save_path}")


def plot_threshold_vs_nsensors(thresholds, n_pixels_list, total_features, save_path):
    """Plot number of retained sensors vs threshold."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thresholds, n_pixels_list, "o-", color="tab:blue", linewidth=2, markersize=6)
    ax1.set_xlabel("Threshold", fontsize=14)
    ax1.set_ylabel("Number of retained sensors", fontsize=14, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(total_features, color="gray", linestyle="--", alpha=0.5, label=f"Total features ({total_features})")

    ax2 = ax1.twinx()
    pct = [100.0 * n / total_features for n in n_pixels_list]
    ax2.plot(thresholds, pct, "s--", color="tab:orange", linewidth=1.5, markersize=5)
    ax2.set_ylabel("% of total features", fontsize=14, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved threshold curve: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Aggregate SHAP + threshold masks")
    parser.add_argument("--n_pixels", type=int, default=2, help="Obstacle expansion pixels")
    parser.add_argument("--block_size", type=int, default=1, help="Superpixel block size (1=pixel-level)")
    parser.add_argument("--n_snapshots", type=int, default=19, help="Number of SHAP snapshot files")
    parser.add_argument("--ncoalitions", type=int, default=500, help="Coalitions used in Phase 2a")
    parser.add_argument("--start", type=int, default=0, help="Start index used in Phase 2a")
    parser.add_argument("--stop", type=int, default=1899, help="Stop index used in Phase 2a")
    parser.add_argument("--step", type=int, default=100, help="Step used in Phase 2a")
    parser.add_argument("--strategy", type=str, default="mean", choices=["mean", "mean_plus_std"])
    parser.add_argument("--thresholds", type=str,
                        default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
                        help="Comma-separated threshold values")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory for SHAP npz files")
    parser.add_argument("--plots_dir", type=str, default=None, help="Override plots directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    plots_dir = Path(args.plots_dir) if args.plots_dir else PLOTS_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    # --- Step 1: Load SHAP values ---
    print("=" * 60)
    print("Step 1: Loading SHAP values")
    print("=" * 60)
    all_shaps, seg_tensor, sensor_indices, x_coords, y_coords = load_plif2d_shap_values(
        data_dir, args.n_pixels, args.block_size, args.n_snapshots, args.ncoalitions,
        args.start, args.stop, args.step
    )

    n_features = all_shaps.shape[0]

    # --- Step 2: Generate spatial SHAP matrices ---
    print("=" * 60)
    print("Step 2: Generating spatial SHAP matrices")
    print("=" * 60)

    seg_tensor_float = seg_tensor.astype(np.float64)
    all_shap_mats = generate_all_shap_mats(
        all_shaps=all_shaps,
        seg_tensor=seg_tensor_float,
        feature_wise_norm=True,
        plot_shap_mats=False,
    )
    print(f"all_shap_mats shape: {all_shap_mats.shape}")

    # Compute mean importance map
    mean_importance = mean_shap_vals(all_shap_mats)
    print(f"Mean importance shape: {mean_importance.shape}, "
          f"range: [{mean_importance.min():.4f}, {mean_importance.max():.4f}]")

    # --- Step 3: Save aggregated data ---
    print("=" * 60)
    print("Step 3: Saving aggregated SHAP data")
    print("=" * 60)
    agg_path = data_dir / f"shap-plif2d-npix{args.n_pixels}-block{args.block_size}-aggregated.npz"
    np.savez(
        agg_path,
        all_shaps=all_shaps,
        all_shap_mats=all_shap_mats,
        mean_importance=mean_importance,
        seg_tensor=seg_tensor,
        sensor_indices=sensor_indices,
        x_coords=x_coords,
        y_coords=y_coords,
        n_pixels=args.n_pixels,
        block_size=args.block_size,
        strategy=args.strategy,
    )
    print(f"Saved aggregated data: {agg_path}")

    # --- Step 4: Sweep thresholds ---
    print("=" * 60)
    print(f"Step 4: Sweeping {len(thresholds)} thresholds with strategy='{args.strategy}'")
    print("=" * 60)

    n_pixels_list = []
    for t in thresholds:
        final_sensor_array, reduced_shap_mask = create_shap_masks(
            all_shap_mats=all_shap_mats,
            strategy=args.strategy,
            threshold=t,
            plot_final_mask=False,
        )
        n_pix = int(np.count_nonzero(final_sensor_array == 1))
        n_pix_reduced = int(np.count_nonzero(reduced_shap_mask == 1))
        n_pixels_list.append(n_pix)

        mask_path = data_dir / f"shap-plif2d-npix{args.n_pixels}-block{args.block_size}-thresh{t:.2f}-strategy-{args.strategy}.npz"
        np.savez(
            mask_path,
            shap_mask=final_sensor_array,
            reduced_shap_mask=reduced_shap_mask,
            threshold=t,
            n_pixels=n_pix,
            n_pixels_reduced=n_pix_reduced,
            strategy=args.strategy,
            x_coords=x_coords,
            y_coords=y_coords,
            n_pixels_expansion=args.n_pixels,
            block_size=args.block_size,
        )
        print(f"  thresh={t:.2f}: {n_pix} pixels (reduced: {n_pix_reduced}) → {mask_path.name}")

    # --- Step 5: Plots ---
    print("=" * 60)
    print("Step 5: Generating plots")
    print("=" * 60)

    plot_importance_heatmap(
        mean_importance, x_coords, y_coords,
        plots_dir / "shap-plif2d-importance-heatmap.pdf",
    )

    plot_threshold_vs_nsensors(
        thresholds, n_pixels_list, n_features,
        plots_dir / "shap-plif2d-threshold-vs-nsensors.pdf",
    )

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Threshold':>10} {'Pixels':>8} {'% of total':>12}")
    print("-" * 32)
    for t, n in zip(thresholds, n_pixels_list):
        print(f"{t:>10.2f} {n:>8d} {100.0 * n / n_features:>11.1f}%")


if __name__ == "__main__":
    main()
