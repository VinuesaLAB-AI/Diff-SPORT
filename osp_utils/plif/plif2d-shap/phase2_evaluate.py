"""
Phase 2d: Compare SHAP-optimized vs random baseline sensor placement.

Loads Phase 2c HDF5 predictions (per threshold × gen_seed), performs
best-of-seeds selection, and plots MSE vs number of sensors alongside
the Phase 1 random baseline.

No GPU required — runs on login node.

Usage:
    python phase2_evaluate.py
    python phase2_evaluate.py --thresholds 0.50,0.60,0.70,0.80 --gen_seeds 0,1,2,3
"""

import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from utils import compute_mse_plif
from configs.plot_config import basic_plt_setup
basic_plt_setup()

TOTAL_PIXELS = 128 * 32  # 4096


def load_shap_metrics(phase2c_dir, thresholds, gen_seeds, n_pixels, block_size=1, c_ref=1.0):
    """
    Load Phase 2c HDF5 files and compute MSE with best-of-seeds selection.

    Returns
    -------
    shap_n_pixels : list of int, number of sensor pixels per threshold
    shap_mean_mse : list of float, mean MSE (best-of-seeds) per threshold
    shap_std_mse : list of float, std MSE per threshold
    shap_best_errors : list of np.ndarray, per-snapshot best MSE
    """
    shap_n_pixels = []
    shap_mean_mse = []
    shap_std_mse = []
    shap_best_errors = []

    for thresh in thresholds:
        seed_errors = []  # list of (n_snapshots,) arrays
        seed_mean_mses = []  # scalar mean MSE per seed (for std across seeds)
        n_pix = None

        for seed in gen_seeds:
            fpath = phase2c_dir / f"shap-plif2d-npix{n_pixels}-block{block_size}-thresh{thresh:.2f}-seed{seed}.h5"
            if not fpath.exists():
                print(f"  WARNING: {fpath.name} not found, skipping")
                continue

            with h5py.File(fpath, "r") as hf:
                data = hf["data"][:]      # (N, 1, H, W)
                mapgd = hf["mapgd"][:]    # (N, 1, H, W)
                mask = hf["mask_tensor"][:]
                if n_pix is None:
                    n_pix = int(np.count_nonzero(mask == 1))

            per_snap, mean_mse = compute_mse_plif(data, mapgd, c_ref=c_ref)
            seed_errors.append(per_snap)
            seed_mean_mses.append(mean_mse)

        if not seed_errors:
            print(f"  No seeds found for threshold {thresh:.2f}, skipping")
            continue

        # Stack: (n_seeds, n_snapshots)
        seed_errors = np.stack(seed_errors, axis=0)

        # Best-of-seeds: per snapshot, pick seed with lowest MSE
        best_idx = np.argmin(seed_errors, axis=0)
        best_errors = seed_errors[best_idx, np.arange(seed_errors.shape[1])]

        # Std across seeds (matching random baseline convention):
        # each seed contributes one scalar mean MSE, std is across these scalars
        seed_mean_mses = np.array(seed_mean_mses)

        shap_n_pixels.append(n_pix)
        shap_mean_mse.append(float(np.mean(best_errors)))
        shap_std_mse.append(float(np.std(seed_mean_mses)))
        shap_best_errors.append(best_errors)

        print(f"  thresh={thresh:.2f}: {n_pix} pixels, "
              f"MSE(best-of-seeds)={np.mean(best_errors):.6e}, "
              f"std(across seeds)={np.std(seed_mean_mses):.6e} "
              f"(seeds used: {seed_errors.shape[0]})")

    return shap_n_pixels, shap_mean_mse, shap_std_mse, shap_best_errors


def load_random_baseline(phase1_path):
    """
    Load Phase 1 random baseline results.

    Returns
    -------
    rand_n_sensors : np.ndarray, (n_sweep,)
    rand_mean_mse : np.ndarray, (n_sweep,)
    rand_std_mse : np.ndarray, (n_sweep,)
    """
    results = np.load(str(phase1_path), allow_pickle=True)
    n_sensors_list = results["n_sensors_list"]
    mean_mse_matrix = results["mean_mse_matrix"]  # (n_sweep, n_seeds)

    rand_mean = np.mean(mean_mse_matrix, axis=1)
    rand_std = np.std(mean_mse_matrix, axis=1)

    return n_sensors_list, rand_mean, rand_std


def plot_comparison(rand_n, rand_mean, rand_std,
                    shap_n, shap_mean, shap_std,
                    out_path, log_scale=False):
    """Plot MSE vs % pixels for both methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    rand_pct = np.array(rand_n) / TOTAL_PIXELS * 100
    shap_pct = np.array(shap_n) / TOTAL_PIXELS * 100

    # Random baseline
    ax.errorbar(rand_pct, rand_mean, yerr=rand_std,
                fmt="o-", capsize=4, capthick=1.5, linewidth=2,
                markersize=6, color="#1f78b4", label="Random baseline",
                alpha=0.8, zorder=2)

    # SHAP optimized
    ax.errorbar(shap_pct, shap_mean, yerr=shap_std,
                fmt="s-", capsize=4, capthick=1.5, linewidth=2,
                markersize=7, color="#e31a1c", label="SHAP-optimized",
                alpha=0.9, zorder=3)

    ax.set_xlabel(r"Sensor pixels (\% of total)", fontsize=14)
    ax.set_ylabel(r"MSE ($C_{ref} = 1.0$)", fontsize=14)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3, which="both" if log_scale else "major")
    ax.tick_params(labelsize=12)

    # Secondary x-axis: absolute pixel count
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{int(t / 100 * TOTAL_PIXELS)}" for t in ticks])
    ax2.set_xlabel("Number of sensor pixels", fontsize=12)
    ax2.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2d: SHAP vs Random evaluation")
    parser.add_argument("--phase1_results", type=str,
                        default="results/phase1_random_baseline/data/phase1_random_baseline.npz")
    parser.add_argument("--phase2c_dir", type=str, default="results/phase2_cond_gen/data")
    parser.add_argument("--thresholds", type=str,
                        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--gen_seeds", type=str, default="0,1,2,3")
    parser.add_argument("--n_pixels", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=1, help="Superpixel block size (1=pixel-level)")
    parser.add_argument("--c_ref", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default="results/phase2_shap/data",
                        help="Output directory for metrics NPZ")
    parser.add_argument("--plots_dir", type=str, default="results/phase2_shap/plots",
                        help="Output directory for comparison plots")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    gen_seeds = [int(s) for s in args.gen_seeds.split(",")]

    phase1_path = Path(args.phase1_results)
    if not phase1_path.is_absolute():
        phase1_path = SCRIPT_DIR / phase1_path
    phase2c_dir = Path(args.phase2c_dir)
    if not phase2c_dir.is_absolute():
        phase2c_dir = SCRIPT_DIR / phase2c_dir
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = SCRIPT_DIR / data_dir
    plots_dir = Path(args.plots_dir)
    if not plots_dir.is_absolute():
        plots_dir = SCRIPT_DIR / plots_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: SHAP metrics ---
    print("=" * 60)
    print("Step 1: Computing SHAP metrics (best-of-seeds)")
    print("=" * 60)
    shap_n, shap_mean, shap_std, shap_best = load_shap_metrics(
        phase2c_dir, thresholds, gen_seeds, args.n_pixels, args.block_size, args.c_ref
    )

    # --- Step 2: Random baseline ---
    print("\n" + "=" * 60)
    print("Step 2: Loading random baseline")
    print("=" * 60)
    rand_n, rand_mean, rand_std = load_random_baseline(phase1_path)
    print(f"Random baseline: {len(rand_n)} sweep points, "
          f"{rand_n[0]}–{rand_n[-1]} pixels")

    # --- Step 3: Save SHAP metrics ---
    shap_results_path = data_dir / "phase2_shap_results.npz"
    np.savez(
        shap_results_path,
        thresholds=np.array(thresholds[:len(shap_n)]),
        n_pixels=np.array(shap_n),
        mean_mse=np.array(shap_mean),
        std_mse=np.array(shap_std),
        gen_seeds=np.array(gen_seeds),
        c_ref=args.c_ref,
    )
    print(f"\nSaved SHAP metrics: {shap_results_path}")

    # --- Step 4: Plots ---
    print("\n" + "=" * 60)
    print("Step 3: Generating comparison plots")
    print("=" * 60)

    plot_comparison(rand_n, rand_mean, rand_std,
                    shap_n, shap_mean, shap_std,
                    plots_dir / "phase2_shap_vs_random.pdf",
                    log_scale=False)

    plot_comparison(rand_n, rand_mean, rand_std,
                    shap_n, shap_mean, shap_std,
                    plots_dir / "phase2_shap_vs_random_log.pdf",
                    log_scale=True)

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("Summary: SHAP-optimized results")
    print("=" * 60)
    print(f"{'Threshold':>10} {'Pixels':>8} {'% total':>9} {'Mean MSE':>12} {'Std MSE':>12}")
    print("-" * 55)
    for i in range(len(shap_n)):
        pct = shap_n[i] / TOTAL_PIXELS * 100
        print(f"{thresholds[i]:>10.2f} {shap_n[i]:>8d} {pct:>8.2f}% {shap_mean[i]:>12.6e} {shap_std[i]:>12.6e}")

    print(f"\nRandom baseline range: {rand_n[0]}–{rand_n[-1]} pixels, "
          f"MSE range: {rand_mean[-1]:.6e}–{rand_mean[0]:.6e}")


if __name__ == "__main__":
    main()
