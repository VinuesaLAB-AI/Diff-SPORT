"""
Phase 1: Random Baseline for PLIF2D OSP.

Sweeps over increasing numbers of randomly placed sensors within the base
measurement region (obstacle boundary expanded by n_pixels). For each sensor
count and random seed, runs MAPGD conditional generation on the full test set
and records per-snapshot MSE.

Usage:
    python phase1_random_baseline.py [--n_pixels 3] [--n_sweep 10]
        [--max_pct 10] [--seeds 0,1,2] [--gen_seed 42]

Output:
    results/phase1_random_baseline.npz
"""

import argparse
import numpy as np
import os
import sys
from pathlib import Path

# Ensure imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    load_model_and_config,
    load_plif2d_data,
    create_base_mask,
    get_sensor_pixel_indices,
    create_random_sensor_mask,
    run_mapgd_inference,
    compute_mse_plif,
)
from libs.cond_gen_plif import normalize_PLIF, renormalize_PLIF
from libs import utils as lib_utils

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Random baseline for PLIF2D OSP")
    parser.add_argument("--n_pixels", type=int, default=3,
                        help="Obstacle expansion in pixels for base mask (default: 3)")
    parser.add_argument("--n_sweep", type=int, default=10,
                        help="Number of sensor-count sweep points (default: 10)")
    parser.add_argument("--min_pct", type=float, default=0.0,
                        help="Min sensor pixels as %% of total pixels (default: 0)")
    parser.add_argument("--max_pct", type=float, default=10.0,
                        help="Max sensor pixels as %% of total pixels (default: 10)")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated random mask seeds (default: 0,1,2)")
    parser.add_argument("--gen_seed", type=int, default=42,
                        help="MAPGD generation seed (default: 42)")
    parser.add_argument("--config", type=str, default="PLIF2D_128x32",
                        help="Config name (default: PLIF2D_128x32)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: results/phase1_random_baseline/ next to this script)")
    return parser.parse_args()


def main():
    args = parse_args()
    mask_seeds = [int(s) for s in args.seeds.split(",")]

    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "results" / "phase1_random_baseline"
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: Random Baseline — PLIF2D OSP")
    print("=" * 60, flush=True)

    config, ddpm_model = load_model_and_config(args.config)

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading test data...", flush=True)
    data, alpha, beta, obstacle_mask, x_coords, y_coords, t = load_plif2d_data(config, datatype="Test")
    N, C, H, W = data.shape
    total_pixels = H * W
    print(f"  Test data: {data.shape}, total pixels: {total_pixels}")

    # ------------------------------------------------------------------
    # 3. Create base mask and sensor indices
    # ------------------------------------------------------------------
    print(f"\n[2/5] Creating base mask (n_pixels={args.n_pixels})...", flush=True)
    base_mask = create_base_mask(obstacle_mask, n_pixels=args.n_pixels)
    sensor_indices = get_sensor_pixel_indices(base_mask, obstacle_mask)
    n_available = len(sensor_indices)
    print(f"  Available sensor pixels in base mask: {n_available}")

    # ------------------------------------------------------------------
    # 4. Compute sweep points
    # ------------------------------------------------------------------
    min_sensors = int(args.min_pct / 100.0 * total_pixels)
    max_sensors = int(args.max_pct / 100.0 * total_pixels)
    max_sensors = min(max_sensors, n_available)  

    n_sensors_list = np.linspace(min_sensors, max_sensors, args.n_sweep, dtype=int)
    n_sensors_list = np.unique(n_sensors_list)  # remove duplicates from rounding
    # Drop 0 if present — no sensors means no observations
    n_sensors_list = n_sensors_list[n_sensors_list > 0]

    print(f"\n[3/5] Sweep: {len(n_sensors_list)} points from {n_sensors_list[0]} to {n_sensors_list[-1]} sensors")
    print(f"  Sensor counts: {n_sensors_list.tolist()}")
    print(f"  Mask seeds: {mask_seeds}")
    print(f"  Gen seed: {args.gen_seed}", flush=True)

    # ------------------------------------------------------------------
    # 5. Normalize data once
    # ------------------------------------------------------------------
    print("\n[4/5] Normalizing data...", flush=True)
    data_norm = normalize_PLIF(data, alpha, beta)
    data_norm_tensor = torch.from_numpy(data_norm).float()
    print(f"  Normalized range: [{data_norm.min():.3f}, {data_norm.max():.3f}]")

    # ------------------------------------------------------------------
    # 6. Sweep
    # ------------------------------------------------------------------
    print(f"\n[5/5] Running sweep ({len(n_sensors_list)} x {len(mask_seeds)} = {len(n_sensors_list) * len(mask_seeds)} runs)...", flush=True)

    # Storage
    all_results = {
        "n_sensors_list": n_sensors_list,
        "mask_seeds": np.array(mask_seeds),
        "gen_seed": args.gen_seed,
        "n_pixels": args.n_pixels,
        "min_pct": args.min_pct,
        "max_pct": args.max_pct,
        "total_pixels": total_pixels,
        "n_available_sensors": n_available,
        "alpha": alpha,
        "beta": beta,
    }

    mean_mse_matrix = np.zeros((len(n_sensors_list), len(mask_seeds)))
    mean_mse_norm_matrix = np.zeros((len(n_sensors_list), len(mask_seeds)))
    per_snap_mse_all = {}

    total_runs = len(n_sensors_list) * len(mask_seeds)
    run_idx = 0
    sweep_start = lib_utils.get_current_time()

    for i, n_sensors in enumerate(n_sensors_list):
        for j, mseed in enumerate(mask_seeds):
            run_idx += 1
            run_start = lib_utils.get_current_time()

            # Create random sensor mask
            sensor_mask = create_random_sensor_mask(
                sensor_indices, n_sensors, (H, W), seed=mseed
            )
            actual_sensors = int(sensor_mask.sum().item())

            # Run MAPGD
            pred_norm = run_mapgd_inference(
                ddpm_model, data_norm_tensor, sensor_mask, config, gen_seed=args.gen_seed
            )

            # MSE in normalized space
            per_snap_norm, mean_mse_norm = compute_mse_plif(data_norm, pred_norm)
            mean_mse_norm_matrix[i, j] = mean_mse_norm
            per_snap_mse_all[f"snap_mse_norm_{n_sensors}_{mseed}"] = per_snap_norm

            # Renormalize to physical space
            pred = renormalize_PLIF(pred_norm, alpha, beta)

            # MSE in physical space
            per_snap, mean_mse = compute_mse_plif(data, pred)
            mean_mse_matrix[i, j] = mean_mse
            per_snap_mse_all[f"snap_mse_{n_sensors}_{mseed}"] = per_snap

            run_end = lib_utils.get_current_time()
            elapsed = lib_utils.get_time_difference(run_start, run_end)
            print(f"  [{run_idx}/{total_runs}] n_sensors={actual_sensors}, seed={mseed}, "
                  f"MSE_phys={mean_mse:.6e}, MSE_norm={mean_mse_norm:.6e}, time={elapsed}", flush=True)

    sweep_end = lib_utils.get_current_time()
    print(f"\nSweep completed in {lib_utils.get_time_difference(sweep_start, sweep_end)}")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    save_dict = {
        **all_results,
        "mean_mse_matrix": mean_mse_matrix,  # (n_sweep, n_seeds) physical space
        "mean_mse_norm_matrix": mean_mse_norm_matrix,  # (n_sweep, n_seeds) normalized space
        "base_mask": base_mask.cpu().numpy(),
        "obstacle_mask": obstacle_mask,
        "x_coords": x_coords,
        "y_coords": y_coords,
    }
    # Add per-snapshot MSEs
    save_dict.update(per_snap_mse_all)

    out_path = data_dir / "phase1_random_baseline.npz"
    np.savez(str(out_path), **save_dict)
    print(f"\nResults saved to: {out_path}")

    # Quick summary
    mean_across_seeds = np.mean(mean_mse_matrix, axis=1)
    std_across_seeds = np.std(mean_mse_matrix, axis=1)
    mean_norm_across_seeds = np.mean(mean_mse_norm_matrix, axis=1)
    std_norm_across_seeds = np.std(mean_mse_norm_matrix, axis=1)
    print("\nSummary (mean ± std across seeds):")
    for i, n_sensors in enumerate(n_sensors_list):
        pct = n_sensors / total_pixels * 100
        print(f"  {n_sensors:4d} sensors ({pct:5.1f}%): MSE_phys = {mean_across_seeds[i]:.6e} ± {std_across_seeds[i]:.6e}"
              f"  MSE_norm = {mean_norm_across_seeds[i]:.6e} ± {std_norm_across_seeds[i]:.6e}")


if __name__ == "__main__":
    main()
