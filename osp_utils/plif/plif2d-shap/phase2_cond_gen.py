"""
Phase 2c: Conditional generation with SHAP-optimized threshold masks.

Loads a threshold mask from Phase 2b, runs MAPGD on PLIF2D test data,
and saves ground truth + predictions as HDF5.

One invocation = one (threshold, gen_seed) pair.
Parallelized via SLURM array job (see job_phase2_cond_gen.sh).

Usage:
    python phase2_cond_gen.py --threshold 0.50 --gen_seed 0 --strategy mean --n_pixels 2
"""

import argparse
import numpy as np
import torch
import h5py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from libs.cond_gen_plif import normalize_PLIF, renormalize_PLIF
from utils import load_model_and_config, load_plif2d_data, run_mapgd_inference

SCRIPT_DIR = Path(__file__).resolve().parent
SHAP_DIR = SCRIPT_DIR / "results" / "phase2_shap" / "data"
OUT_DIR = SCRIPT_DIR / "results" / "phase2_cond_gen" / "data"


def main():
    parser = argparse.ArgumentParser(description="Phase 2c: SHAP mask conditional generation")
    parser.add_argument("--threshold", type=float, required=True, help="SHAP threshold (e.g. 0.50)")
    parser.add_argument("--gen_seed", type=int, required=True, help="Generation seed")
    parser.add_argument("--strategy", type=str, default="mean", help="Aggregation strategy used in Phase 2b")
    parser.add_argument("--n_pixels", type=int, default=2, help="Obstacle expansion used in Phase 2a/2b")
    parser.add_argument("--block_size", type=int, default=1, help="Superpixel block size (1=pixel-level)")
    parser.add_argument("--config", type=str, default="PLIF2D_128x32")
    parser.add_argument("--shap_dir", type=str, default=None, help="Override SHAP mask directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    shap_dir = Path(args.shap_dir) if args.shap_dir else SHAP_DIR
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load threshold mask from Phase 2b ---
    mask_path = shap_dir / f"shap-plif2d-npix{args.n_pixels}-block{args.block_size}-thresh{args.threshold:.2f}-strategy-{args.strategy}.npz"
    if not mask_path.exists():
        print(f"ERROR: Mask file not found: {mask_path}")
        sys.exit(1)

    mask_data = np.load(mask_path)
    mask_tensor = torch.from_numpy(mask_data["shap_mask"].astype(np.float32))  # (1, 128, 32)
    n_mask_pixels = int(np.count_nonzero(mask_data["shap_mask"] == 1))
    print(f"Loaded mask: thresh={args.threshold:.2f}, {n_mask_pixels} pixels, shape={mask_tensor.shape}")

    # --- Load model ---
    print("Loading model...")
    config, ddpm_model = load_model_and_config(args.config)

    # --- Load test data ---
    print("Loading test data...")
    data, alpha, beta, obstacle_mask, x_coords, y_coords, t = load_plif2d_data(config, datatype="Test")
    print(f"Test data: {data.shape}, alpha={alpha}, beta={beta}")

    # --- Normalize ---
    data_norm = normalize_PLIF(data, alpha, beta)
    data_norm_tensor = torch.from_numpy(data_norm).float()

    # --- Run MAPGD ---
    print(f"Running MAPGD: T_sub={config.sampling.T_sub_mapgd}, gditer={config.sampling.gditer_mapgd}, "
          f"batch_size={config.sampling.batch_size}, gen_seed={args.gen_seed}")
    pred_norm = run_mapgd_inference(ddpm_model, data_norm_tensor, mask_tensor, config, gen_seed=args.gen_seed)

    # --- Renormalize predictions ---
    pred_raw = renormalize_PLIF(pred_norm, alpha, beta)
    print(f"Predictions: {pred_raw.shape}")

    # --- Save HDF5 ---
    out_path = out_dir / f"shap-plif2d-npix{args.n_pixels}-block{args.block_size}-thresh{args.threshold:.2f}-seed{args.gen_seed}.h5"
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("data", data=data)
        hf.create_dataset("mapgd", data=pred_raw)
        hf.create_dataset("mask_tensor", data=mask_tensor.numpy())
        hf.create_dataset("x", data=x_coords)
        hf.create_dataset("y", data=y_coords)
        hf.create_dataset("t", data=t)
        hf.create_dataset("threshold", data=args.threshold)
        hf.create_dataset("gen_seed", data=args.gen_seed)
        hf.create_dataset("block_size", data=args.block_size)
        hf.create_dataset("n_pixels", data=n_mask_pixels)
        hf.create_dataset("strategy", data=args.strategy)
        hf.create_dataset("T_sub_mapgd", data=config.sampling.T_sub_mapgd)
        hf.create_dataset("gditer_mapgd", data=config.sampling.gditer_mapgd)

    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
