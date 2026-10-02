"""
Phase 2: Compute SHAP values for PLIF2D pixel-level sensor placement.

Each invocation processes a single snapshot from the (subsampled) training set.
Designed for SLURM array parallelization: one job per snapshot.

Usage:
    python phase2_compute_shap.py --snap_idx 0 --n_pixels 2 \
        --data_file data/plif_train_noaug_data_file.h5 \
        --start 0 --stop 1899 --step 10 --ncoalitions 500

The snap_idx selects which snapshot (after subsampling) to process.
"""

import argparse
import numpy as np
import torch
import sys
import os

from utils import (
    REPO_ROOT,
    load_plif2d_data,
    load_model_and_config,
    create_base_mask,
    get_sensor_pixel_indices,
    create_pixel_segmentation,
    create_superpixel_segmentation,
    normalize_PLIF,
    shap_osp_plif,
)
from libs import utils as lib_utils

sys.path.insert(0, str(REPO_ROOT))
import libs.shap.shap as shap


def parse_args():
    parser = argparse.ArgumentParser(description="PLIF2D SHAP value computation")
    parser.add_argument("--snap_idx", type=int, required=True,
                        help="Index into the subsampled snapshot array to process")
    parser.add_argument("--n_pixels", type=int, default=2,
                        help="Obstacle mask expansion in pixels (default: 2)")
    parser.add_argument("--block_size", type=int, default=1,
                        help="Superpixel block size (1=pixel-level, 2=2x2 blocks, etc.)")
    parser.add_argument("--data_file", type=str,
                        default="data/plif_train_noaug_data_file.h5",
                        help="Training data file (relative to REPO_ROOT)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index for subsampling train data")
    parser.add_argument("--stop", type=int, default=1899,
                        help="Stop index for subsampling train data")
    parser.add_argument("--step", type=int, default=10,
                        help="Step for subsampling train data (default: 10)")
    parser.add_argument("--ncoalitions", type=int, default=500,
                        help="Number of SHAP coalitions (default: 500)")
    parser.add_argument("--gen_seeds", type=str, default="0,1,2,3",
                        help="Comma-separated generation seeds for best-of-seeds")
    parser.add_argument("--outdir", type=str, default="results/phase2_shap/data",
                        help="Output directory for SHAP .npz files")
    return parser.parse_args()


def main():
    args = parse_args()
    gen_seeds = [int(s) for s in args.gen_seeds.split(",")]

    start_time = lib_utils.get_current_time()
    print(f"=== PLIF2D SHAP Phase 2 === snap_idx={args.snap_idx}, "
          f"ncoalitions={args.ncoalitions}, n_pixels={args.n_pixels}")
    print(f"Start time: {start_time}", flush=True)

    # --- Load model ---
    config, ddpm_model = load_model_and_config()

    # --- Load training data (no augmentation) ---
    data_file_abs = os.path.join(str(REPO_ROOT), args.data_file)
    data, alpha, beta, obstacle_mask, x_coords, y_coords, t = load_plif2d_data(
        config, datatype="Train", data_file=data_file_abs
    )
    print(f"Loaded train data: {data.shape}, alpha={alpha}, beta={beta}")

    # Subsample
    data = data[args.start:args.stop:args.step]
    print(f"Subsampled: {data.shape} (start={args.start}, stop={args.stop}, step={args.step})")

    # Validate snap_idx
    n_snapshots = data.shape[0]
    if args.snap_idx >= n_snapshots:
        print(f"ERROR: snap_idx={args.snap_idx} >= n_snapshots={n_snapshots}. Exiting.")
        sys.exit(1)

    # Select single snapshot
    snap_data = data[args.snap_idx:args.snap_idx + 1]  # (1, 1, H, W)
    print(f"Processing snapshot {args.snap_idx}/{n_snapshots}, shape: {snap_data.shape}")

    H, W = snap_data.shape[2], snap_data.shape[3]

    # --- Create masks and segmentation ---
    base_mask = create_base_mask(obstacle_mask, n_pixels=args.n_pixels)
    sensor_indices = get_sensor_pixel_indices(base_mask, obstacle_mask)
    n_sensor_pixels = len(sensor_indices)
    print(f"Base mask: n_pixels={args.n_pixels}, sensor pixels={n_sensor_pixels}")

    # Choose pixel-level or superpixel segmentation
    if args.block_size == 1:
        seg_tensor, unique_values = create_pixel_segmentation(sensor_indices, (H, W))
        print(f"Segmentation: pixel-level, {len(unique_values)} features")
    else:
        seg_tensor, unique_values = create_superpixel_segmentation(
            sensor_indices, (H, W), args.block_size
        )
        print(f"Segmentation: {args.block_size}x{args.block_size} blocks, "
              f"{len(unique_values)} features (from {n_sensor_pixels} pixels)")

    n_features = len(unique_values)

    # Zero mask (no sensors) and full mask (all sensors in base region)
    zero_mask = torch.zeros(1, H, W, dtype=torch.float32)
    full_mask = base_mask.clone().float()

    # SHAP feature vectors
    zzeros = np.zeros((1, n_features))
    zshap = np.ones((1, n_features))

    # --- Create SHAP instance ---
    gtruth_tensor = torch.from_numpy(snap_data).float().cuda()

    shap_inst = shap_osp_plif(
        model=ddpm_model,
        gtruth_tensor=gtruth_tensor,
        alpha=alpha,
        beta=beta,
        zero_mask=zero_mask,
        full_mask=full_mask,
        seg_tensor=seg_tensor,
        unique_segments=unique_values,
        zzero=zzeros,
        zshap=zshap,
        config=config,
    )

    # --- Run SHAP ---
    print(f"Running KernelExplainer with {args.ncoalitions} coalitions, "
          f"{n_features} features, gen_seeds={gen_seeds}...", flush=True)

    explainer = shap.KernelExplainer(
        lambda zs: shap_inst.model_function_mask_batches(zs, gen_seeds=gen_seeds),
        zzeros,
    )
    shap_values = explainer.shap_values(zshap, nsamples=args.ncoalitions)

    end_time = lib_utils.get_current_time()
    elapsed = lib_utils.get_time_difference(start_time, end_time)
    print(f"SHAP computation done in {elapsed}", flush=True)
    print(f"SHAP values shape: {shap_values.shape}")

    # --- Save results ---
    os.makedirs(args.outdir, exist_ok=True)
    case_name = (f"shap-plif2d-npix{args.n_pixels}-block{args.block_size}-snap{args.snap_idx}"
                 f"-ncoali{args.ncoalitions}-start{args.start}-stop{args.stop}-step{args.step}")
    outfile = os.path.join(args.outdir, f"{case_name}.npz")

    np.savez(
        outfile,
        shaps=shap_values,
        seg_tensor=seg_tensor.numpy(),
        sensor_indices=sensor_indices,
        n_pixels=args.n_pixels,
        block_size=args.block_size,
        snap_idx=args.snap_idx,
        start=args.start,
        stop=args.stop,
        step=args.step,
        ncoalitions=args.ncoalitions,
        gen_seeds=np.array(gen_seeds),
        alpha=alpha,
        beta=beta,
        x_coords=x_coords,
        y_coords=y_coords,
        zzeros=zzeros,
        zshap=zshap,
        elapsed_time=str(elapsed),
    )
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
