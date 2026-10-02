#!/usr/bin/env python3
"""
PLIF2D Evaluation Pipeline

Supports two evaluation modes:
1. Unconditional: Compare generated samples vs ground truth (visual + basic stats)
2. Conditional: Evaluate MAPGD inpainting quality (MSE, error fields, error stats)

Usage:
    # Unconditional mode
    python eval-plif2d.py --mode unconditional --config PLIF2D_128x32 --n-samples 4

    # Conditional mode
    python eval-plif2d.py --mode conditional --config PLIF2D_128x32 \
        --cond-file cond_gen_PLIF-n_pixel_mask-n10-ckpt1000-T20-gd50-100snaps-mseed0-gseed0.h5
"""

import sys
import os
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Ensure imports work regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from libs.dataset import normalize_distribution, denormalize_distribution
from libs import runner
from configs import PLIF2D_128x32
from configs.plot_config import basic_plt_setup

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Apply consistent plot formatting
basic_plt_setup()

# Reference value for MSE normalization
C_REF = 1.0


# =============================================================================
# Configuration
# =============================================================================

CONFIG_DICT = {
    'PLIF2D_128x32': PLIF2D_128x32.config_dict,
}

def is_minmax_dataset(config):
    """True if the dataset config sets data_min/data_max (min-max normalization).
    PLIF2D sets neither and uses the log1p scheme."""
    return hasattr(config.dataset, "data_min") and hasattr(config.dataset, "data_max")


def field_label(config):
    """Field variable name for plot titles/labels. Prefer the dataset config's
    explicit field_symbol if set; fall back to the old is_minmax-based guess
    for configs that predate field_symbol."""
    if hasattr(config.dataset, "field_symbol"):
        return config.dataset.field_symbol
    return "u'" if is_minmax_dataset(config) else "C'"


def generic_normalize(data, gt_meta):
    """Forward transform: physical units -> [-1, 1], dispatched by dataset type."""
    if gt_meta.get("is_minmax"):
        eps = 1e-9
        scaled = (data - gt_meta["data_min"]) / (gt_meta["data_max"] - gt_meta["data_min"] + eps)
        return (2 * scaled) - 1
    return normalize_distribution(data, alpha=gt_meta["alpha"], beta=gt_meta["beta"])


def generic_denormalize(data_norm, gt_meta):
    """Inverse transform: [-1, 1] -> physical units, dispatched by dataset type."""
    if gt_meta.get("is_minmax"):
        eps = 1e-9
        scaled = (np.clip(data_norm, -1, 1) + 1) / 2
        return gt_meta["data_min"] + scaled * (gt_meta["data_max"] - gt_meta["data_min"] + eps)
    return denormalize_distribution(np.clip(data_norm, -1, 1), alpha=gt_meta["alpha"], beta=gt_meta["beta"])


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_ground_truth(config):
    """
    Load ground truth concentration fluctuations from dataset.

    Parameters
    ----------
    config : namespace
        Configuration object with dataset paths

    Returns
    -------
    dict with keys: 'data', 'x_coords', 'y_coords', 'mask', 'alpha', 'beta'
    """
    data_file = config.dataset.data_file

    with h5py.File(data_file, 'r') as f:
        data = f['data'][:]  # (N, 1, H, W) - raw fluctuations
        x_coords = f['x_coords'][:]
        y_coords = f['y_coords'][:]
        mask = f['mask'][:] if 'mask' in f else None

    is_minmax = is_minmax_dataset(config)

    print(f"[GT] Loaded from: {data_file}")
    print(f"[GT] Shape: {data.shape}, dtype: {data.dtype}")

    result = {
        'data': data,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'mask': mask,
        'is_minmax': is_minmax,
    }

    if is_minmax:
        result['data_min'] = float(np.min(data))
        result['data_max'] = float(np.max(data))
        print(f"[GT] Normalization: min-max, data_min={result['data_min']:.4f}, data_max={result['data_max']:.4f}")
    else:
        result['alpha'] = config.dataset.alpha
        result['beta'] = config.dataset.beta
        print(f"[GT] Normalization: log1p, alpha={result['alpha']}, beta={result['beta']}")

    return result


def load_unconditional_predictions(config):
    """
    Load unconditionally generated samples.

    Parameters
    ----------
    config : namespace
        Configuration object with prediction file path

    Returns
    -------
    np.ndarray
        Predictions in normalized space, shape (N, H, W)
    """
    pred_file = config.dataset.pred_data_file

    with h5py.File(pred_file, 'r') as f:
        seed_keys = sorted([k for k in f.keys() if k.startswith('seed_')])
        print(f"[Uncond] Found {len(seed_keys)} seed datasets: {seed_keys}")

        pred_list = []
        for seed_key in seed_keys:
            pred_list.append(f[seed_key][:, 0, :, :])  # (N, H, W)

        pred_norm = np.concatenate(pred_list, axis=0)

    print(f"[Uncond] Loaded from: {pred_file}")
    print(f"[Uncond] Shape: {pred_norm.shape}")

    return pred_norm


def load_conditional_results(cond_file):
    """
    Load conditional generation results (MAPGD only).

    Parameters
    ----------
    cond_file : str or Path
        Path to conditional generation HDF5 file

    Returns
    -------
    dict with keys: 'gtruth', 'mapgd', 'mask_tensor', 'x', 'y', 't', 'alpha', 'beta'
    """
    with h5py.File(cond_file, 'r') as f:
        gtruth = f['data'][:]  # (N, 1, H, W)
        mapgd = f['mapgd'][:]  # (M, 1, H, W) - may differ from N
        mask_tensor = f['mask_tensor'][:]  # (1, H, W)
        obstacle_mask = f['obstacle_mask'][:] if 'obstacle_mask' in f else None
        x = f['x'][:]
        y = f['y'][:]
        t = f['t'][:]

        if 'alpha' in f and 'beta' in f:
            norm_info = {'alpha': float(f['alpha'][()]), 'beta': float(f['beta'][()])}
        elif 'data_min' in f and 'data_max' in f:
            norm_info = {'data_min': float(f['data_min'][()]), 'data_max': float(f['data_max'][()])}
        else:
            norm_info = {}

  
    n_mapgd = mapgd.shape[0]
    if gtruth.shape[0] > n_mapgd:
        print(f"[Cond] Warning: GT has {gtruth.shape[0]} samples, MAPGD has {n_mapgd}. Truncating GT.")
        gtruth = gtruth[:n_mapgd]
        t = t[:n_mapgd]

    print(f"[Cond] Loaded from: {cond_file}")
    print(f"[Cond] GT shape: {gtruth.shape}, MAPGD shape: {mapgd.shape}")
    print(f"[Cond] Mask shape: {mask_tensor.shape}")
    print(f"[Cond] Normalization: {norm_info}")

    return {
        'gtruth': gtruth,
        'mapgd': mapgd,
        'mask_tensor': mask_tensor,
        'obstacle_mask': obstacle_mask,
        'x': x,
        'y': y,
        't': t,
        **norm_info,
    }


# =============================================================================
# MSE Computation
# =============================================================================

def compute_mse(gtruth, pred, c_ref=C_REF):
    """
    Compute MSE between ground truth and prediction.

    MSE = mean((C'_gt - C'_pred)^2) / C_ref^2

    Parameters
    ----------
    gtruth : np.ndarray
        Ground truth, shape (N, 1, H, W) or (N, H, W)
    pred : np.ndarray
        Prediction, shape (N, 1, H, W) or (N, H, W)
    c_ref : float
        Reference concentration for normalization

    Returns
    -------
    dict with keys:
        'per_snapshot': (N,) MSE per snapshot
        'time_averaged': (H, W) time-averaged MSE field
        'mean': scalar mean MSE
        'error_field': (N, H, W) squared error field
    """
    # Ensure 3D shape (N, H, W)
    if gtruth.ndim == 4:
        gtruth = gtruth[:, 0, :, :]
    if pred.ndim == 4:
        pred = pred[:, 0, :, :]

    # Squared error field
    error_field = (gtruth - pred) ** 2 / (c_ref ** 2)

    # Per-snapshot MSE (spatial mean)
    per_snapshot = np.mean(error_field, axis=(1, 2))

    # Time-averaged MSE field
    time_averaged = np.mean(error_field, axis=0)

    # Overall mean MSE
    mean_mse = np.mean(error_field)

    return {
        'per_snapshot': per_snapshot,
        'time_averaged': time_averaged,
        'mean': mean_mse,
        'error_field': error_field
    }


def compute_basic_stats(data):
    """
    Compute basic statistics for concentration field.

    Parameters
    ----------
    data : np.ndarray
        Data array, shape (N, 1, H, W) or (N, H, W)

    Returns
    -------
    dict with keys: 'mean', 'std', 'variance', 'min', 'max'
    """
    if data.ndim == 4:
        data = data[:, 0, :, :]

    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'variance': np.var(data),
        'min': np.min(data),
        'max': np.max(data)
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_field(data, x_coords, y_coords, title, output_path, cmap='RdBu_r',
               vmin=None, vmax=None, symmetric=False, cbar_label="C'"):
    """
    Plot a single concentration field and save as .png.

    Parameters
    ----------
    data : np.ndarray
        2D field, shape (H, W)
    x_coords, y_coords : np.ndarray
        Coordinate arrays
    title : str
        Plot title
    output_path : str or Path
        Output file path
    cmap : str
        Colormap
    vmin, vmax : float
        Color limits
    symmetric : bool
        If True, make colorbar symmetric around zero
    cbar_label : str
        Colorbar label
    """
    extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

    if symmetric and vmin is None and vmax is None:
        max_abs = np.max(np.abs(data))
        vmin, vmax = -max_abs, max_abs

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(data.T, origin='lower', aspect='auto', cmap=cmap,
                   extent=extent, vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison_row(gt, pred, error, x_coords, y_coords, sample_idx,
                        output_path, vmin_field=None, vmax_field=None,
                        vmax_error=None, field_name="C'"):
    """
    Plot GT, Pred, Error side by side for a single sample.

    Parameters
    ----------
    gt, pred, error : np.ndarray
        2D fields, shape (H, W)
    x_coords, y_coords : np.ndarray
        Coordinate arrays
    sample_idx : int
        Sample index (for title)
    output_path : str or Path
        Output file path
    """
    extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

    # Determine color limits
    if vmin_field is None or vmax_field is None:
        max_abs = max(np.max(np.abs(gt)), np.max(np.abs(pred)))
        vmin_field, vmax_field = -max_abs, max_abs

    if vmax_error is None:
        vmax_error = np.percentile(error, 99)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # GT
    im0 = axes[0].imshow(gt.T, origin='lower', aspect='auto', cmap='RdBu_r',
                         extent=extent, vmin=vmin_field, vmax=vmax_field)
    axes[0].set_title(f"{field_name}_gt (Sample {sample_idx})")
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Pred
    im1 = axes[1].imshow(pred.T, origin='lower', aspect='auto', cmap='RdBu_r',
                         extent=extent, vmin=vmin_field, vmax=vmax_field)
    axes[1].set_title(f"{field_name}_pred (MAPGD)")
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Error
    im2 = axes[2].imshow(error.T, origin='lower', aspect='auto', cmap='hot',
                         extent=extent, vmin=0, vmax=vmax_error)
    axes[2].set_title(f"Squared Error")
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_evolution(mse_per_snapshot, time_axis, output_path):
    """
    Plot MSE evolution over time/snapshots.

    Parameters
    ----------
    mse_per_snapshot : np.ndarray
        (N,) MSE values per snapshot
    time_axis : np.ndarray
        (N,) time or snapshot indices
    output_path : str or Path
        Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_axis, mse_per_snapshot, 'b-', linewidth=1.5, label='MSE')
    ax.axhline(np.mean(mse_per_snapshot), color='r', linestyle='--',
               label=f'Mean: {np.mean(mse_per_snapshot):.4e}')
    ax.set_xlabel('Snapshot')
    ax.set_ylabel('MSE')
    ax.set_title('Error Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_pdf(mse_per_snapshot, output_path):
    """
    Plot histogram/PDF of MSE values.

    Parameters
    ----------
    mse_per_snapshot : np.ndarray
        (N,) MSE values per snapshot
    output_path : str or Path
        Output file path
    """
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(8, 6))

    # Histogram
    ax.hist(mse_per_snapshot, bins=50, density=True, alpha=0.7, color='steelblue',
            edgecolor='white', label='Histogram')

    # KDE
    kde = gaussian_kde(mse_per_snapshot, bw_method=0.3)
    x_kde = np.linspace(mse_per_snapshot.min(), mse_per_snapshot.max(), 200)
    ax.plot(x_kde, kde(x_kde), 'r-', linewidth=2, label='KDE')

    ax.axvline(np.mean(mse_per_snapshot), color='k', linestyle='--',
               label=f'Mean: {np.mean(mse_per_snapshot):.4e}')

    ax.set_xlabel('MSE')
    ax.set_ylabel('Density')
    ax.set_title('Error PDF')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_unconditional_comparison(gtruth_denorm, pred_norm, x_coords, y_coords,
                                  gt_meta, n_samples, seed, output_dir, field_name="C'"):
    """
    Plot visual comparison for unconditional generation.

    Parameters
    ----------
    gtruth_denorm : np.ndarray
        Ground truth in denormalized space, shape (N, 1, H, W)
    pred_norm : np.ndarray
        Predictions in normalized space, shape (N, H, W)
    gt_meta : dict
        Metadata from load_ground_truth(), used to dispatch normalization scheme
    """
    np.random.seed(seed)

    # Extract and transform
    gt_denorm = gtruth_denorm[:, 0, :, :]  # (N, H, W)
    gt_norm = generic_normalize(gt_denorm, gt_meta)
    pred_norm_clipped = np.clip(pred_norm, -1, 1)
    pred_denorm = generic_denormalize(pred_norm_clipped, gt_meta)

    # Select random samples
    n_gt = gt_denorm.shape[0]
    n_pred = pred_denorm.shape[0]
    gt_indices = np.sort(np.random.choice(n_gt, size=n_samples, replace=False))
    pred_indices = np.sort(np.random.choice(n_pred, size=n_samples, replace=False))

    # Color limits
    denorm_max = np.percentile(np.abs(gt_denorm), 99.5)

    for i, (gt_idx, pred_idx) in enumerate(zip(gt_indices, pred_indices)):
        # Denormalized comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        im0 = axes[0].imshow(gt_denorm[gt_idx].T, origin='lower', aspect='auto',
                             cmap='RdBu_r', extent=extent, vmin=-denorm_max, vmax=denorm_max)
        axes[0].set_title(f"{field_name}_gt (Sample {gt_idx})")
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(pred_denorm[pred_idx].T, origin='lower', aspect='auto',
                             cmap='RdBu_r', extent=extent, vmin=-denorm_max, vmax=denorm_max)
        axes[1].set_title(f"{field_name}_pred (Sample {pred_idx})")
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(output_dir / f'uncond_comparison_denorm_{i:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Normalized comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        im0 = axes[0].imshow(gt_norm[gt_idx].T, origin='lower', aspect='auto',
                             cmap='RdBu_r', extent=extent, vmin=-1, vmax=1)
        axes[0].set_title(f"{field_name}_gt normalized (Sample {gt_idx})")
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(pred_norm_clipped[pred_idx].T, origin='lower', aspect='auto',
                             cmap='RdBu_r', extent=extent, vmin=-1, vmax=1)
        axes[1].set_title(f"{field_name}_pred normalized (Sample {pred_idx})")
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(output_dir / f'uncond_comparison_norm_{i:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {n_samples} unconditional comparison figures to {output_dir}")


# =============================================================================
# Evaluation Modes
# =============================================================================

def run_unconditional_evaluation(config, n_samples, seed, output_dir):
    """
    Run unconditional evaluation mode.

    - Visual comparison (GT vs Pred, denorm/norm)
    - Basic statistics computation
    """
    print("\n" + "="*60)
    print("UNCONDITIONAL EVALUATION MODE")
    print("="*60)

    # Load data
    gt_data = load_ground_truth(config)
    pred_norm = load_unconditional_predictions(config)

    # Compute basic stats
    print("\n--- Basic Statistics ---")
    gt_stats = compute_basic_stats(gt_data['data'])
    pred_denorm = generic_denormalize(pred_norm, gt_data)
    pred_stats = compute_basic_stats(pred_denorm)

    print(f"GT field:   mean={gt_stats['mean']:.4e}, std={gt_stats['std']:.4e}, var={gt_stats['variance']:.4e}")
    print(f"Pred field: mean={pred_stats['mean']:.4e}, std={pred_stats['std']:.4e}, var={pred_stats['variance']:.4e}")

    # Visual comparison
    print("\n--- Generating Visual Comparisons ---")
    plot_unconditional_comparison(
        gtruth_denorm=gt_data['data'],
        pred_norm=pred_norm,
        x_coords=gt_data['x_coords'],
        y_coords=gt_data['y_coords'],
        gt_meta=gt_data,
        n_samples=n_samples,
        seed=seed,
        output_dir=output_dir,
        field_name=field_label(config)
    )

    print("\nUnconditional evaluation complete.")


def run_conditional_evaluation(config, cond_file, n_samples, seed, output_dir, field_name="C'"):
    """
    Run conditional evaluation mode (MAPGD only).

    - MSE computation (per-snapshot, time-averaged, mean)
    - Visual comparisons: C'_gt, C'_pred, Error
    - Error evolution plot
    - Error PDF/histogram
    """
    print("\n" + "="*60)
    print("CONDITIONAL EVALUATION MODE (MAPGD)")
    print("="*60)

    # Load conditional results
    cond_data = load_conditional_results(cond_file)

    gtruth = cond_data['gtruth']
    mapgd = cond_data['mapgd']
    x = cond_data['x']
    y = cond_data['y']
    t = cond_data['t']

    # Compute MSE
    print("\n--- MSE Computation ---")
    mse_results = compute_mse(gtruth, mapgd, c_ref=C_REF)

    print(f"Mean MSE: {mse_results['mean']:.6e}")
    print(f"MSE per snapshot: min={mse_results['per_snapshot'].min():.6e}, "
          f"max={mse_results['per_snapshot'].max():.6e}")

    # Select samples for visualization
    np.random.seed(seed)
    n_total = gtruth.shape[0]
    sample_indices = np.sort(np.random.choice(n_total, size=min(n_samples, n_total), replace=False))

    # Plot visual comparisons
    print("\n--- Generating Visual Comparisons ---")

    # Compute global color limits
    gt_3d = gtruth[:, 0, :, :] if gtruth.ndim == 4 else gtruth
    pred_3d = mapgd[:, 0, :, :] if mapgd.ndim == 4 else mapgd
    field_max = np.percentile(np.abs(np.concatenate([gt_3d, pred_3d])), 99.5)
    error_max = np.percentile(mse_results['error_field'], 99)

    for i, idx in enumerate(sample_indices):
        gt_snap = gt_3d[idx]
        pred_snap = pred_3d[idx]
        error_snap = mse_results['error_field'][idx]

        plot_comparison_row(
            gt=gt_snap,
            pred=pred_snap,
            error=error_snap,
            x_coords=x,
            y_coords=y,
            sample_idx=idx,
            output_path=output_dir / f'cond_comparison_{i:02d}_snap{idx}.png',
            vmin_field=-field_max,
            vmax_field=field_max,
            vmax_error=error_max,
            field_name=field_name
        )

    # Plot time-averaged error field
    print("\n--- Time-Averaged Error Field ---")
    plot_field(
        data=mse_results['time_averaged'],
        x_coords=x,
        y_coords=y,
        title='Time-Averaged MSE',
        output_path=output_dir / 'cond_time_avg_error.png',
        cmap='hot',
        vmin=0,
        vmax=np.percentile(mse_results['time_averaged'], 99),
        cbar_label='MSE'
    )

    # Plot error evolution
    print("\n--- Error Evolution ---")
    plot_error_evolution(
        mse_per_snapshot=mse_results['per_snapshot'],
        time_axis=np.arange(len(mse_results['per_snapshot'])),
        output_path=output_dir / 'cond_error_evolution.png'
    )

    # Plot error PDF
    print("\n--- Error PDF ---")
    plot_error_pdf(
        mse_per_snapshot=mse_results['per_snapshot'],
        output_path=output_dir / 'cond_error_pdf.png'
    )

    print("\nConditional evaluation complete.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PLIF2D Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unconditional mode
  python eval-plif2d.py --mode unconditional --config PLIF2D_128x32 --n-samples 4

  # Conditional mode
  python eval-plif2d.py --mode conditional --config PLIF2D_128x32 \\
      --cond-file cond_gen_PLIF-n_pixel_mask-n10-ckpt1000-T20-gd50-100snaps-mseed0-gseed0.h5
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['unconditional', 'conditional'],
        help='Evaluation mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='PLIF2D_128x32',
        help='Configuration name (default: PLIF2D_128x32)'
    )
    parser.add_argument(
        '--cond-file',
        type=str,
        default=None,
        help='Conditional generation HDF5 file (required for conditional mode)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=4,
        help='Number of samples to visualize (default: 4)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: evaluate_utils/results/<config>/<mode>)'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config not in CONFIG_DICT:
        print(f"Error: Unknown config '{args.config}'. Available: {list(CONFIG_DICT.keys())}")
        sys.exit(1)

    config = runner.dict2namespace(CONFIG_DICT[args.config])

    # Run evaluation
    if args.mode == 'unconditional':
        # Setup output directory: results/<config>/<mode>/
        if args.output_dir is None:
            output_dir = REPO_ROOT / 'evaluate_utils' / 'results' / args.config / args.mode
        else:
            output_dir = Path(args.output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        run_unconditional_evaluation(
            config=config,
            n_samples=args.n_samples,
            seed=args.seed,
            output_dir=output_dir
        )
    elif args.mode == 'conditional':
        if args.cond_file is None:
            print("Error: --cond-file is required for conditional mode")
            sys.exit(1)

        # Resolve cond-file path
        cond_file = Path(args.cond_file)
        if not cond_file.is_absolute():
            # Try relative to cond_gen_dir
            cond_file = Path(config.eval.cond_gen_dir) / args.cond_file

        if not cond_file.exists():
            print(f"Error: Conditional file not found: {cond_file}")
            sys.exit(1)


        cond_filename = cond_file.stem  
        
        for prefix in ('cond_gen_PLIF-', 'cond_gen_PIV-'):
            if cond_filename.startswith(prefix):
                cond_filename_stripped = cond_filename[len(prefix):]
                break
        else:
            cond_filename_stripped = cond_filename
        parts = cond_filename_stripped.split('-')
        if len(parts) >= 2:
            cond_subdir = f"{parts[0]}-{parts[1]}"  
        else:
            cond_subdir = cond_filename

        if args.output_dir is None:
            output_dir = REPO_ROOT / 'evaluate_utils' / 'results' / args.config / args.mode / cond_subdir
        else:
            output_dir = Path(args.output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        run_conditional_evaluation(
            config=config,
            cond_file=cond_file,
            n_samples=args.n_samples,
            seed=args.seed,
            output_dir=output_dir,
            field_name=field_label(config)
        )


if __name__ == '__main__':
    main()
