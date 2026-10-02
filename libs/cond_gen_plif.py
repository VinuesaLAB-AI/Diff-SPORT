"""
Helper functions for PLIF2D conditional generation.

All PLIF2D-specific functions are contained here to keep them separate
from the OneObs2D conditional generation utilities.
"""

import numpy as np
import torch
import h5py
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# Ensure imports work regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from libs.dataset import normalize_distribution, denormalize_distribution
from libs import dataset


def normalize_PLIF(data, alpha, beta):
    """
    Normalize PLIF2D data using logarithmic distribution normalization.

    Parameters:
    -----------
    data : np.ndarray or torch.Tensor
        Input data with shape (N, 1, H, W)
    alpha : float
        Scale parameter for log1p normalization (controls compression of large values)
    beta : float
        Division factor to bring normalized values to [-1, 1] range

    Returns:
    --------
    np.ndarray or torch.Tensor
        Normalized data in range approximately [-1, 1]
    """
    return normalize_distribution(data, alpha=alpha, beta=beta)


def renormalize_PLIF(data_norm, alpha, beta):
    """
    Renormalize PLIF2D data back to original scale.

    Parameters:
    -----------
    data_norm : np.ndarray or torch.Tensor
        Normalized data with shape (N, 1, H, W) in range [-1, 1]
    alpha : float
        Scale parameter used in normalization
    beta : float
        Division factor used in normalization

    Returns:
    --------
    np.ndarray or torch.Tensor
        Denormalized data in original range
    """
    return denormalize_distribution(data_norm, alpha=alpha, beta=beta)


def get_data_PLIF(config, datatype="Test"):
    """
    Load PLIF2D data using the standardized get_dataset function.

    Parameters:
    -----------
    config : namespace
        Configuration object with dataset and model parameters
    datatype : str
        Label used in the log message only. The file read is always
        config.dataset.data_file; to load the test set, point data_file at
        config.dataset.test_data_file before calling.

    Returns:
    --------
    tuple: (data, alpha, beta, obstacle_mask, x_coords, y_coords, t)
        - data: (N, 1, H, W) raw fluctuation field (not normalized)
        - alpha, beta: normalization parameters from config
        - obstacle_mask: (H, W) boolean, True=valid region, False=obstacle
        - x_coords: (H,) spatial x coordinates
        - y_coords: (W,) spatial y coordinates
        - t: (N,) time indices (if available)
    """
    # Use standard get_dataset function
    dataset_obj = dataset.get_dataset(config)

    # Get raw data (without normalization for conditional generation)
    # We'll normalize manually in the script
    data = dataset_obj.data  # (N, 1, H, W)
    x_coords = dataset_obj.x  # (H,)
    y_coords = dataset_obj.y  # (W,)
    obstacle_mask = dataset_obj.mask  # (H, W) boolean

    # Get normalization parameters
    alpha = dataset_obj.alpha
    beta = dataset_obj.beta


    t = np.arange(len(data))

    print(f"Loaded PLIF2D {datatype} data: shape={data.shape}, dtype={data.dtype}")
    print(f"Normalization params: alpha={alpha}, beta={beta}")
    print(f"Obstacle mask: {np.sum(~obstacle_mask)} obstacle pixels, {np.sum(obstacle_mask)} valid pixels")

    return data, alpha, beta, obstacle_mask, x_coords, y_coords, t


def get_obstacle_from_dataset(data_file):
    """
    Load obstacle boolean mask directly from PLIF2D HDF5 file.

    Parameters:
    -----------
    data_file : str
        Path to HDF5 file

    Returns:
    --------
    np.ndarray
        Boolean array (H, W) where False=obstacle, True=valid flow region
    """
    with h5py.File(data_file, 'r') as f:
        obstacle_mask = np.asarray(f['mask'][:])
    return obstacle_mask


def expand_obstacle_mask(obstacle_mask, n_pixels):
    """
    Expand obstacle boundary by n pixels using morphological dilation.

    The resulting mask will include:
    - Original obstacle (False regions in input)
    - Border of n pixels around the obstacle

    Parameters:
    -----------
    obstacle_mask : np.ndarray
        Boolean array (H, W) where False=obstacle, True=valid region
    n_pixels : int
        Number of pixels to expand from obstacle boundary

    Returns:
    --------
    torch.Tensor
        Mask tensor (1, H, W) where:
        - 1 = sensor/known region (obstacle + n-pixel border)
        - 0 = unknown region (to be reconstructed)
    """
    # Invert: True=obstacle, False=valid (for dilation to work correctly)
    obstacle_binary = ~obstacle_mask

    # Dilate obstacle by n_pixels
    # This expands the True (obstacle) region outward
    structure = np.ones((3, 3))  # 8-connectivity
    expanded_obstacle = binary_dilation(obstacle_binary, structure=structure, iterations=n_pixels)

    # Create sensor mask: 1 where we have measurements (expanded obstacle), 0 elsewhere
    sensor_mask = expanded_obstacle.astype(np.float32)

    # Add channel dimension: (H, W) -> (1, H, W)
    sensor_mask = sensor_mask[np.newaxis, ...]

    # Convert to torch tensor
    mask_tensor = torch.from_numpy(sensor_mask)

    n_sensor_pixels = np.sum(expanded_obstacle)
    n_unknown_pixels = np.sum(~expanded_obstacle)

    print(f"Expanded obstacle by {n_pixels} pixels:")
    print(f"  Sensor pixels (known): {n_sensor_pixels}")
    print(f"  Unknown pixels (to reconstruct): {n_unknown_pixels}")
    print(f"  Coverage: {n_sensor_pixels / (n_sensor_pixels + n_unknown_pixels) * 100:.1f}%")

    return mask_tensor


def create_mask_PLIF(data, obstacle_mask, mask_type, n_pixels=None, random_percentage=None,
                     band_position=None, seed=0):
    """
    Create measurement mask for PLIF2D conditional generation.

    Parameters:
    -----------
    data : np.ndarray or torch.Tensor
        Data array (N, 1, H, W) - used for shape info
    obstacle_mask : np.ndarray
        Boolean obstacle mask (H, W) from dataset
    mask_type : str
        Type of mask: 'n_pixel_mask', 'random', 'horizontal_band', 'vertical_band'
    n_pixels : int, optional
        For 'n_pixel_mask': number of pixels to expand from obstacle
    random_percentage : float, optional
        For 'random': percentage of pixels to use as sensors (0-100)
    band_position : tuple, optional
        For band masks: (start, end) in pixels
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    tuple: (mask_tensor, data_torch)
        - mask_tensor: (1, H, W) sensor mask on GPU
        - data_torch: (N, 1, H, W) data as torch tensor on GPU
    """
    # Convert data to torch
    if isinstance(data, np.ndarray):
        data_torch = torch.from_numpy(data).cuda()
    else:
        data_torch = data.cuda()

    N, C, H, W = data_torch.shape
    assert C == 1, f"PLIF2D should have 1 channel, got {C}"

    # Initialize mask
    mask_tensor = torch.zeros((1, H, W), dtype=torch.float32)

    if mask_type == "n_pixel_mask":
        assert n_pixels is not None, "n_pixels must be provided for n_pixel_mask"
        mask_tensor = expand_obstacle_mask(obstacle_mask, n_pixels)

    elif mask_type == "random":
        assert random_percentage is not None, "random_percentage must be provided for random mask"
        np.random.seed(seed)
        total_pixels = H * W
        n_select = int(total_pixels * random_percentage / 100)

        # Get valid pixel locations (not in obstacle)
        valid_pixels = np.where(obstacle_mask.flatten())[0]
        selected_pixels = np.random.choice(valid_pixels, size=min(n_select, len(valid_pixels)), replace=False)

        mask_flat = np.zeros(total_pixels, dtype=np.float32)
        mask_flat[selected_pixels] = 1
        mask_tensor = torch.from_numpy(mask_flat.reshape(H, W)[np.newaxis, ...])

        print(f"Random mask: {n_select} pixels selected ({random_percentage}%)")

    elif mask_type == "horizontal_band":
        assert band_position is not None, "band_position (start, end) must be provided"
        y_start, y_end = band_position
        mask_tensor[:, :, y_start:y_end] = 1
        # Respect obstacle: only measure in valid regions
        obstacle_tensor = torch.from_numpy(obstacle_mask[np.newaxis, ...].astype(np.float32))
        mask_tensor = mask_tensor * obstacle_tensor

        print(f"Horizontal band mask: y=[{y_start}, {y_end})")

    elif mask_type == "vertical_band":
        assert band_position is not None, "band_position (start, end) must be provided"
        x_start, x_end = band_position
        mask_tensor[:, x_start:x_end, :] = 1
        # Respect obstacle: only measure in valid regions
        obstacle_tensor = torch.from_numpy(obstacle_mask[np.newaxis, ...].astype(np.float32))
        mask_tensor = mask_tensor * obstacle_tensor

        print(f"Vertical band mask: x=[{x_start}, {x_end})")

    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    # Move to GPU
    mask_tensor = mask_tensor.cuda()

    print(f"Mask tensor: shape={mask_tensor.shape}, dtype={mask_tensor.dtype}, device={mask_tensor.device}")

    return mask_tensor, data_torch


def add_obstacle_patch_PLIF(ax, obstacle_mask, x_coords, y_coords, obs_color='lightgray', alpha=0.7):
    """
    Add obstacle patch to matplotlib axis using dataset mask.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to add patch to
    obstacle_mask : np.ndarray
        Boolean mask (H, W) where False=obstacle
    x_coords : np.ndarray
        X coordinates (H,)
    y_coords : np.ndarray
        Y coordinates (W,)
    obs_color : str
        Color for obstacle
    alpha : float
        Transparency
    """
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    H, W = obstacle_mask.shape
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1

    # Find obstacle pixels
    obstacle_pixels = np.where(~obstacle_mask)

    if len(obstacle_pixels[0]) == 0:
        return  # No obstacle

    patches = []
    for i, j in zip(obstacle_pixels[0], obstacle_pixels[1]):
        x = x_coords[i]
        y = y_coords[j]
        rect = Rectangle((x - dx/2, y - dy/2), dx, dy)
        patches.append(rect)

    pc = PatchCollection(patches, facecolor=obs_color, edgecolor='none', alpha=alpha)
    ax.add_collection(pc)


def plot_mask_PLIF(mask_tensor, obstacle_mask, x_coords, y_coords, mask_type, n_pixels=None,
                   obs=None, out_pdf=False, out_png=False, out_show=False):
    """
    Plot PLIF2D mask with obstacle overlay.

    Parameters:
    -----------
    mask_tensor : torch.Tensor or np.ndarray
        Mask tensor (1, H, W)
    obstacle_mask : np.ndarray
        Obstacle boolean mask (H, W)
    x_coords, y_coords : np.ndarray
        Coordinate arrays
    mask_type : str
        Type of mask (for filename)
    n_pixels : int, optional
        Number of expanded pixels (for filename)
    obs : torch.Tensor, optional
        Observation data to overlay
    out_pdf, out_png, out_show : bool
        Output options
    """
    if isinstance(mask_tensor, torch.Tensor):
        mask_tensor = mask_tensor.cpu().numpy()

    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    if obs is not None:
        sensor = obs.cpu().numpy()
        sensor = sensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1]))
        im = ax.imshow(sensor[0, 0].T, cmap='RdBu_r', extent=extent, origin='lower', aspect='auto')
        ax.set_title("Sensor Inputs (With Mask)")
        fig.colorbar(im, ax=ax, orientation='vertical')
    else:
        # Show mask
        im = ax.imshow(mask_tensor[0].T, cmap='gray', extent=extent, origin='lower', aspect='auto',
                      vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"Mask: {mask_type}" + (f" (n={n_pixels})" if n_pixels else ""))
        fig.colorbar(im, ax=ax, orientation='vertical', label='Sensor (1=known, 0=unknown)')

    # Add obstacle overlay
    add_obstacle_patch_PLIF(ax, obstacle_mask, x_coords, y_coords, obs_color='red', alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    filename = f'mask-PLIF-{mask_type}'
    if n_pixels is not None:
        filename += f'-n{n_pixels}'

    if out_pdf:
        plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')
    if out_png:
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    if out_show:
        plt.show()

    plt.close(fig)