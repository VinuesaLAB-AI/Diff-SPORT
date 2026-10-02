"""
Shared utilities for PLIF2D SHAP-OSP experiments.

Provides data loading, mask creation, MAPGD inference, and MSE helpers
that are reused across Phase 1 (random baseline) and Phase 2 (SHAP OSP).
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import copy
from pathlib import Path

# Resolve repo root and ensure imports work
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from libs import runner
from libs.cond_gen_plif import (
    get_data_PLIF,
    normalize_PLIF,
    renormalize_PLIF,
    expand_obstacle_mask,
)
from libs.lib_svd import Inpainting_custom, Inpainting_custom_mask_batches
from libs import utils as lib_utils
from configs import PLIF2D_128x32


CONFIG_DICT = {
    'PLIF2D_128x32': PLIF2D_128x32.config_dict,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_plif2d_data(config, datatype="Test", data_file=None):
    """
    Load PLIF2D data, switching to test file when datatype="Test".

    get_data_PLIF always uses config.dataset.data_file (train).
    We temporarily swap it to test_data_file for test loading.

    Parameters
    ----------
    config : namespace
    datatype : str, "Train" or "Test"
    data_file : str or None, explicit data file path override

    Returns
    -------
    data : np.ndarray, (N, 1, H, W) raw fluctuations
    alpha, beta : float, normalization params
    obstacle_mask : np.ndarray, (H, W) boolean
    x_coords, y_coords : np.ndarray
    t : np.ndarray, (N,)
    """
    cfg = copy.deepcopy(config)
    if data_file is not None:
        cfg.dataset.data_file = data_file
    elif datatype == "Test":
        cfg.dataset.data_file = cfg.dataset.test_data_file

    data, alpha, beta, obstacle_mask, x_coords, y_coords, t = get_data_PLIF(cfg, datatype=datatype)
    return data, alpha, beta, obstacle_mask, x_coords, y_coords, t


# ---------------------------------------------------------------------------
# Mask creation
# ---------------------------------------------------------------------------

def create_base_mask(obstacle_mask, n_pixels=3):
    """
    Create the base measurement region by expanding obstacle boundary.

    Returns
    -------
    base_mask : torch.Tensor, (1, H, W), 1=measurement region, 0=unknown
    """
    return expand_obstacle_mask(obstacle_mask, n_pixels)


def get_sensor_pixel_indices(base_mask, obstacle_mask):
    """
    Get flat indices of valid sensor locations within the base mask.

    Valid = base_mask==1 AND obstacle_mask==True (flow region, not obstacle interior).

    Parameters
    ----------
    base_mask : torch.Tensor, (1, H, W)
    obstacle_mask : np.ndarray, (H, W) boolean, True=valid flow

    Returns
    -------
    sensor_indices : np.ndarray of int, flat indices into (H, W)
    """
    mask_np = base_mask[0].cpu().numpy().astype(bool)  # (H, W)
    valid = mask_np & obstacle_mask  # only flow pixels within measurement region
    sensor_indices = np.flatnonzero(valid)
    return sensor_indices


def create_random_sensor_mask(sensor_indices, n_select, shape_hw, seed=0):
    """
    Randomly select n_select pixels from sensor_indices, return binary mask.

    Parameters
    ----------
    sensor_indices : np.ndarray, flat indices of candidate pixels
    n_select : int, number of sensors to place
    shape_hw : tuple (H, W)
    seed : int

    Returns
    -------
    mask_tensor : torch.Tensor, (1, H, W), float32, 1=sensor, 0=unknown
    """
    rng = np.random.RandomState(seed)
    chosen = rng.choice(sensor_indices, size=min(n_select, len(sensor_indices)), replace=False)

    mask_flat = np.zeros(shape_hw[0] * shape_hw[1], dtype=np.float32)
    mask_flat[chosen] = 1.0
    mask_2d = mask_flat.reshape(shape_hw)
    return torch.from_numpy(mask_2d[np.newaxis, ...])  # (1, H, W)


# ---------------------------------------------------------------------------
# MAPGD inference
# ---------------------------------------------------------------------------

def run_mapgd_inference(ddpm_model, data_norm, mask_tensor, config, gen_seed=42):
    """
    Run batched MAPGD conditional generation.

    Parameters
    ----------
    ddpm_model : DDPM_R model instance
    data_norm : torch.Tensor, (N, 1, H, W) normalized data on CPU
    mask_tensor : torch.Tensor, (1, H, W) sensor mask
    config : namespace with sampling params
    gen_seed : int

    Returns
    -------
    pred_norm : np.ndarray, (N, 1, H, W) normalized predictions
    """
    T_sub_mapgd = config.sampling.T_sub_mapgd
    gditer_mapgd = config.sampling.gditer_mapgd
    batch_size = config.sampling.batch_size

    H = Inpainting_custom(mask_tensor=mask_tensor, device="cuda")
    obs = H.H(data_norm.cuda())  # (N, 1*H*W) flattened observations

    N = data_norm.shape[0]
    num_batches = max(1, N // batch_size)
    pred_list = []

    for i in range(num_batches):
        s, e = i * batch_size, min((i + 1) * batch_size, N)
        obs_batch = obs[s:e].cuda()
        shape_batch = (e - s, *data_norm.shape[1:])

        x_recovered = ddpm_model.reverse_diffusion_gd(
            shape_batch, obs_batch, sigma_y=0, H=H,
            T_ddrm=T_sub_mapgd, cuda=True, onlymean=False,
            karras=False, adaptive=False, y_c=None,
            class_cond=False, gditer=gditer_mapgd, seed=gen_seed,
        )
        pred_list.append(x_recovered.cpu().numpy())

    # Handle remainder
    remainder = N - num_batches * batch_size
    if remainder > 0:
        s = num_batches * batch_size
        obs_batch = obs[s:].cuda()
        shape_batch = (remainder, *data_norm.shape[1:])

        x_recovered = ddpm_model.reverse_diffusion_gd(
            shape_batch, obs_batch, sigma_y=0, H=H,
            T_ddrm=T_sub_mapgd, cuda=True, onlymean=False,
            karras=False, adaptive=False, y_c=None,
            class_cond=False, gditer=gditer_mapgd, seed=gen_seed,
        )
        pred_list.append(x_recovered.cpu().numpy())

    return np.concatenate(pred_list, axis=0)


# ---------------------------------------------------------------------------
# MSE computation
# ---------------------------------------------------------------------------

def compute_mse_plif(gtruth, pred, c_ref=1.0):
    """
    Compute MSE between ground truth and prediction for single-channel PLIF2D.

    MSE = mean((C'_gt - C'_pred)^2) / c_ref^2

    Parameters
    ----------
    gtruth, pred : np.ndarray, (N, 1, H, W) or (N, H, W)
    c_ref : float

    Returns
    -------
    per_snapshot : np.ndarray, (N,) MSE per snapshot
    mean_mse : float, overall mean MSE
    """
    if gtruth.ndim == 4:
        gtruth = gtruth[:, 0]
    if pred.ndim == 4:
        pred = pred[:, 0]

    error = (gtruth - pred) ** 2 / (c_ref ** 2)
    per_snapshot = np.mean(error, axis=(1, 2))
    return per_snapshot, float(np.mean(per_snapshot))


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def load_model_and_config(config_name='PLIF2D_128x32'):
    """
    Load DDPM model and config namespace.

    Returns
    -------
    config : namespace
    ddpm_model : loaded model on GPU
    """
    config_dict = CONFIG_DICT[config_name]
    config, ddpm_model, schedule = runner.create_model_and_initialize_ddpm(config_dict)
    ddpm_model = runner.load_model(config, ddpm_model)
    return config, ddpm_model


# ---------------------------------------------------------------------------
# Pixel-level segmentation for SHAP
# ---------------------------------------------------------------------------

def create_pixel_segmentation(sensor_indices, shape_hw):
    """
    Create a segmentation tensor where each valid sensor pixel is a unique feature.

    Parameters
    ----------
    sensor_indices : np.ndarray, flat indices of valid sensor pixels
    shape_hw : tuple (H, W)

    Returns
    -------
    seg_tensor : torch.Tensor, (1, H, W), int. 0=background, 1..N=feature IDs
    unique_values : torch.Tensor, (N,), the feature IDs (1..N)
    """
    seg_flat = np.zeros(shape_hw[0] * shape_hw[1], dtype=np.int64)
    for i, idx in enumerate(sensor_indices):
        seg_flat[idx] = i + 1  # 1-indexed feature IDs
    seg_2d = seg_flat.reshape(shape_hw)
    seg_tensor = torch.from_numpy(seg_2d[np.newaxis, ...])  # (1, H, W)

    unique_values = torch.arange(1, len(sensor_indices) + 1, dtype=torch.int64)
    return seg_tensor, unique_values


def create_superpixel_segmentation(sensor_indices, shape_hw, block_size):
    """
    Group sensor pixels into block_size × block_size superpixels.
    Each superpixel = one SHAP feature.

    The measurement region (from expand_obstacle_mask) is irregular. A regular grid
    of block_size × block_size is overlaid on the full (H, W) domain. For each grid
    cell that contains at least one valid sensor pixel, ALL sensor pixels within that
    cell share a single SHAP feature ID.

    Grid cells at the measurement region boundary may contain fewer pixels than
    interior cells - this is fine, no special handling needed.

    Parameters
    ----------
    sensor_indices : np.ndarray, flat indices of valid sensor pixels
    shape_hw : tuple (H, W)
    block_size : int, side length of blocks (must be >= 1)

    Returns
    -------
    seg_tensor : torch.Tensor, (1, H, W), int. 0=background, 1..N=feature IDs
    unique_values : torch.Tensor, (N,), feature IDs 1..N

    Notes
    -----
    If block_size == 1, delegates to create_pixel_segmentation() for efficiency.
    """
    if block_size == 1:
        return create_pixel_segmentation(sensor_indices, shape_hw)

    H, W = shape_hw

    # Convert flat sensor indices to 2D coordinates
    sensor_rows = sensor_indices // W
    sensor_cols = sensor_indices % W

    # Compute block assignments for each sensor pixel
    block_rows = sensor_rows // block_size
    block_cols = sensor_cols // block_size

    # Number of block columns in the grid
    n_block_cols = (W + block_size - 1) // block_size  # ceil(W / block_size)

    # Unique block ID for each sensor pixel (using row-major ordering)
    block_ids = block_rows * n_block_cols + block_cols

    # Get unique block IDs and remap to consecutive 1-indexed feature IDs
    unique_block_ids = np.unique(block_ids)
    block_id_to_feature_id = {bid: i + 1 for i, bid in enumerate(unique_block_ids)}

    # Build segmentation tensor
    seg_flat = np.zeros(H * W, dtype=np.int64)
    for sensor_idx, block_id in zip(sensor_indices, block_ids):
        seg_flat[sensor_idx] = block_id_to_feature_id[block_id]

    seg_2d = seg_flat.reshape(H, W)
    seg_tensor = torch.from_numpy(seg_2d[np.newaxis, ...])  # (1, H, W)

    n_features = len(unique_block_ids)
    unique_values = torch.arange(1, n_features + 1, dtype=torch.int64)

    return seg_tensor, unique_values


# ---------------------------------------------------------------------------
# SHAP-OSP class for PLIF2D
# ---------------------------------------------------------------------------

class shap_osp_plif:
    """
    SHAP model function wrapper for PLIF2D optimal sensor placement.

    Adapted from libs.shap_osp.shap_osp (OneObs2D) with:
    - Log1p normalization (normalize_PLIF / renormalize_PLIF)
    - Single-channel MSE (no magnitude norm)
    - Pixel-level segmentation (each pixel = 1 feature)
    """

    def __init__(self, model, gtruth_tensor, alpha, beta,
                 zero_mask, full_mask, seg_tensor, unique_segments,
                 zzero, zshap, config):
        """
        Parameters
        ----------
        model : DDPM_R model instance
        gtruth_tensor : torch.Tensor, (N, 1, H, W) raw (unnormalized) data on GPU
        alpha, beta : float, log1p normalization params
        zero_mask : torch.Tensor, (1, H, W), all-zeros mask (no sensors)
        full_mask : torch.Tensor, (1, H, W), full measurement region mask
        seg_tensor : torch.Tensor, (1, H, W), pixel segmentation
        unique_segments : torch.Tensor, (N_features,), feature IDs
        zzero : np.ndarray, (1, N_features), background vector (all zeros)
        zshap : np.ndarray, (1, N_features), foreground vector (all ones)
        config : namespace with sampling params
        """
        self.model = model
        self.gtruth_tensor = gtruth_tensor
        self.alpha = alpha
        self.beta = beta
        self.zero_mask = zero_mask
        self.full_mask = full_mask
        self.seg_tensor = seg_tensor
        self.unique_segments = unique_segments
        self.zzero = zzero
        self.zshap = zshap
        self.config = config
        self.T_sub_mapgd = config.sampling.T_sub_mapgd
        self.gditer_mapgd = config.sampling.gditer_mapgd

    def mask_dom_mask_batches(self, zs):
        """
        Build per-coalition spatial masks from binary feature vectors.

        Parameters
        ----------
        zs : np.ndarray, (num_coalitions, N_features), binary

        Returns
        -------
        masks : torch.Tensor, (num_coalitions, 1, H, W)
        """
        background = self.zero_mask.clone()
        masks = []

        for ii in range(zs.shape[0]):
            mask_out = self.full_mask.clone()
            for jj, seg_id in enumerate(self.unique_segments):
                if zs[ii, jj] == 0:
                    mask_out[self.seg_tensor == seg_id] = background[self.seg_tensor == seg_id]
            masks.append(mask_out)

        return torch.stack(masks, dim=0)

    def shap_wrapper_mapgd_mask_batches(self, H, obs_batch, data_norm_batch, seed=0):
        """
        Run MAPGD with per-sample masks and compute normalized-space MSE.

        Returns
        -------
        mse_batch : np.ndarray, (batch, 1), MSE per coalition
        """
        x_output_norm = self.model.reverse_diffusion_gd_mask_batches(
            data_norm_batch.shape, obs_batch, sigma_y=0, H=H,
            T_ddrm=self.T_sub_mapgd, cuda=True, onlymean=False,
            karras=False, adaptive=False, y_c=None,
            class_cond=False, gditer=self.gditer_mapgd, seed=seed,
        )

        # MSE in normalized space, single channel → (batch, 1)
        loss = F.mse_loss(x_output_norm.detach().to("cuda"), data_norm_batch, reduction='none')
        loss = loss.mean(dim=(2, 3))  # (batch, 1)
        return loss.cpu().numpy()

    def model_function_mask_batches(self, zs, gen_seeds=[0, 1, 2, 3]):
        """
        SHAP model function: evaluate MSE for each coalition configuration.

        Called by shap.KernelExplainer. Takes binary coalition matrix,
        returns MSE per coalition (best across gen_seeds).

        Parameters
        ----------
        zs : np.ndarray, (num_coalitions, N_features), binary

        Returns
        -------
        all_best_errors : np.ndarray, (num_coalitions, 1)
        """
        lm = zs.shape[0]
        batch_size = max(self.config.sampling.batch_size, 1)
        num_batches = max(lm // batch_size, 1)

        start_time = lib_utils.get_current_time()
        print(f"Starting SHAP eval @ {start_time}, coalitions: {lm}", flush=True)

        # Build all coalition masks
        masks = self.mask_dom_mask_batches(zs)

        # Normalize ground truth: raw → log1p normalized
        data_raw_np = self.gtruth_tensor.cpu().numpy()
        data_norm_np = normalize_PLIF(data_raw_np, self.alpha, self.beta)
        data_norm = torch.from_numpy(data_norm_np).float()

        # Match batch dimensions (coalitions vs data samples)
        if data_norm.shape[0] != masks.shape[0]:
            if data_norm.shape[0] > masks.shape[0]:
                repeat_factor = (data_norm.shape[0] + masks.shape[0] - 1) // masks.shape[0]
                masks = masks.repeat(repeat_factor, 1, 1, 1)[:data_norm.shape[0], ...]
            else:
                repeat_factor = (masks.shape[0] + data_norm.shape[0] - 1) // data_norm.shape[0]
                data_norm = data_norm.repeat(repeat_factor, 1, 1, 1)[:masks.shape[0], ...]

        assert data_norm.shape == masks.shape

        if lm == 1:
            batch_size = 1
            num_batches = 1

        all_best_errors = []
        for i in range(num_batches):
            masks_batch = masks[i * batch_size:(i + 1) * batch_size].cuda()
            data_norm_batch = data_norm[i * batch_size:(i + 1) * batch_size].cuda()

            H = Inpainting_custom_mask_batches(mask_tensor=masks_batch, device=masks_batch.device)
            obs_batch = H.H(data_norm_batch)

            instance_errors = []
            for gen_seed in gen_seeds:
                mse_batch = self.shap_wrapper_mapgd_mask_batches(
                    H, obs_batch, data_norm_batch, seed=gen_seed
                )
                # Single channel: mse_batch is (batch, 1), use directly
                instance_errors.append(mse_batch)

            # Best-of-seeds selection
            instance_errors = np.concatenate(instance_errors, axis=1)  # (batch, n_seeds)
            best_indices = np.argmin(instance_errors, axis=1)
            best_errors = instance_errors[np.arange(len(instance_errors)), best_indices].reshape(-1, 1)

            end_time = lib_utils.get_current_time()
            print(f"  batch {i+1}/{num_batches} done in {lib_utils.get_time_difference(start_time, end_time)}, "
                  f"best_indices: {best_indices}", flush=True)

            all_best_errors.append(best_errors)

        # Handle remainder
        remainder = lm - num_batches * batch_size
        if remainder > 0:
            masks_batch = masks[num_batches * batch_size:].cuda()
            data_norm_batch = data_norm[num_batches * batch_size:].cuda()

            H = Inpainting_custom_mask_batches(mask_tensor=masks_batch, device=masks_batch.device)
            obs_batch = H.H(data_norm_batch)

            instance_errors = []
            for gen_seed in gen_seeds:
                mse_batch = self.shap_wrapper_mapgd_mask_batches(
                    H, obs_batch, data_norm_batch, seed=gen_seed
                )
                instance_errors.append(mse_batch)

            instance_errors = np.concatenate(instance_errors, axis=1)
            best_indices = np.argmin(instance_errors, axis=1)
            best_errors = instance_errors[np.arange(len(instance_errors)), best_indices].reshape(-1, 1)
            all_best_errors.append(best_errors)

        all_best_errors = np.concatenate(all_best_errors, axis=0)
        print(f"SHAP eval done: {all_best_errors.shape[0]} coalitions, "
              f"total time: {lib_utils.get_time_difference(start_time, lib_utils.get_current_time())}", flush=True)

        return all_best_errors
