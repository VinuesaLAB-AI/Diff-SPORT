import numpy as np
import torch
import sys
import os
from pathlib import Path

# Ensure imports work regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from libs import runner, utils
from libs.cond_gen_plif import (
    get_data_PLIF,
    normalize_PLIF,
    renormalize_PLIF,
    create_mask_PLIF,
    plot_mask_PLIF
)
from configs import *
from libs.lib_svd import Inpainting_custom
import h5py


# ============================================================================
# Configuration Dictionary
# ============================================================================
CONFIG_DICT = {
    'PLIF2D_128x32': PLIF2D_128x32.config_dict,
}


# ============================================================================
# Parse Command Line Arguments
# ============================================================================
config_name = sys.argv[1]      # e.g., 'PLIF2D_128x32'
mask_type   = sys.argv[2]      # e.g., 'n_pixel_mask', 'random', 'horizontal_band', 'vertical_band'
mask_param  = float(sys.argv[3])  # For n_pixel_mask: n_pixels; for random: percentage
snaps       = int(sys.argv[4])    # Number of snapshots to generate
gen_seed    = int(sys.argv[5])    # Random seed for generation
mask_seed   = int(sys.argv[6])    # Random seed for mask creation

print(f"=== PLIF2D Conditional Generation ===")
print(f"Config: {config_name}")
print(f"Mask type: {mask_type}")
print(f"Mask param: {mask_param}")
print(f"Snapshots: {snaps}")
print(f"Generation seed: {gen_seed}")
print(f"Mask seed: {mask_seed}")
print("=" * 50, flush=True)

# Get config
config_dict = CONFIG_DICT[config_name]


# ============================================================================
# Load Model
# ============================================================================
print(f"\n[1/6] Loading model...", flush=True)
config, ddpm_model, schedule = runner.create_model_and_initialize_ddpm(config_dict)
ddpm_model = runner.load_model(config, ddpm_model)

T_sub = config.sampling.T_sub
T_sub_mapgd = config.sampling.T_sub_mapgd
gditer_mapgd = config.sampling.gditer_mapgd

model_precision_type = "float16" if config.model.use_fp16 else "float32"
data_precision_type = "float16" if config.model.use_fp16_for_data else "float32"

print(f"  Model loaded: {config.load_model}")
print(f"  T_sub_mapgd: {T_sub_mapgd}, gditer_mapgd: {gditer_mapgd}")
print(f"  Model precision: {model_precision_type}, Data precision: {data_precision_type}", flush=True)


# ============================================================================
# Load PLIF2D Data
# ============================================================================
print(f"\n[2/6] Loading PLIF2D data...", flush=True)
config.dataset.data_file = config.dataset.test_data_file
data, alpha, beta, obstacle_mask, x, y, t = get_data_PLIF(config, datatype="Test")

# Select only requested number of snapshots
data = data[:snaps]
t = t[:snaps]

print(f"  Data shape: {data.shape}")
print(f"  Coordinates: x={x.shape}, y={y.shape}")
print(f"  Time steps: {len(t)}", flush=True)


# ============================================================================
# Normalize Data
# ============================================================================
print(f"\n[3/6] Normalizing data...", flush=True)
data_norm = normalize_PLIF(data, alpha, beta)
print(f"  Normalized data range: [{data_norm.min():.3f}, {data_norm.max():.3f}]", flush=True)


# ============================================================================
# Create Mask
# ============================================================================
print(f"\n[4/6] Creating mask: {mask_type}", flush=True)

# Parse mask_param based on mask_type
if mask_type == "n_pixel_mask":
    n_pixels = int(mask_param)
    mask_tensor, data_norm = create_mask_PLIF(
        data=data_norm,
        obstacle_mask=obstacle_mask,
        mask_type=mask_type,
        n_pixels=n_pixels,
        seed=mask_seed
    )
    mask_label = f"n{n_pixels}"

elif mask_type == "random":
    random_percentage = mask_param
    mask_tensor, data_norm = create_mask_PLIF(
        data=data_norm,
        obstacle_mask=obstacle_mask,
        mask_type=mask_type,
        random_percentage=random_percentage,
        seed=mask_seed
    )
    mask_label = f"{int(random_percentage)}pct"

elif mask_type in ["horizontal_band", "vertical_band"]:
    # For band masks, mask_param is interpreted as band size in pixels
    band_size = int(mask_param)
    H, W = data_norm.shape[2], data_norm.shape[3]

    if mask_type == "horizontal_band":
        # Center the band
        y_start = (W - band_size) // 2
        y_end = y_start + band_size
        band_position = (y_start, y_end)
    else:  # vertical_band
        # Center the band
        x_start = (H - band_size) // 2
        x_end = x_start + band_size
        band_position = (x_start, x_end)

    mask_tensor, data_norm = create_mask_PLIF(
        data=data_norm,
        obstacle_mask=obstacle_mask,
        mask_type=mask_type,
        band_position=band_position,
        seed=mask_seed
    )
    mask_label = f"band{band_size}"

else:
    raise ValueError(f"Unknown mask type: {mask_type}")

# Create H operator for inpainting
H = Inpainting_custom(mask_tensor=mask_tensor, device="cuda")

# Generate observations from ground truth
obs = H.H(data_norm)

# Plot mask visualization
plot_mask_PLIF(
    mask_tensor=mask_tensor,
    obstacle_mask=obstacle_mask,
    x_coords=x,
    y_coords=y,
    mask_type=mask_type,
    n_pixels=int(mask_param) if mask_type == "n_pixel_mask" else None,
    obs=None,
    out_pdf=True,
    out_png=False,
    out_show=False
)

print(f"  Mask created and saved", flush=True)


# ============================================================================
# Conditional Generation
# ============================================================================
print(f"\n[5/6] Starting conditional generation...", flush=True)

pred_images_mapgd = []
batch_size = config.sampling.batch_size
num_batch = (data_norm.shape[0] + batch_size - 1) // batch_size

start_time = utils.get_current_time()
print(f"  Start time: {start_time}")
print(f"  Batch size: {batch_size}, Num batches: {num_batch}", flush=True)

for i in range(num_batch):
    batch_start_time = utils.get_current_time()

    obs_batch = obs[i*batch_size:(i+1)*batch_size].cuda()
    data_norm_batch = data_norm[i*batch_size:(i+1)*batch_size].cuda()

    # Run MAPGD conditional generation
    mapgd_time_start = utils.get_current_time()
    x_recovered_mapgd = ddpm_model.reverse_diffusion_gd(
        data_norm_batch.shape,
        obs_batch,
        sigma_y=0,
        H=H,
        T_ddrm=T_sub_mapgd,
        cuda=True,
        onlymean=False,
        karras=False,
        adaptive=False,
        y_c=None,
        class_cond=False,
        gditer=gditer_mapgd,
        seed=gen_seed
    )
    mapgd_time_end = utils.get_current_time()

    x_recovered_cpu_mapgd = x_recovered_mapgd.cpu().numpy()
    pred_images_mapgd.append(x_recovered_cpu_mapgd)

    batch_end_time = utils.get_current_time()
    print(f"  Batch {i+1}/{num_batch} completed in {utils.get_time_difference(batch_start_time, batch_end_time)}", flush=True)

end_time = utils.get_current_time()
print(f"  Total generation time: {utils.get_time_difference(start_time, end_time)}", flush=True)


# ============================================================================
# Renormalize and Save Results
# ============================================================================
print(f"\n[6/6] Saving results...", flush=True)

# Concatenate predictions
pred_mapgd = np.concatenate(pred_images_mapgd, axis=0)

# Renormalize to original scale
pred_mapgd_renorm = renormalize_PLIF(pred_mapgd, alpha, beta)

# Prepare output directory
out_dir = REPO_ROOT / 'inference_utils' / 'conditional_generation' / 'generated_samples'
os.makedirs(str(out_dir), exist_ok=True)

# Create filename
filename_hdf5 = (
    f"cond_gen_PLIF-{mask_type}-{mask_label}-"
    f"ckpt{config.sampling.inf_epochs}-"
    f"T{T_sub_mapgd}-gd{gditer_mapgd}-"
    f"{snaps}snaps-"
    f"mseed{mask_seed}-gseed{gen_seed}.h5"
)

# Save to HDF5
with h5py.File(str(out_dir / filename_hdf5), 'w') as hf:
    hf.create_dataset("data", data=data)
    hf.create_dataset("mapgd", data=pred_mapgd_renorm)
    hf.create_dataset("mask_tensor", data=mask_tensor.cpu().numpy())
    hf.create_dataset("obstacle_mask", data=obstacle_mask)
    hf.create_dataset("x", data=x)
    hf.create_dataset("y", data=y)
    hf.create_dataset("t", data=t)
    hf.create_dataset("alpha", data=alpha)
    hf.create_dataset("beta", data=beta)
    hf.create_dataset("model_precision_type", data=model_precision_type, dtype='S10')
    hf.create_dataset("data_precision_type", data=data_precision_type, dtype='S10')

print(f"  Results saved to: {filename_hdf5}")
print(f"\n{'='*50}")
print(f"Conditional generation completed successfully!")
print(f"{'='*50}", flush=True)