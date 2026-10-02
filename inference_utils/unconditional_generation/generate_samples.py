import numpy as np
import torch
import sys
import os
from pathlib import Path

# Ensure imports work regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from libs import runner
from configs import *

CONFIG_DICT = { 
                    'OneObs2D_ds1_10M': OneObs2D_ds1_10M.config_dict,
                    'PLIF2D_128x32': PLIF2D_128x32.config_dict,
            }

seed = int(sys.argv[1])
config_name = sys.argv[2]
savedir =  sys.argv[3] if len(sys.argv) > 3 else None
print(f"selected args: seed={seed}, config_name={config_name}")
config_dict = CONFIG_DICT[config_name]


config = runner.dict2namespace(config_dict)
config, ddpm_model, schedule = runner.create_model_and_initialize_ddpm(config_dict)
ddpm_model = runner.load_model(config, ddpm_model)

np.random.seed(seed)
torch.manual_seed(seed)
print(f"Generating Samples:")  
pred_samples = runner.generate_samples(config, ddpm_model, sampler=config.sampling.sampler, num_samples=config.sampling.num_samples, 
                                        sample_batch_size=config.sampling.sample_batch_size, T_sub=config.sampling.T_sub, 
                                        eta=config.sampling.eta, onlymean=config.sampling.onlymean)

# Determine output directory
default_out = REPO_ROOT / 'inference_utils' / 'unconditional_generation' / 'generated_samples'
out_dir = Path(savedir) if savedir else default_out
if not out_dir.is_absolute():
    out_dir = REPO_ROOT / out_dir
os.makedirs(str(out_dir), exist_ok=True)

filename = out_dir / f"generated_samples_{config_name}_seed{seed}_ep500.npy"
np.save(str(filename), pred_samples)
print(f"Done!")  
