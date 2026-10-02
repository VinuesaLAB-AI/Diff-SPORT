import sys
from pathlib import Path
# Ensure imports work regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from libs import runner
from configs import *

from datetime import datetime

# Get the current time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("Training started at time:", current_time)


CONFIG_DICT = { 
                    'OneObs2D_ds1_10M': OneObs2D_ds1_10M.config_dict,
                    'PLIF2D_128x32': PLIF2D_128x32.config_dict,
            }

modelname = sys.argv[1]
print(f"selected model config: modelsize={modelname}")
config_dict = CONFIG_DICT[modelname]

config, ddpm_model, schedule = runner.create_model_and_initialize_ddpm(config_dict)
data_loader = runner.load_dataset(config)
runner.train(config, ddpm_model, data_loader)

# Get the current time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("Training ended at time:", current_time)
