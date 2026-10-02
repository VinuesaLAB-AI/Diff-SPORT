from configs.plot_config import plot_dict
import re
from pathlib import Path

# Resolve repository root (diffSPORT directory)
REPO_ROOT = Path(__file__).resolve().parents[1]

# PLIF2D dataset paths
train_data_file = str(REPO_ROOT / 'data' / 'plif_train_data_file.h5')
test_data_file  = str(REPO_ROOT / 'data' / 'plif_test_data_file.h5')

ckpt_dir        = str(REPO_ROOT / 'ckpts') + '/'
eval_dir        = str(REPO_ROOT / 'evaluate_utils') + '/'
cond_gen_dir    = str(REPO_ROOT / 'inference_utils' / 'conditional_generation' / 'generated_samples') + '/'
osp_dir         = str(REPO_ROOT / 'osp_utils') + '/'

prefix = "PLIF2D_128x32"

inf_epochs = 1000

# NOTE: pack_into_hdf5.py writes 'generated_samples-25k-<config>_ep1000.h5';
# rename the packed file to the name below, or edit this path.
pred_data_file  = str(REPO_ROOT / 'inference_utils' / 'unconditional_generation' / 'generated_samples' / f'generated_samples-{prefix}_ep{inf_epochs}.h5')
resume_ckpt_path = ckpt_dir + prefix +  f'/ckpt_{inf_epochs}.ckpt'
inference_ckpt_path = ckpt_dir + prefix + f'/ckpt_{inf_epochs}.ckpt'

config_dict = dict(

                config_name = prefix,
                dataset = dict(
                        name = "PLIF2D",
                        data_file = train_data_file,
                        test_data_file = test_data_file,
                        pred_data_file = pred_data_file,
                        filetype = "hdf5",
                        transform = None,
                        ds_ratio = 1, 
                        normalize = True,
                        alpha = 0.001,  # Log1p scale parameter for normalization
                        beta = 7.0,   # Normalization divisor to bring values to [-1, 1]
                        ),

                model = dict(
                            image_size = (128, 32),  
                            in_channels = 1,         
                            model_channels = 32,
                            out_channels = 1,        
                            num_res_blocks = 2,
                            attention_resolutions = [4, 8],  # Attentions at levels 2 and 3
                            dropout = 0,
                            channel_mult = (1, 2, 2),  # 4 levels: 128→64→32→16→8, 32→16→8→4→2
                            conv_resample = True,
                            dims = 2,
                            num_classes = None,
                            use_checkpoint = True,
                            use_fp16 = False,
                            use_fp16_for_data = False,
                            num_heads = 1,
                            num_head_channels = -1,
                            num_heads_upsample = -1,
                            use_scale_shift_norm = False,
                            resblock_updown = False,
                            use_new_attention_order = False,
                        ),

                diffusion = dict(
                        beta_schedule = 'linear',
                        beta_start = 0.0001,
                        beta_end = 0.02,
                        num_diffusion_timesteps = 1000,
                        weightedloss = False,
                        ),

                train = dict(
                        batch_size = 64, 
                        shuffle = True,
                        num_workers = 8,
                        lr = 5e-4,
                        num_epochs = 1000,
                        savefreq = 50,
                        prefix = prefix,
                        ckpt_dir = ckpt_dir,
                        MulG_train = True,
                        MulGMulN_train = False,
                        ),

                sampling = dict(
                        batch_size = 1000,
                        sampler = 'ddim',
                        num_samples = 100,
                        T_sub_mapgd = 20,
                        gditer_mapgd = 50,  
                        T_sub = 1000,
                        eta = 1.0,
                        onlymean = False,
                        sample_batch_size = 100,
                        inf_epochs = inf_epochs,
                        ),

                resume = dict(
                              ckpt = resume_ckpt_path,
                              batch_size = 128,  
                              num_epochs = 1000,
                              savefreq = 50,
                              prefix = prefix,
                              ckpt_dir = ckpt_dir,
                        ),

                eval = dict(
                              eval_dir = eval_dir,
                              cond_gen_dir = cond_gen_dir,
                              osp_dir = osp_dir,
                              plot_dict = plot_dict,
                        ),

                load_model = inference_ckpt_path,

                )

if __name__ == "__main__":
        print(config_dict)
