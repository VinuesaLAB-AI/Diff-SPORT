# Diff-SPORT

**Diffusion-based sensor placement optimization and reconstruction of turbulent flows in urban environments**

Abhijeet Vishwasrao, Sai Bharath Chandra Gutha, Andres Cremades, Klas Wijk, Aakash Patil, H.D. Lim, Christina Vanderwel, Catherine Gorle, Beverley J. McKeon, Hossein Azizpour, Ricardo Vinuesa

Preprint: [arXiv:2506.00214](https://arxiv.org/abs/2506.00214)

Turbulent urban flows are inherently stochastic and expensive to simulate accurately in real time with traditional methods. Here, the authors report Diff–SPORT, which uses a diffusion-model prior for optimal sensor placement and sparse-sensor flow reconstruction, enabling efficient urban flow monitoring validated on both simulated and experimental urban datasets.

![Overview of the Diff-SPORT framework: (a) diffusion prior, (b) MAP-GA sparse reconstruction, (c) SHAP-based optimal sensor placement](docs/fig1_overview.png)

## Abstract

Rapid urbanization demands efficient monitoring of turbulent wind and pollutant dispersion, yet existing reconstruction and sensor placement strategies fail under realistic sparsity constraints. Here, we introduce Diff-SPORT, a diffusion-based framework that combines a generative diffusion prior with maximum a posteriori inference and Shapley-value attribution for high-fidelity flow reconstruction and optimal sensor placement. By training a diffusion prior model once over a domain, Diff-SPORT enables non-linear optimal sensor placement and near-real-time flow reconstruction from sparse measurements orders of magnitude faster than RANS or LES simulations, consistently outperforming state-of-the-art methods. The framework also extends, without algorithmic modification, to experimental passive scalar concentration dataset, a direct proxy for pollutant dispersion, measured in a 1:2400 scale water-flume model of the Beijing Haidian neighbourhood under realistic urban flow conditions. Shapley-guided sensor placement achieves up to 57% lower reconstruction error than randomly placed sensors at extreme sparsity, identifying compact and physically interpretable configurations. These results establish Diff-SPORT as a modular foundation offering a zero-shot alternative to retraining-intensive downstream strategies, supporting scalable urban flow monitoring for air quality management and resilient city design.

## Repository map

| Path | Contents |
|---|---|
| `libs/` | Core library: DDPM wrapper and samplers (`lib_diffusion.py`), U-Net (`guided_diffusion/`), HDF5 dataset loader (`dataset.py`), model/training/sampling API (`runner.py`), conditional generation (`cond_gen.py`), statistics and instantaneous evaluation (`stats_eval.py`, `inst_eval.py`), SHAP sensor-placement helpers (`shap_eval.py`, `shap_osp.py`), plotting utilities (`utils.py`) |
| `libs/shap/` | Vendored copy of the SHAP library with a modified KernelSHAP coalition kernel; see [`libs/shap/VENDORED.md`](libs/shap/VENDORED.md) |
| `configs/` | Experiment configuration (data paths, model, training and inference settings) |
| `train_utils/` | Training entry point |
| `inference_utils/` | Unconditional sampling and conditional (sparse-sensor) reconstruction |
| `osp_utils/` | Optimal sensor placement: SHAP values, QR-pivoting baseline, random baseline, best-selection evaluation |
| `evaluate_utils/` | Evaluation scripts for unconditional and conditional generation |
| `run.sh` | Single command-line entry point for the whole workflow |
| `Dockerfile` | Container recipe with all dependencies |

### Naming in the code versus the paper

| In the code | In the paper |
|---|---|
| `MAPGD` | MAPGA (maximum a posteriori gradient ascent) |
| `PGDM`, `pigdm` | ΠGDM (pseudoinverse-guided diffusion model) |
| `mask`, `mask_type` | sensor coverage (% of pixels) and feasible sensor region |
| `coalitions` | sensor-subregion coalitions sampled by KernelSHAP |

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended (CPU works for small tests).

```bash
pip install -r requirements.txt
```

The `Dockerfile` builds an equivalent environment. The SHAP dependency is vendored under `libs/shap/` and does not need to be installed separately.

## Data

- **DNS dataset** (flow around a wall-mounted square cylinder): available on Zenodo at <https://doi.org/10.5281/zenodo.22737509>.
- **Experimental PLIF dataset** of the LT2400 urban canopy (Lim et al., *Exp. Fluids* 63, 92, 2022): available from the University of Southampton repository at <https://doi.org/10.5258/SOTON/D2217>.

The training/test split is defined by the `data_file` and `test_data_file` entries in `configs/`; point them at your own HDF5 files to use a different split or dataset.

## Reproducing the workflow

All steps go through `run.sh`; `bash run.sh --help` lists the commands and options. The defaults are small smoke-test values so that every command runs quickly. The numbers used in the paper (number of snapshots, sensor coverage, SHAP snapshot range and step, number of coalitions, mask thresholds) are given in the Methods and Supplementary Information and are passed as options.

1. **Train the diffusion prior**
   ```bash
   bash run.sh train --config OneObs2D_ds1_10M
   ```
2. **Unconditional generation and statistical evaluation** (Reynolds stresses, PDFs, spectra)
   ```bash
   bash run.sh gen-uncond --config OneObs2D_ds1_10M --seeds 1,2,3,4,5
   bash run.sh pack-uncond --config OneObs2D_ds1_10M
   bash run.sh eval-uncond --config OneObs2D_ds1_10M
   ```
3. **Conditional reconstruction from a sensor mask** with MAPGD, ΠGDM and DDRM, and its evaluation
   ```bash
   bash run.sh gen-cond  --config OneObs2D_ds1_10M --mask_type from_ground_and_wall --mask <coverage %> --snaps <n>
   bash run.sh eval-cond --config OneObs2D_ds1_10M --mask_type from_ground_and_wall --mask <coverage %> --snaps <n>
   ```
   Add `--random_sensors <n>` to `gen-cond` for the random-placement baseline.
4. **SHAP values of the sensor subregions** over a range of training snapshots
   ```bash
   bash run.sh shap-values --config OneObs2D_ds1_10M --shap_start <i0> --shap_stop <i1> --shap_step <s> --shap_ncoalitions <n>
   ```
5. **Build sensor masks (manual notebook steps)**
   - QR-pivoting masks: `osp_utils/qr-pivoting/qr-pivoting-v4.ipynb`
   - SHAP threshold masks: `osp_utils/shap/related_notebooks/v5.5-tdiff20-gditer50-snaps-25000-step-50.ipynb`
6. **Reconstruction sweeps over the SHAP and QR masks**, then the comparison against random placement
   ```bash
   bash run.sh gen-osp --config OneObs2D_ds1_10M --method shap --seeds 0,1,2,3,4,5
   bash run.sh gen-osp --config OneObs2D_ds1_10M --method qr   --seeds 0,1,2,3,4,5
   bash run.sh eval-best --config OneObs2D_ds1_10M
   bash run.sh shap-summary
   ```

The `job_*.sh` scripts next to the Python entry points are the thin wrappers we used on a SLURM cluster; add your own scheduler header to use them.

## Citation

```bibtex
@article{vishwasrao2025diffsport,
  title   = {Diffusion-based sensor placement optimization and reconstruction of turbulent flows in urban environments},
  author  = {Vishwasrao, Abhijeet and Gutha, Sai Bharath Chandra and Cremades, Andres and Wijk, Klas and Patil, Aakash and Lim, H. D. and Vanderwel, Christina and Gorle, Catherine and McKeon, Beverley J. and Azizpour, Hossein and Vinuesa, Ricardo},
  journal = {arXiv preprint arXiv:2506.00214},
  year    = {2025},
  doi     = {10.48550/arXiv.2506.00214}
}
```

A `CITATION.cff` file is included; the journal reference will be added once the article is published.

## License

MIT License (see `LICENSE`). The vendored SHAP library under `libs/shap/` is MIT-licensed, Copyright (c) 2018 Scott Lundberg.
