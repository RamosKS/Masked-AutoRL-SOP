# Masked-AutoRL-SOP

This repository provides the **Masked AutoRL-SOP** framework:
**Automated Reinforcement Learning with Bayesian Hyperparameter Optimization and Invalid Action Masking for the Sequential Ordering Problem**.

The primary configurations combine invalid action masking with Bayesian hyperparameter optimization based on the Tree-structured Parzen Estimator (TPE) or Gaussian Process (GP). Hyperband can optionally be applied as a multi-fidelity pruning mechanism. The repository also includes control and ablation configurations based on Random Search, sequential HyperTuningSK optimization, and action-resampling.

## Repository contents

- `Masked AutoRL-SOP.py`: interactive launcher for all implementations.
- `variants/`: the 11 model implementations and their corresponding `Results/` and `Train Plots/` directories.
- `SOP_Datasets/`: the SOP instances shared by all implementations.
- `Viewer.ipynb`: interactive analysis of the stored Optuna databases and CSV logs.
- `figures_pdf/`: output hierarchy used by the Viewer. Generated PDFs are ignored by Git; `.gitkeep` files preserve the directory structure.

## Installation and setup

The project was tested with **Python 3.13.10**. The pinned environment files target `linux-64` and are intended for Docker, WSL, or native Linux.

### Complete Conda environment

The recommended setup recreates the complete environment, including the dependencies required by the GP variants:

```bash
conda env create -f environment.yml
conda activate rl
```

The environment includes Optuna 4.6.0 and PyTorch, which is required by Optuna's `GPSampler`.

### Explicit Linux package export

`requirements.txt` is an explicit `linux-64` Conda export. It can recreate the Conda-managed package layer on the same platform:

```bash
conda create --name rl --file requirements.txt
conda activate rl
```

Because pip-managed dependencies are recorded in `environment.yml`, that file should be used for the complete project environment.

## Running the code

Open the interactive launcher from the repository root:

```bash
python "Masked AutoRL-SOP.py"
```

The launcher first requests the model and then the SOP instance. Its model menu follows this exact order. The **Article label** column gives the abbreviated identifier used in the manuscript's tables and discussion.

| Option | Launcher entry | Article label | Configuration |
|---:|---|---|---|
| 1 | `[MULTIVARIATE] HYPERBAND TPE Masked AutoRL-SOP` | `TPE-M HB ON` | Multivariate TPE, invalid action masking, and Hyperband. |
| 2 | `[MULTIVARIATE] TPE Masked AutoRL-SOP` | `TPE-M HB OFF` | Multivariate TPE and invalid action masking, without Hyperband. |
| 3 | `[UNIVARIATE] HYPERBAND TPE Masked AutoRL-SOP` | `TPE-U HB ON` | Univariate TPE, invalid action masking, and Hyperband. |
| 4 | `[UNIVARIATE] TPE Masked AutoRL-SOP` | `TPE-U HB OFF` | Univariate TPE and invalid action masking, without Hyperband. |
| 5 | `HYPERBAND GP Masked AutoRL-SOP` | `GP HB ON` | GP, invalid action masking, and Hyperband. |
| 6 | `GP Masked AutoRL-SOP` | `GP HB OFF` | GP and invalid action masking, without Hyperband. |
| 7 | `NO_BAYESIAN Masked AutoRL-SOP` | `NO_BAYESIAN` | Sequential HyperTuningSK optimization and invalid action masking. |
| 8 | `[MULTIVARIATE] NO_MASK Masked AutoRL-SOP` | `NO_MASK-M` | Multivariate TPE with action-resampling instead of invalid action masking. |
| 9 | `[UNIVARIATE] NO_MASK Masked AutoRL-SOP` | `NO_MASK-U` | Univariate TPE with action-resampling instead of invalid action masking. |
| 10 | `NO_MASK_NO_BAYESIAN Masked AutoRL-SOP` | `AutoRL-SOP (Python)` | Sequential HyperTuningSK optimization and action-resampling. |
| 11 | `RANDOM_SEARCH Masked AutoRL-SOP` | `RS` | Random Search and invalid action masking. |

Options 1–6 are the main Bayesian configurations. Options 7–11 provide non-Bayesian controls and component-level ablations.

Each implementation reads the shared data from `SOP_Datasets/` and writes its outputs inside its own variant directory. Optuna-based implementations store SQLite studies in `Results/`; HyperTuningSK implementations store complete CSV optimization logs. Final-training plots are written to `Train Plots/`.

## Interactive result analysis

Start Jupyter and open `Viewer.ipynb`:

```bash
jupyter notebook Viewer.ipynb
```

The notebook requests the model and instance before loading the corresponding database or CSV log. It provides convergence, exploration, parameter, importance, and final-training analyses. The fANOVA evaluator uses `seed=42` for reproducible importance estimates.

PDF reports can be generated from the notebook under `figures_pdf/<variant>/<instance>/`. Interactive plots do not require PDF export; exporting Plotly figures requires Kaleido and a compatible Chrome installation.

## Docker

Build the image from the repository root:

```bash
docker build -t masked-autorl-sop .
```

Mount the repository when running the launcher so that databases, CSV logs, and plots persist on the host.

Windows PowerShell:

```powershell
docker run --rm -it -v "${PWD}:/app" masked-autorl-sop
```

Linux or macOS:

```bash
docker run --rm -it -v "$(pwd):/app" masked-autorl-sop
```

To run the Viewer in Docker:

```bash
docker run --rm -it -p 8888:8888 -v "$(pwd):/app" masked-autorl-sop jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

On Windows PowerShell, replace `$(pwd)` with `${PWD}`. The Docker image includes Kaleido but does not install Chrome, so PDF export from Plotly may require an additional browser installation in the container.

Container runtimes can differ from native runtimes, particularly with Docker Desktop and WSL. Runtime comparisons reported in the paper should therefore be reproduced under the stated native environment.

## Features

- Invalid action masking for precedence-constrained action spaces.
- Univariate and multivariate TPE Bayesian optimization.
- GP Bayesian optimization.
- Optional Hyperband multi-fidelity pruning.
- Random Search and HyperTuningSK comparison models.
- Action-resampling and HPO ablation models.
- Reproducible per-variant result storage and interactive analysis.

## Citation

If you use this code or the provided results in your research, please cite the reference below.
The manuscript associated with these codes is currently under review, and the citation information will be updated upon publication.

```bibtex
@article{ramos2026maskedautorlsop,
  title={Automated Reinforcement Learning with Bayesian Hyperparameter Optimization and Invalid Action Masking for the Sequential Ordering Problem},
  author={Ramos, Kerollan da Silva and Ottoni, André Luiz Carvalho and Pinto, Thomás and Santos, Allan Erlikhman Medeiros},
  journal={Computers \& Operations Research},
  volume={TBD},
  number={TBD},
  pages={TBD--TBD},
  year={TBD},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.
