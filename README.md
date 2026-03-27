# Masked-AutoRL-SOP

This repository presents the **Masked AutoRL-SOP** algorithm:
**Automated Reinforcement Learning with Bayesian Hyperparameter Optimization and Invalid Action Masking for the Sequential Ordering Problem**.

In addition to the implementation, this repository includes:

* For reproducibility, all results reported in the paper are available in the `Results/` directory
* A Jupyter notebook for interactive visualization and analysis of results

---

## 📦 Installation and Setup

The code was tested using **Python 3.14.1**.

To install the required dependencies, navigate to the project directory and choose one of the following options:

### Option 1 — Using `requirements.txt`

```bash
conda create --name <env_name> --file requirements.txt
conda activate <env_name>
```

### Option 2 — Using `environment.yml` (recommended for full reproducibility)

```bash
conda env create -f environment.yml
conda activate <env_name>
```

---

## 🐳 Docker

For full reproducibility with identical environment settings, a Docker setup is provided.

> ⚠️ Runtimes in Docker may be slightly longer due to container overhead, especially on Windows/WSL without optimized Docker configurations.

### Build the Docker image

```bash
docker build -t masked-autorl-sop .
```

### Run the main algorithm

```bash
docker run -it masked-autorl-sop
```

### Run Jupyter Notebook for visualization

```bash
docker run -it -p 8888:8888 masked-autorl-sop \
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then open the generated link (e.g., `http://127.0.0.1:8888/tree...`) and run:

```
Viewer.ipynb
```

### Stopping the notebook

In the console, press:

```
Ctrl + C
```

Then confirm with:

```
y
```

### Notes

* Exporting figures as PDF requires **Kaleido**
* Kaleido depends on **Google Chrome**, which is **not included** in the Docker container
* The Docker containers do **not persist SQL database files** across runs  

### Stop all containers

```bash
docker stop $(docker ps -aq)
```

### Remove containers

```bash
docker rm $(docker ps -aq)
```

### Remove Docker images

```bash
docker rmi -f $(docker images -q)
```

---

## ▶️ Running the Code

To execute the main **Masked AutoRL-SOP** algorithm:

```bash
python Masked-AutoRL-SOP.py
```

You will then be prompted to select the instance index.

To visualize interactive results:

* Open `Viewer.ipynb` in your preferred notebook environment
* Execute the cells to explore the results

---

## 📊 Features

* Invalid action masking for structured action spaces
* Bayesian hyperparameter optimization
* Designed for the Sequential Ordering Problem (SOP)
* Reproducible experimental pipeline

---

## 📄 Citation

If you use this code or the provided results in your research, please cite:

```bibtex
@article{ramos2008maskedautorlsop,
  title={Automated Reinforcement Learning with Bayesian Hyperparameter Optimization and Invalid Action Masking for the Sequential Ordering Problem},
  author = {Ramos, Kerollan da Silva and Ottoni, André Luiz Carvalho and Pinto, Thomás and Santos, Allan Erlikhman Medeiros}
  journal={Computers \& Operations Research},
  volume={TBD},
  number={TBD},
  pages={TBD--TBD},
  year={TBD},
  publisher={Elsevier}
}
```

## 🤝 License

This project is licensed under the MIT License — see the LICENSE file for details.
