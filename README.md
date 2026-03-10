# DNNKit

![CI](https://github.com/Festus0/dnnkit/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18945766.svg)](https://doi.org/10.5281/zenodo.18945766)
**DNNKit** is a lightweight **PyTorch framework for reproducible deep learning experiments**, designed for research workflows.  
It provides modular components for models, training pipelines, evaluation, and experiment management, making it easier to develop **reproducible machine learning experiments and academic prototypes**.

---

# Features

- PyTorch-based training and evaluation pipelines
- Modular architecture for models and datasets
- Reproducible experiment structure
- Built-in unit testing with **pytest**
- Continuous integration with **GitHub Actions**
- Academic paper scaffold for research projects
- Example **MNIST benchmark training pipeline**
- Command-line interface for quick experimentation

---

# Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Festus0/dnnkit.git
cd dnnkit
pip install -e .
pip install -r requirements-dev.txt
Command Line Interface

Train a model using the CLI:

dnnkit train --epochs 3 --dataset mnist --batch-size 64 --lr 0.001
Example Training Curve

Project Structure
dnnkit/
├── dnnkit/
│   ├── models/
│   ├── datasets/
│   ├── training/
│   └── evaluation/
│
├── tests/
├── docs/
├── scripts/
└── examples/
Development

Run tests locally:

pytest

Continuous integration runs automatically via GitHub Actions.

Roadmap

Planned improvements include:

Additional benchmark datasets

Hyperparameter search utilities

Experiment tracking integration

Model zoo

Distributed training support

License

This project is licensed under the MIT License.

Citation

If you use this repository in academic work, please cite:

Slade, F. (2026). DNNKit: A lightweight framework for reproducible deep learning experiments.
