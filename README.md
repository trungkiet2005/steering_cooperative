# Latent Altruism: Steering Cooperative Intent in Large Language Models via Activation Engineering

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue.svg)](https://neurips.cc/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle Reproducible](https://img.shields.io/badge/Kaggle-Reproducible-informational.svg)](https://kaggle.com)

Code repository for the NeurIPS 2026 paper: **Latent Altruism: Steering Cooperative Intent in LLMs via Activation Engineering**.

This repository contains the full experimental pipeline to run our 4 novel Contextual Override and Mechanistic Interpretability experiments (Veto Circuit, Sparse Steering, Theory of Mind, and Dynamic Steering), as well as rigorous analytical tools (PPL tracking, Control Vectors, and Layer ablations).

## 🚀 Quick Start (Kaggle Reproducibility)

The easiest way to reproduce the findings in the paper is to run our master Jupyter Notebook on Kaggle (Requires GPU: T4x2, P100, or better).

- `notebooks/run_neurips_experiments.ipynb`

Simply upload this notebook to Kaggle and click **Run All**. It will automatically install requirements, run all ablation experiments, and output the PDF figures directly within the notebook interface.

## 📂 Repository Structure

- `experiments/`: Contains the core reproducible python scripts.
  - `kaggle_all_experiments.py`: Runs the 4 main oral experiments.
  - `kaggle_reviewer_ready_experiments.py`: Contains full reviewer rebuttal analytics (Perplexity, SADI, Orthogonal controls, Model sizing).
- `notebooks/`: Contains the end-to-end reproducible Jupyter Notebooks.
- `requirements.txt`: Python package dependencies.

## ⚙️ Local Installation

If you prefer to run the scripts locally on your own GPU:

```bash
git clone https://github.com/trungkiet2005/steering_cooperative.git
cd steering_cooperative
pip install -r requirements.txt

# Run the 4 Core Experments
python experiments/kaggle_all_experiments.py

# Run the Rebuttal Deep Analytics
python experiments/kaggle_reviewer_ready_experiments.py
```
