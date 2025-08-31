```markdown
# Netflix Recommendation System

## Overview

This project implements a comprehensive recommendation system for the Netflix dataset, comparing classical collaborative filtering methods with state-of-the-art neural models. It includes baseline models, matrix factorization with SVD, and a neural collaborative filtering (NCF) model built in PyTorch.

---

## Project Structure

```
netflix-recsys/
├── data/                  # Raw and processed datasets
├── model_artifacts/       # Saved models and checkpoints
├── notebooks/             # Jupyter notebooks for EDA & experiments
├── results/               # Markdown summaries and evaluation reports
├── src/                   # Core codebase (data, models, utils, experiments, api)
├── tests/                 # Unit tests covering all core modules
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
```

---

## Features

- **Baseline Models:** Popularity and Random recommendation baselines.
- **Matrix Factorization:** SVD implementation using `surprise` library with hyperparameter tuning.
- **Neural Collaborative Filtering:** Deep learning model in PyTorch with embeddings and MLP layers, including advanced hyperparameter tuning.
- **Evaluation:** Standard metrics such as Precision@k, Recall@k, and NDCG@k.
- **Modular Codebase:** Well-structured modules for dataset handling, model definitions, training routines, and evaluation.
- **Testing:** Robust test suite using `unittest` framework ensures code reliability.
- **Extensible:** Designed for easy addition of new models, datasets, and deployment strategies.

---

## Setup Instructions

1. **Clone repository**

```
git clone https://github.com/SamrithaShree/netflix-recsys.git
cd netflix-recsys
```

2. **Create Python virtual environment and activate**

```
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```
pip install -r requirements.txt
```

4. **Prepare data**

Place Netflix raw data files in `data/raw/` and run preprocessing:

```
python src/data/preprocess.py
```

This generates processed splits in `data/processed/`.

---

## Running Experiments

- **Baselines**

```
python src/experiments/run_baselines.py
```

- **Matrix Factorization**

```
python src/experiments/run_mf_training.py
```

- **Neural Collaborative Filtering with hyperparameter tuning**

```
python src/experiments/run_ncf_training.py
```

---

## Notebooks

Interactive notebooks are available in the `notebooks/` directory for:

- Exploratory Data Analysis (`eda.ipynb`)
- Baseline models (`baselines.ipynb`)
- Matrix Factorization (`mf_training.ipynb`)
- Neural Collaborative Filtering (`ncf_training.ipynb`)

---

## Testing

Run all unit tests with:

```
python -m unittest discover tests
```

---

## Future Work

- Deploy models as REST API using FastAPI (stub included).
- Extend with hybrid models combining content features.
- Optimize models with advanced ranking losses and data augmentation.
- Containerize project with Docker for reproducible deployment.

---

## Contact

For any questions or contributions, please reach out at samrithashree23@gmail.com.

---

*Project developed by Samritha Shree as a comprehensive example of scalable recommender systems.*
```
