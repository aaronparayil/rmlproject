# Fine-Grained Fake News Detection (Graph-Transformer Hybrid)

This repository implements the research framework for **Fine-Grained Fake News Detection** on the **LIAR benchmark dataset** (Wang, ACL 2017), combining contextual text semantics with relational metadata and credit history modeling.

## Project Structure

`
.
├── configs/               # YAML configuration files for data and models
│   ├── default.yaml
│   └── data_config.yaml
├── data/
│   ├── raw/               # Raw LIAR TSV splits (train, valid, test)
│   └── processed/         # Cleaned, leakage-free engineered datasets
├── Documentation/         # Research paper PDF and preliminary EDA notebooks
├── scripts/               # Executable CLI scripts (data preparation, EDA, training)
│   ├── prepare_data.py
│   └── run_eda.py
├── src/                   # Modular Python package
│   ├── data/              # TSV/CSV loaders, preprocessors, and dataset abstractions
│   ├── features/          # Feature engineering, credibility scoring & text metrics
│   ├── models/            # Model implementations (Baselines, Transformers, GNNs)
│   └── utils/             # Logging, YAML configuration, and evaluation metrics
├── tests/                 # Unit and integration test suite
└── requirements.txt       # Dependencies
`

## Quick Start

### 1. Install Dependencies
`ash
pip install -r requirements.txt
`

### 2. Preprocess Data and Engineer Features
Run the data preparation pipeline:
`ash
python scripts/prepare_data.py
`

### 3. Run Tests
`ash
pytest tests/
`
