# Fine-Grained Fake News Detection

A research framework for fine-grained fake-news detection using the LIAR benchmark dataset. The project combines contextual text semantics with relational modeling in a graph-transformer-style pipeline.

## Goals

- Prepare and validate the LIAR dataset without data leakage
- Engineer linguistic and credibility-related features
- Compare baseline, transformer, and graph-based models
- Evaluate fine-grained classification performance
- Provide reproducible scripts, configuration, and tests

## Technology

- Python
- PyTorch and/or transformer-based modeling components
- Graph neural network components
- YAML configuration
- pytest

The exact dependency versions are defined in [`requirements.txt`](requirements.txt).

## Project structure

```text
configs/               YAML configuration files
  default.yaml
  data_config.yaml
data/                  Raw and processed datasets
Documentation/         Research documents and exploratory analysis
scripts/               Data preparation, EDA, and training entry points
src/
  data/                Loaders, preprocessors, and dataset abstractions
  features/            Feature engineering and credibility metrics
  models/              Baselines, transformers, and graph models
  utils/               Configuration, logging, and evaluation utilities
tests/                 Unit and integration tests
requirements.txt       Python dependencies
```

## Getting started

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\\Scripts\\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare data

Place the permitted LIAR dataset files in the expected `data/raw/` location, then run:

```bash
python scripts/prepare_data.py
```

### 4. Run exploratory analysis

```bash
python scripts/run_eda.py
```

### 5. Run tests

```bash
pytest tests/
```

## Reproducibility and data use

The LIAR dataset is an external benchmark. Review its original terms and citation requirements before redistributing or publishing derived data. Keep raw data, generated artifacts, and experiment outputs separate from source code where appropriate.

## Research status

This repository is intended for research and experimentation. Record configuration files, dataset versions, random seeds, and evaluation metrics when comparing experiments.

## License

No license has been specified for this repository yet.

## Author

Created by [Aaron Parayil](https://github.com/aaronparayil).
