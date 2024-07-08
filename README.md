# Discgolf Throw Classification

This is the submission repository for the course "Data Science for Wearables" at the Prof. Dr. Arnich chair of the summer semester 2024.

## Installation

After creating a new Python environment, install the required dependencies for running this project:

```
pip install -r requirements.txt
```

## Getting started

### 1. Create Dataset

The data folder contains time series with multiple throws. To be able to train the classification model, first use our developed automatic and manual cutters (throw_cutter.py and manual_cutter.py) to retrieve a time series for each throw:

```
python manual_cutter.py
python create_dataset.py
```

### 2. Train models

After the datasets are ready, you can train our classification models like so:

```
python classification.py
```