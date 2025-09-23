# AlgoTrading Repository

This repository contains a collection of tools, notebooks, and web applications for algorithmic trading research, backtesting, and data analysis. It is organized into several key components:
## Repository Structure

- **backtesting/**
	- Contains Jupyter notebooks for strategy development and backtesting, including feature engineering and signal generation using Python (e.g., `backtrader.ipynb`).
   
- **marketDataNasdaqFutures/**
	- Provides raw and processed market data for Nasdaq futures, including minute-level CSVs, feature sets, metadata, and symbology files. This data is used for research, backtesting, and model training.

 - Data has been cleaned, outliers removed, and very importantly the futures contracts have been backadjusted to create a continuous time series.

*Pre-back-adjusted futures data: Raw contracts*

<img src="https://raw.githubusercontent.com/Dsavvi180/AlgorithmicTrading/main/preBackAdjustment.png" alt="Pre-backadjusted futures data" width="600">

*Back-adjusted continuous futures data*

<img src="https://raw.githubusercontent.com/Dsavvi180/AlgorithmicTrading/main/backAdjustedNQ.png" alt="Back-adjusted continuous futures data" width="600">

- **HMM-MarketRegime/**
  - Following a hidden markov model approach to market regime quantification 

- **ta-reversals/**
	- Includes research notebooks (e.g., `ta-reversals.ipynb`) focused on technical analysis and reversal strategies.
- **Transformers/DirectionalPredictions/**
	- Houses deep learning experiments for directional prediction using transformer models. Contains training/validation data, model artifacts, logs, and a requirements file for reproducibility. The `googleColab/` subfolder includes Colab-ready notebooks and datasets.


## Getting Started
1. **Python Notebooks:**
	- Open any `.ipynb` file in Jupyter or VS Code to explore and run backtesting experiments.

2. **Deep Learning Models:**
	- See `Transformers/DirectionalPredictions/` for transformer-based model training and evaluation.

## Notes
- All data and code are organized for rapid experimentation and research in algorithmic trading.

## Pipeline

<img src="https://raw.githubusercontent.com/Dsavvi180/AlgorithmicTrading/main/strategyDevelopmentPipeline.png" alt="Strategy Development Pipeline" width="500">

---
For questions or contributions, please open an issue or submit a pull request.
