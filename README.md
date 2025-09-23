# AlgoTrading Repository

This repository contains a collection of tools, notebooks, and web applications for algorithmic trading research, backtesting, and data analysis. It is organized into several key components:
## Repository Structure

- **backtesting/**
	- Contains Jupyter notebooks for strategy development and backtesting, including feature engineering and signal generation using Python (e.g., `backtrader.ipynb`).
- **marketDataNasdaqFutures/**
	- Provides raw and processed market data for Nasdaq futures, including minute-level CSVs, feature sets, metadata, and symbology files. This data is used for research, backtesting, and model training.

 - Data has been cleaned, outliers removed, and very importantly the futures contracts have been backadjusted to create a continuous time series.

- **ta-reversals/**
	- Includes research notebooks (e.g., `ta-reversals.ipynb`) focused on technical analysis and reversal strategies.
- **Transformers/DirectionalPredictions/**
	- Houses deep learning experiments for directional prediction using transformer models. Contains training/validation data, model artifacts, logs, and a requirements file for reproducibility. The `googleColab/` subfolder includes Colab-ready notebooks and datasets.

## Web-Based Backtesting UI
The repository features a modern, responsive web application for backtesting and visualizing trading strategies using TradingView (Lightweight Charts). Key features:

- **Auto-loads the first 20,000 rows of the main NQ futures CSV on page load** (no manual file selection required).
- **Slick, full-viewport UI** with a header, chart, metrics dock, and settings drawer.
- **No extraneous controls**—the interface is clean and focused on data visualization and strategy evaluation.

## Getting Started
1. **Python Notebooks:**
	- Open any `.ipynb` file in Jupyter or VS Code to explore and run backtesting experiments.

2. **Deep Learning Models:**
	- See `Transformers/DirectionalPredictions/` for transformer-based model training and evaluation.

## Notes
- All data and code are organized for rapid experimentation and research in algorithmic trading.

## Pipeline

![Alt text]['Algorithmic Trading'/strategyDevelopmentPipeline.png]

---
For questions or contributions, please open an issue or submit a pull request.
