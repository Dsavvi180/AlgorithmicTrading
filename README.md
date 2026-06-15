# AlgoTrading Repository

This repository contains a collection of Jupyter notebooks exploring trading strategies, with extensive data wrangling for data processing, feature engineering, and label creation for downstream use in machine learning and strategy research.

**Please note:** some notebooks are incomplete and the underlying ideas haven't been fully implemented (due to university commitments).

The primary data source is described below, where raw NQ futures contracts from Databento have been back-adjusted to form a continuous time series:

- **marketDataNasdaqFutures/**
  - Provides raw and processed market data for Nasdaq futures, including minute-level CSVs, feature sets, metadata, and symbology files. Used for research, backtesting, and model training.
  - Data has been cleaned, outliers removed, and the futures contracts back-adjusted to create a continuous time series.

*Pre-back-adjusted futures data: raw contracts*
<img src="https://raw.githubusercontent.com/Dsavvi180/AlgorithmicTrading/main/images/preBackAdjustment.png" alt="Pre-backadjusted futures data" width="800">

*Back-adjusted continuous futures data*
<img src="https://raw.githubusercontent.com/Dsavvi180/AlgorithmicTrading/main/images/backAdjustedNQ.png" alt="Back-adjusted continuous futures data" width="800">

- **RandomResearch/HMM-MarketRegime/**
  - Hidden Markov Model approach to market regime quantification.

## Getting Started

1. **Python Notebooks**
   - Open any `.ipynb` file in Jupyter or VS Code to explore and run backtesting experiments.

## Notes

- All data and code are organized for rapid experimentation and research in algorithmic trading.

## Pipeline

<img src="https://raw.githubusercontent.com/Dsavvi180/AlgorithmicTrading/main/images/strategyDevelopmentPipeline.png" alt="Strategy Development Pipeline" width="600">

---

For questions or contributions, please open an issue or submit a pull request. Contact me via dsavvasavvi18@gmail.com
