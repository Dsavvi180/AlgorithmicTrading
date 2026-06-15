# AlgoTrading Repository

This repository contains a collection of jupyter notebooks for exploring trading strategies. Extensive data wrangling has been done for data processing, feature engineering and label creation for downstream use in machine learning or strategy research. 

The primary source of data can be found in the below repository, where raw futures contracts from Databento have been back adjusted to form a continuous NQ futures time series: 

- **marketDataNasdaqFutures/**
	- Provides raw and processed market data for Nasdaq futures, including minute-level CSVs, feature sets, metadata, and symbology files. This data is used for research, backtesting, and model training.

 - Data has been cleaned, outliers removed, and very importantly the futures contracts have been backadjusted to create a continuous time series.

*Pre-back-adjusted futures data: Raw contracts*

<img src="https://raw.githubusercontent.com/Dsavvi180/AlgorithmicTrading/main/images/preBackAdjustment.png" alt="Pre-backadjusted futures data" width="800">

*Back-adjusted continuous futures data*

<img src="https://raw.githubusercontent.com/Dsavvi180/AlgorithmicTrading/main/images/backAdjustedNQ.png" alt="Back-adjusted continuous futures data" width="800">

- **HMM-MarketRegime/**
  - Following a hidden markov model approach to market regime quantification 

## Getting Started
1. **Python Notebooks:**
	- Open any `.ipynb` file in Jupyter or VS Code to explore and run backtesting experiments.
 - 
## Notes
- All data and code are organized for rapid experimentation and research in algorithmic trading.

## Pipeline

<img src="https://raw.githubusercontent.com/Dsavvi180/AlgorithmicTrading/main/images/strategyDevelopmentPipeline.png" alt="Strategy Development Pipeline" width="600">

---
For questions or contributions, please open an issue or submit a pull request. Contact me via dsavvasavvi18@gmail.com
