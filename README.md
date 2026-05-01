# Pairs Trading Strategy with AI Stock Tickers

This notebook implements a dynamic pairs trading strategy focused on Artificial Intelligence (AI) related stock tickers. The strategy identifies statistically cointegrated pairs, optimizes their trading parameters over training and testing periods, and simulates daily trading with periodic re-optimization to adapt to changing market conditions.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Features](#features)
3.  [Setup and Dependencies](#setup-and-dependencies)
4.  [Usage](#usage)
5.  [Key Files and Outputs](#key-files-and-outputs)
6.  [Analysis and Visualization](#analysis-and-visualization)

## Project Overview
This project aims to profit from the mean-reverting nature of cointegrated stock pairs. It involves:
*   **Cointegration Testing**: Identifying pairs of stocks that move together over the long term.
*   **Strategy Optimization**: Backtesting identified pairs to find best pairs to trade based on historical performance.
*   **Daily Trading Simulation**: Executing trades based on Z-scores of the spread between pair prices.
*   **Periodic Re-optimization**: Adapting the set of active trading pairs and their parameters to maintain strategy effectiveness.

## Features
*   **Dynamic Pair Selection**: Utilizes Engle-Granger cointegration test to find suitable pairs from a given list of tickers.
*   **Parameter Optimization**: Evaluates pairs based on training and testing period performance metrics (returns, Sharpe ratio, drawdown, number of trades).
*   **Robust Trade Management**: Implements entry, exit, and stop-loss conditions based on Z-scores of the spread.
*   **Hourly Data Backtesting**: Simulates trading using hourly price data, providing granular insights.
*   **Trade History and Time-Series Tracking**: Records detailed trade information and daily snapshots of Z-scores and profits.

## Setup and Dependencies
To run this notebook, you'll need the following Python libraries. You can install them using pip:

```bash
!pip install pandas numpy yfinance statsmodels 
```

Ensure your Python environment has these installed. This notebook specifically uses `yfinance` to download historical market data and `statsmodels` for cointegration testing and OLS regression.

## Usage
1.  **Define AI Tickers**: Customize the `ai_tickers` list with the stock symbols you want to analyze.
2.  **Set Parameters**: Adjust the various trading and optimization parameters, such as `entry_threshold_param`, `exit_threshold_param`, `stop_loss_threshold_param`, `reoptimization_days`, and minimum performance criteria.
3.  **Initial Training and Optimization**: The notebook first performs an initial training and optimization phase to identify promising pairs. This involves downloading historical data, running cointegration tests, and backtesting pairs over defined training and testing periods.
4.  **Daily Trading Simulation**: After initial setup, the script loops through a specified `range_days`, simulating daily trading. For each day, it downloads the latest hourly data, applies the trading strategy to active pairs, and updates trade records.
5.  **Periodic Re-optimization**: Every `reoptimization_days`, the strategy re-runs the optimization process to refresh its set of active pairs, ensuring adaptability to changing market dynamics.

### Key Functions:
*   `TradingHr` class: Manages individual trades, calculates profits, and applies entry/exit/stop-loss logic on data simulating "live" trading.
*   `PairsTradingManager` class: Orchestrates cointegration tests, pair performance analysis, and selection of optimized pairs on "older" data.
*   `training_and_optimization` function: Drives the initial and periodic pair selection process.
*   `day_trade` function (in TradingHr.py) : Executes the hourly trading logic for a given day.

## Key Files and Outputs
*   trades_history_file_name: Stores the final status and cumulative profit for each traded pair.
*   time_series_file_name: Records detailed time-series data for each pair, including Z-scores, current profits, and trade statuses, enabling granular analysis and plotting.
*   `all_pair_results_YYYY-MM-DD.csv`: Generated during optimization, this file contains performance metrics for all tested pairs (training and testing periods).
*   `full_coint_results_initial_testing.csv`: Contains the results of all cointegration tests performed during the initial optimization phase.
