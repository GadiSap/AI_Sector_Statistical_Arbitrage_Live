# Pairs Trading Strategy with AI Stock Tickers
This project implements a dynamic pairs trading strategy focused on Artificial Intelligence (AI) related stock tickers. The system features a robust Meta-Optimization pipeline to identify statistically cointegrated pairs and a Walk-Forward Simulation to trade them under realistic market conditions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Research & Training Pipeline](#research--training-pipeline)
3. [Execution (Simulation)](#execution-simulation)
4. [Setup and Dependencies](#setup-and-dependencies)
5. [Usage](#usage)
6. [Key Files and Outputs](#key-files-and-outputs)

## Project Overview
This project aims to capture Alpha from the mean-reverting nature of cointegrated stock pairs within the volatile AI sector. The workflow is divided into two distinct phases:

*   **Research (Training)**: A brute-force grid search across hundreds of parameter combinations using historical data to find the most stable pairs and thresholds.
*   **Execution (Simulation)**: A daily trading loop that applies locked hedge ratios and Z-score logic to simulate live-market performance.

## Research & Training Pipeline
To mitigate overfitting and identify structural relationships, the project includes a `Training.py` script that executes a high-performance evaluation of strategy parameters:

*   **Parameter Grid Search**: Systematically tests combinations of Entry/Exit thresholds, rolling windows, and hedge lookbacks.
*   **Parallel Computing**: Utilizes Python's `multiprocessing` to distribute 900+ simulation combinations across all available CPU cores for maximum efficiency.
*   **Stability Validation**: Pairs are filtered not just by total return, but by Sharpe Stability and Annual Return Stability ratios between training and testing periods to optimize for long-term consistency.
*   **Data Utility**: This data is used to find the best parameters for the trading strategy and for selecting the most robust pairs to trade.

## Execution (Simulation)
Once parameters are optimized, the strategy enters the execution phase, which simulates a live trading environment with the following logic:

*   **Optimization and Pair Selection**: Performs Cointegration Testing and Backtesting over training and testing periods to select the specific pairs to trade based on the selection criteria determined in the Research & Training phase.
*   **Hourly Decision Engine**: Processes market data on an hourly basis to calculate Z-scores and manage positions.
*   **Rolling & Locked Hedge Ratios**: Calculates a 90-day rolling hedge ratio (Beta) while a pair is "flat." Once a position is initiated, the hedge ratio is locked to reflect real-world execution where share quantities remain fixed until the trade is closed.
*   **Risk Controls**: Implements an `enter_trade_max` filter to block entries during extreme volatility shocks (non-reverting moves) and a hard stop-loss at a 3.5σ deviation.
*   **Transaction Fee Modeling**: Deducts a 0.5% fee per trade to provide a realistic assessment of net profitability.
*   **Adaptive Re-optimization**: Automatically triggers a full re-optimization every 60 days to prune underperforming pairs and rotate into new cointegrated opportunities.

## Setup and Dependencies
Install the required libraries via `pip`:

```bash
pip install pandas numpy yfinance statsmodels tqdm
```

## Usage
### 1. Strategy Training (Optimization)
Run the training script to find the best parameters for the current market regime:

```bash
python Training.py
```
This generates a comprehensive CSV in `files/training/` detailing how different thresholds and windows performed across training, testing, and simulation phases.

### 2. Trading Simulation
Update the parameters in the main simulation script based on your training results and run:

*   **Initial Selection**: Orchestrates pair selection via the `training_and_optimization` function.
*   **Daily Simulation**: The script loops through the specified date range, downloading hourly data and updating trade records in real-time.

## Key Files and Outputs
*   `training_trades_history_hedge.csv`: The primary output of the meta-optimization, used to select the final strategy parameters and active pairs.
*   `trades_history.csv`: Stores the current status, entry prices, locked hedge ratios, and cumulative profit for all active trading pairs.
*   `trade_time_series.csv`: Granular hourly records of Z-scores, unrealized PnL, and trade statuses for in-depth analysis used for plotting.
*   `all_pair_results_YYYY-MM-DD.csv`: Detailed performance breakdown for every potential pair identified during the selection phase.
