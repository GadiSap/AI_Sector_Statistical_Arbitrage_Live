# Imports
import pandas as pd
import numpy as np
import yfinance as yf # To download historical market data from Yahoo! Finance.
import statsmodels.api as sm # Provides classes and functions for the estimation of statistical models
from statsmodels.tsa.stattools import coint # Used for performing the Engle-Granger cointegration test
from datetime import timedelta
import os
from TradingHr import TradingHr
from TradingHr import day_trade
from PairsTradingManager import PairsTradingManager
from PairsTradingManager import training_and_optimization


ai_tickers = [
    "NVDA", "AMD", "AVGO", "TSM", "ASML", "ARM", "MU", "AMAT", "LRCX", "KLAC",
    "INTC", "MRVL", "QCOM", "ADI", "TXN", "TER", "NVMI", "ENTG", "MKSI",
    "LSCC", "MPWR", "AEIS", "TSEM", "SMCI", "MSFT", "AMZN", "GOOGL", "ORCL",
    "VRT", "DELL", "ANET", "EQIX", "DLR", "CSCO", "HPE", "AKAM", "WDC", "STX",
    "PLTR", "META", "ADBE", "CRM", "SNOW", "MDB", "PATH", "AI", "DT", "DDOG",
    "S", "CRWD", "ZS", "PANW", "NET", "NOW", "WDAY", "TEAM", "HUBS", "APP",
    "U", "TEM", "DOCU", "GTLB", "MBLY", "AUR", "PONY", "WRD", "SYM", "TSLA",
    "SOUN", "UPST", "RXRX", "BBAI", "INOD", "ODD", "SERV", "PDYN", "CRNC", "AISP",
    "KSCP", "LTRN", "BFRG", "TCEHY", "BABA", "BIDU", "CDNS", "SNPS",
    "PSTG", "KVYO", "ESTC", "FRSH", "IOT", "MSTR"
]

# Define file names for trade history and time series data
trades_history_file_name = 'trades_history.csv' # File to store the history of all trades made by the strategy.
time_series_file_name = 'trade_time_series.csv' # File to store time-series data for analysis and plotting (Z-score, profits, etc.).
# Parameters:
initial_date_ref_str = '2026-01-01' # Starting point for the initial training and optimization phase.
date = '2026-04-16' # The end date for the entire simulation period.
range_days = 104  # The total number of days the simulation will run for.
days_back = 18 # Number of past days to consider for fetching historical data for daily trading decisions.
reoptimization_days = 60 # Frequency (in days) at which the strategy re-optimizes its trading pairs.

entry_threshold_param = 1.5 # Z-score threshold to enter a new trade.
exit_threshold_param = 0.5 # Z-score threshold to exit an open trade.
stop_loss_threshold_param = 3.5 # Z-score threshold to trigger a stop-loss and close an open trade.
enter_trade_max_param = 2.5 # Maximum absolute Z-score to allow entering a new trade; prevents trading in extreme conditions.
window_param = 90 # Rolling window size in hr for calculating spread mean and standard deviation for the training and optimization phase.
p_min_coint_param = 0.05 # P-value threshold for the cointegration test; pairs with p-value below this are considered cointegrated.
fee_param = 0.005 # Transaction fee per trade, expressed as a percentage.
min_training_return_param = 1 # Minimum annualized return required for a pair during the training period to be considered optimized.
min_training_sharpe_param = 1 # Minimum Sharpe Ratio required for a pair during the training period.
min_training_trades_param = 0 # Minimum number of entry trades required for a pair during the training period.
max_training_drawdown_param = -0.3 # Maximum acceptable drawdown for a pair during the training period (e.g., -0.3 means no more than 30% loss).
min_testing_return_param = 1 # Minimum annualized return required for a pair during the testing period.
min_testing_sharpe_param = 1 # Minimum Sharpe Ratio required for a pair during the testing period.
max_testing_drawdown_param = -0.3 # Maximum acceptable drawdown for a pair during the testing period.
min_sharpe_ratio_stability_param = 0.5 # Minimum ratio of Testing Sharpe Ratio to Training Sharpe Ratio, ensuring consistent performance.

# --- Initial Training and Optimization ---
# This step identifies the initial set of cointegrated pairs and optimizes their parameters.
# The results are stored in trades_history_file_name .
training_and_optimization(ai_tickers, initial_date_ref_str, trades_history_file_name,
                 entry_threshold=entry_threshold_param, exit_threshold=exit_threshold_param, stop_loss_threshold=stop_loss_threshold_param,
                 window=window_param, p_min_coint=p_min_coint_param, fee=fee_param, enter_trade_max=enter_trade_max_param,
                 min_training_return=min_training_return_param, min_training_sharpe=min_training_sharpe_param, min_training_trades=min_training_trades_param,
                 max_training_drawdown=max_training_drawdown_param, min_testing_return=min_testing_return_param, min_testing_sharpe=min_testing_sharpe_param,
                 max_testing_drawdown=max_testing_drawdown_param, min_sharpe_ratio_stability=min_sharpe_ratio_stability_param)
# --- Daily Trading Simulation ---
# Initialize the time series file to ensure a clean start with all expected columns.
# This file will record daily Z-scores, pair profits, and trade statuses for later analysis.

pd.DataFrame(columns=['Date', 'Ticker1', 'Ticker2', 'Z-score', 'Pair Profit', 'Intrade Status', 'Current Trade PnL']).to_csv(time_series_file_name, index=False)

print("\nStarting daily trading simulation...")

# Loop through each day in the specified range to simulate trading
for i in range(range_days):
  # Calculate the end date for the current day's data download
  end_date = (pd.to_datetime(date) - timedelta(days=(range_days-1-i))).strftime('%Y-%m-%d')


  start_date = (pd.to_datetime(end_date) - timedelta(days=days_back)).strftime('%Y-%m-%d')

  # Download hourly data for the selected AI tickers for the current look-back window
  df_AI_daily = yf.download(ai_tickers, start=start_date, end=end_date, interval='1h', auto_adjust=True)['Close']
  df_AI_daily = df_AI_daily.dropna() # Drop rows with any missing values

  # Check for empty DataFrame after download and dropna
  if df_AI_daily.empty:
    print(f"Skipping trade for {end_date} due to empty DataFrame after download and dropna.")
    continue # Skip to the next day if no valid data

  # Execute the daily trading strategy
  day_trade(df_AI_daily, trades_history_file_name, time_series_file_name, fee_param, entry_threshold_param, exit_threshold_param, stop_loss_threshold_param, enter_trade_max_param)

  # Check re-optimization
  today = pd.to_datetime(end_date)
  if today.day_of_year % reoptimization_days == 0:
    print(f"\nPerforming re-optimization for pairs at {end_date}...")
    training_and_optimization(ai_tickers, initial_date_ref_str=end_date, trades_history_file_name=trades_history_file_name,
                              entry_threshold=entry_threshold_param, exit_threshold=exit_threshold_param, stop_loss_threshold=stop_loss_threshold_param,
                              window=window_param, p_min_coint=p_min_coint_param, fee=fee_param, enter_trade_max=enter_trade_max_param,
                              min_training_return=min_training_return_param, min_training_sharpe=min_training_sharpe_param, min_training_trades=min_training_trades_param,
                              max_training_drawdown=max_training_drawdown_param, min_testing_return=min_testing_return_param, min_testing_sharpe=min_testing_sharpe_param,
                              max_testing_drawdown=max_testing_drawdown_param, min_sharpe_ratio_stability=min_sharpe_ratio_stability_param)

print("\nDaily trading simulation completed.")

