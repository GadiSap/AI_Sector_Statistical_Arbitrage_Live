# Imports
import pandas as pd
import numpy as np
import yfinance as yf # To download historical market data from Yahoo! Finance.
import statsmodels.api as sm # Provides classes and functions for the estimation of statistical models
from statsmodels.tsa.stattools import coint # Used for performing the Engle-Granger cointegration test
from datetime import timedelta
import os


class TradingHr:
  """
  A class to manage a pairs trading strategy, including entry, exit, and stop-loss conditions.
  Trades are managed based on Z-scores of the spread between two cointegrated assets.
  """
  def __init__(self, trades_history_file_name = 'trades.csv', time_series_file_name = 'trade_time_series.csv', fee = 0.005, entry_threshold = 1.5, exit_threshold = 0.2, stop_loss_threshold = 4.5, enter_trade_max = 3.0, compounded_profit=False):
    """
    Initializes the TradingHr strategy parameters.

    Args:
        trades_history_file_name (str): The name of the CSV file to store trade history.
        time_series_file_name (str): The name of the CSV file to store time-series trade data (Z-score, Profit).
        fee (float): Transaction fee per trade as a percentage.
        entry_threshold (float): Z-score threshold to enter a new trade.
        exit_threshold (float): Z-score threshold to exit an open trade.
        stop_loss_threshold (float): Z-score threshold to stop-loss an open trade.
        enter_trade_max (float): Maximum absolute Z-score to allow entering a new trade.
        compounded_profit (bool): If True, profits are compounded; otherwise, they are added (default: False).
    """
    self.entry_threshold = entry_threshold
    self.exit_threshold = exit_threshold
    self.stop_loss_threshold = stop_loss_threshold
    self.enter_trade_max = enter_trade_max
    self.fee = fee
    self.trades_history_file_name = trades_history_file_name
    self.time_series_file_name = time_series_file_name
    self.compounded_profit = compounded_profit

    # Initialize the time series file with header if it doesn't exist
    if not os.path.exists(self.time_series_file_name):
        pd.DataFrame(columns=['Date', 'Ticker1', 'Ticker2', 'Z-score', 'Pair Profit', 'Intrade Status', 'Current Trade PnL']).to_csv(self.time_series_file_name, index=False)


  def sell(self, enter_price_stock1: float, enter_price_stock2: float, price_stock1: float, price_stock2: float, hedge_ratio: float) -> float:
    """
    Calculates the profit/loss for closing a pairs trade.

    Args:
        enter_price_stock1 (float): Entry price of Ticker1 (negative if shorted).
        enter_price_stock2 (float): Entry price of Ticker2 (negative if shorted).
        price_stock1 (float): Current price of Ticker1.
        price_stock2 (float): Current price of Ticker2.
        hedge_ratio (float): The hedge ratio between Ticker1 and Ticker2.

    Returns:
        float: The profit or loss of the closed pairs trade, net of fees.
    """
    # Calculate percentage change for Ticker1. If enter_price_stock1 was negative,
    # it signifies a short position, so we invert the profit calculation.
    prct_stock1 = (price_stock1 - abs(enter_price_stock1)) / abs(enter_price_stock1)
    if enter_price_stock1 < 0:
      prct_stock1 = -1 * prct_stock1

    # Calculate percentage change for Ticker2. If enter_price_stock2 was negative,
    # it signifies a short position, so we invert the profit calculation.
    prct_stock2 = (price_stock2 - abs(enter_price_stock2)) / abs(enter_price_stock2)
    if enter_price_stock2 < 0:
      prct_stock2 = -1 * prct_stock2

    # Calculate portfolio returns, accounting for the hedge ratio and transaction fee.
    # The hedge ratio (beta) scales the return of the second stock relative to the first.
    # The denominator (1 + abs(hedge_ratio)) normalizes the return by the total capital at risk.
    pair_returns = (prct_stock1 + hedge_ratio * prct_stock2) / (1 + abs(hedge_ratio)) - self.fee

    return pair_returns

  def trading(self, data_training: pd.DataFrame):
    """
    Executes the pairs trading strategy based on the current market data and trade history.
    It calculates Z-scores, checks for trade entry/exit/stop-loss conditions, and updates trade records.

    Args:
        data_training (pd.DataFrame): DataFrame containing historical 'Close' price data
                                      for the current look-back window, used for calculating
                                      the hedge ratio and spread statistics.
    """
    df_trade_history = pd.read_csv(self.trades_history_file_name).copy()
    # Ensure 'Trade Hedge Ratio' column exists, initialize if not.
    if 'Trade Hedge Ratio' not in df_trade_history.columns:
        df_trade_history['Trade Hedge Ratio'] = 0.0

    # Add check for empty data_training DataFrame right at the beginning
    if data_training.empty:
        print("Warning: data_training DataFrame is empty. Skipping trading for this period.")
        return # Exit the function if there's no data to process

    current_date = data_training.index[-1] # Get the current date for time-series recording
    time_series_records_for_this_day = [] # To store data for this day's snapshot

    # Iterate through each defined trading pair in the trade history.
    for i in range(df_trade_history.shape[0]):
      row = df_trade_history.loc[i].copy()

      ticker1 = row['Ticker1']
      ticker2 = row['Ticker2']

      # Check if both tickers are present in the current data_training DataFrame
      if ticker1 not in data_training.columns or ticker2 not in data_training.columns:
        print(f"Skipping pair {ticker1}/{ticker2} for this period due to missing data for one or both tickers.")
        # Record current state with NaN for Z-score if data is missing, but keep previous profit
        time_series_records_for_this_day.append({
            'Date': current_date,
            'Ticker1': ticker1,
            'Ticker2': ticker2,
            'Z-score': np.nan,
            'Pair Profit': row['profit'],
            'Two Months Profit': row['Two Months Profit'],
            'Intrade Status': row['intrade'],
            'Current Trade PnL': np.nan # No current trade PnL if data is missing
        })
        continue # Skip to the next pair

      # Additional check for sufficient valid data for OLS and price extraction
      if data_training[ticker1].isnull().all() or data_training[ticker2].isnull().all() or len(data_training[ticker1].dropna()) < 2 or len(data_training[ticker2].dropna()) < 2:
          print(f"Skipping pair {ticker1}/{ticker2} for this period due to insufficient valid data for OLS or price access.")
          # Record current state with NaN for Z-score if data is insufficient
          time_series_records_for_this_day.append({
              'Date': current_date,
              'Ticker1': ticker1,
              'Ticker2': ticker2,
              'Z-score': np.nan,
              'Pair Profit': row['profit'],
              'Two Months Profit': row['Two Months Profit'],
              'Intrade Status': row['intrade'],
              'Current Trade PnL': np.nan
          })
          continue # Skip to the next pair

      try:
          price_stock1 = data_training.iloc[-1][ticker1] # Get the most recent price for Ticker1
          price_stock2 = data_training.iloc[-1][ticker2] # Get the most recent price for Ticker2
      except KeyError as e:
          # This block should ideally not be reached if the above column check is robust.
          # However, as a defensive measure, catch unexpected KeyErrors here.
          print(f"ERROR: Unexpected KeyError when accessing prices for {ticker1}/{ticker2} at .iloc[-1][ticker]. Error: {e}. Skipping pair for this period.")
          time_series_records_for_this_day.append({
              'Date': current_date,
              'Ticker1': ticker1,
              'Ticker2': ticker2,
              'Z-score': np.nan,
              'Pair Profit': row['profit'],
              'Two Months Profit': row['Two Months Profit'],
              'Intrade Status': row['intrade'],
              'Current Trade PnL': np.nan
          })
          continue # Skip to the next pair

      # --- Core Calculations: Hedge Ratio, Spread, and Z-score ---
      # These calculations are performed for every pair, regardless of its 'status' or 'intrade' status,
      # as they are needed to assess potential trade actions or manage existing ones.

      # Calculate the hedge ratio (beta) using OLS regression. Ticker1 is the dependent variable.
      X_train = sm.add_constant(data_training[ticker2]) # Corrected: add_add_constant to add_constant
      model = sm.OLS(data_training[ticker1], X_train).fit()
      hedge_ratio = model.params[ticker2]

      # Calculate the spread: Ticker1's price minus hedge_ratio * Ticker2's price.
      # The spread represents the difference between the two assets, adjusted by their historical relationship.
      spread = data_training[ticker1] - hedge_ratio * data_training[ticker2]

      # Calculate the mean and standard deviation of the spread over the `data_training` period.
      z_mean = spread.mean()
      z_std = spread.std()

      # Calculate the current Z-score of the spread.
      # The Z-score measures how many standard deviations the current spread is from its mean.
      current_z_score = (spread.iloc[-1] - z_mean) / z_std

      # Initialize current_pair_profit with the already realized profit
      cumulative_realized_profit = row['profit']
      current_pair_profit_for_timeseries = cumulative_realized_profit # Default to realized profit
      current_trade_pnl = np.nan # Initialize Current Trade PnL for time-series

      # --- Trade Management Logic ---

      # First, check if there is an open trade (intrade == 'yes') that needs to be closed.
      if row['intrade'] == 'yes':
        # Calculate unrealized profit for the *current open trade* (percentage return for this trade)
        enter_price_stock1 = row['Ticker1 Buy Price']
        enter_price_stock2 = row['Ticker2 Buy Price']
        # Use the stored hedge ratio for calculations related to an open trade
        unrealized_profit_current_trade = self.sell(enter_price_stock1, enter_price_stock2, price_stock1, price_stock2, row['Trade Hedge Ratio'])
        current_trade_pnl = unrealized_profit_current_trade # Record for time series

        # Calculate the *total profit for the pair* if this open trade were closed now.
        if self.compounded_profit:
            # Compounded profit calculation
            if cumulative_realized_profit == 0.0:
                current_pair_profit_for_timeseries = unrealized_profit_current_trade
            else:
                current_pair_profit_for_timeseries = (1 + cumulative_realized_profit) * (1 + unrealized_profit_current_trade) - 1
        else:
            # Simple addition profit calculation
            current_pair_profit_for_timeseries = cumulative_realized_profit + unrealized_profit_current_trade

        # Now, check for closing conditions
        if abs(current_z_score) <= self.exit_threshold or abs(current_z_score) >= self.stop_loss_threshold:
          # Trade is being closed. Update the *realized* cumulative profit for the pair.
          if self.compounded_profit:
              if row['profit'] == 0.0:
                  row['profit'] = unrealized_profit_current_trade
              else:
                  row['profit'] = (1 + row['profit']) * (1 + unrealized_profit_current_trade) - 1

              if row['Two Months Profit'] == 0.0:
                  row['Two Months Profit'] = unrealized_profit_current_trade
              else:
                  row['Two Months Profit'] = (1 + row['Two Months Profit']) * (1 + unrealized_profit_current_trade) - 1
          else:
              # Simple addition
              row['profit'] = row['profit'] + unrealized_profit_current_trade
              row['Two Months Profit'] = row['Two Months Profit'] + unrealized_profit_current_trade

          row['intrade'] = 'no'         # Mark the trade as closed.
          row['Ticker1 Buy Price'] = 0.0 # Reset entry prices.
          row['Ticker2 Buy Price'] = 0.0
          row['Trade Hedge Ratio'] = 0.0 # Reset stored hedge ratio

          current_pair_profit_for_timeseries = row['profit'] # Update for time series
      # Second, if the pair is 'active' and there is no trade currently open, check for new entry opportunities.
      # New trades are initiated when the Z-score crosses the entry threshold.
      # 'active' status indicates that the pair is eligible for new trades.
      elif row['status'] == 'active' and row['intrade'] == 'no':
        if current_z_score >= self.entry_threshold and current_z_score <= self.enter_trade_max: # Z-score is high, spread is wide: Short Ticker1, Long Ticker2
          row['intrade'] = 'yes'
          row['Ticker1 Buy Price'] = -price_stock1 # Store negative price to indicate short position.
          row['Ticker2 Buy Price'] = price_stock2  # Store positive price to indicate long position.
          row['Trade Hedge Ratio'] = hedge_ratio # Store the hedge ratio at the time of entry

        elif current_z_score <= -self.entry_threshold and current_z_score >= -self.enter_trade_max: # Z-score is low, spread is narrow: Long Ticker1, Short Ticker2
          row['intrade'] = 'yes'
          row['Ticker1 Buy Price'] = price_stock1  # Store positive price to indicate long position.
          row['Ticker2 Buy Price'] = -price_stock2 # Store negative price to indicate short position.
          row['Trade Hedge Ratio'] = hedge_ratio # Store the hedge ratio at the time of entry

      df_trade_history.loc[i] = row

      # Capture data for time series tracking for this pair and this day
      time_series_records_for_this_day.append({
          'Date': current_date,
          'Ticker1': ticker1,
          'Ticker2': ticker2,
          'Z-score': current_z_score,
          'Pair Profit': current_pair_profit_for_timeseries, # This is the summed cumulative profit for the pair up to this point
          'Intrade Status': row['intrade'],
          'Current Trade PnL': current_trade_pnl,
          'Two Months Profit': row['Two Months Profit']
      })

    # After iterating through all pairs, save the updated trade history to the CSV file.
    df_trade_history.to_csv(self.trades_history_file_name, index=False)

    # Append current day's time-series data to the time series file
    if time_series_records_for_this_day:
        df_time_series_today = pd.DataFrame(time_series_records_for_this_day)
        df_time_series_today.to_csv(self.time_series_file_name, mode='a', header=False, index=False)

def day_trade(df_AI_data, trades_history_file_name, time_series_file_name, fee = 0.005, entry_threshold = 1.5, exit_threshold = 0.2, stop_loss_threshold = 4.5, enter_trade_max = 3.0):
  """
  This function simulates the execution of the `TradingHr` strategy for a single trading day,
  processing data in trades_history_file_name and time_series_file_name.
  """
  for i in range(6, -0, -1):
    strategy = TradingHr(trades_history_file_name, time_series_file_name, fee, entry_threshold, exit_threshold, stop_loss_threshold, enter_trade_max)
    # Call the trading function
    strategy.trading(df_AI_data[:-i])
  strategy = TradingHr(trades_history_file_name, time_series_file_name, fee, entry_threshold, exit_threshold, stop_loss_threshold, enter_trade_max)
  # Call the trading function
  strategy.trading(df_AI_data)


