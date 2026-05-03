# Imports
import pandas as pd
import numpy as np
import yfinance as yf # To download historical market data from Yahoo! Finance.
import statsmodels.api as sm # Provides classes and functions for the estimation of statistical models
from statsmodels.tsa.stattools import coint # Used for performing the Engle-Granger cointegration test
from datetime import timedelta
import os

class PairsTradingManager:
  """
  Manages the entire pairs trading workflow including cointegration testing,
  strategy optimization, and daily trading with bi-weekly pair updates.
  """

  def __init__(self, ai_tickers, trades_file='trades.csv',
               entry_threshold=1.5, exit_threshold=0.2, stop_loss_threshold=4.5,
               window=90, p_min_coint=0.05, fee=0.005,
               min_training_return=1, min_training_sharpe=1, min_training_trades=0,
               max_training_drawdown=-0.4, min_testing_return=0.8, min_testing_sharpe=1,
               max_testing_drawdown=-0.4, min_sharpe_ratio_stability=0.3, enter_trade_max = 3.0, min_two_months_profit_for_active=0.03, results_output_dir='.'):
    """
    Initializes the PairsTradingManager.

    Args:
        ai_tickers (list): List of AI stock tickers to consider.
        trades_file (str): CSV file to store and update active trading pairs.
        entry_threshold (float): Z-score threshold for entering a trade.
        exit_threshold (float): Z-score threshold to exit an trade.
        stop_loss_threshold (float): Z-score threshold to trigger a stop loss.
        window (int): Rolling window size for mean and standard deviation calculations.
        p_min_coint (float): P-value threshold for cointegration test.
        fee (float): Transaction fee per trade as a percentage.
    """
    self.ai_tickers = ai_tickers
    self.trades_file = trades_file
    self.entry_threshold = entry_threshold
    self.exit_threshold = exit_threshold
    self.stop_loss_threshold = stop_loss_threshold
    self.window = window
    self.p_min_coint = p_min_coint
    self.fee = fee
    self.annualization_factor = 252 * 6.5 # Assumes 252 trading days/year * 6.5 trading hours/day for hourly data.
    self.enter_trade_max = enter_trade_max

    # Optimization parameters
    self.min_training_return = min_training_return
    self.min_training_sharpe = min_training_sharpe
    self.min_training_trades = min_training_trades
    self.max_training_drawdown = max_training_drawdown
    self.min_testing_return = min_testing_return
    self.min_testing_sharpe = min_testing_sharpe
    self.max_testing_drawdown = max_testing_drawdown
    self.min_sharpe_ratio_stability = min_sharpe_ratio_stability
    self.min_two_months_profit_for_active = min_two_months_profit_for_active
    self.results_output_dir = results_output_dir

  def _perform_coint_test(self, df_AI_data, start_date, end_date):
    """
    Performs Engle-Granger cointegration test on all possible pairs.

    Args:
        df_AI_data (pd.DataFrame): DataFrame with historical data for all tickers.
        start_date (str): Start date for cointegration test data.
        end_date (str): End date for cointegration test data.

    Returns:
        tuple:
            - list: A list of tuples, where each tuple contains (Ticker1, Ticker2, pvalue, score, crit_value)
                    for cointegrated pairs (pvalue < p_min_coint).
            - list: A list of dictionaries, where each dictionary contains (

                    for ALL tested pairs.
    """
    results_list = [] # To store full results for all pairs
    stock_pairs_p_min = [] # To store filltred pairs
    # Slicing the DataFrame for the relevant period for cointegration testing.
    df_AI_coint = df_AI_data.loc[start_date: end_date]

    for i in range(len(self.ai_tickers)):
      for j in range(i + 1, len(self.ai_tickers)):
        ticker1 = self.ai_tickers[i]
        ticker2 = self.ai_tickers[j]
        # Ensure both tickers exist in the sliced DataFrame and have sufficient data.
        if ticker1 in df_AI_coint.columns and ticker2 in df_AI_coint.columns and \
           not df_AI_coint[ticker1].isnull().all() and not df_AI_coint[ticker2].isnull().all() and \
           len(df_AI_coint[ticker1].dropna()) > 1 and len(df_AI_coint[ticker2].dropna()) > 1:
          try:
            score, pvalue, crit_value = coint(df_AI_coint[ticker1], df_AI_coint[ticker2])
          except ValueError:
            # Handle cases where coint might raise ValueError due to singular matrix (e.g., all same values)
            results_list.append({'Ticker1': ticker1, 'Ticker2': ticker2, 'p-value': np.nan, 'Score': np.nan, 'Crit_Value': np.nan, 'Error': 'ValueError in coint test'})
            continue
          except Exception as e:
            # Catch other potential errors during cointegration test
            print(f"Error performing coint test for {ticker1}/{ticker2}: {e}")
            results_list.append({'Ticker1': ticker1, 'Ticker2': ticker2, 'p-value': np.nan, 'Score': np.nan, 'Crit_Value': np.nan, 'Error': str(e)})
            continue
          else:
            results_list.append({'Ticker1': ticker1, 'Ticker2': ticker2, 'p-value': pvalue, 'Score': score, 'Crit_Value': crit_value})
            if pvalue < self.p_min_coint:
              stock_pairs_p_min.append((ticker1, ticker2, pvalue, score, crit_value))
        else:
          # Record pairs that were skipped due to insufficient data
          results_list.append({'Ticker1': ticker1, 'Ticker2': ticker2, 'p-value': np.nan, 'Score': np.nan, 'Crit_Value': np.nan, 'Error': 'Insufficient data'})

    return stock_pairs_p_min, results_list


  def _simulate_trades_and_calculate_returns(self, data_period, hedge_ratio, ticker1, ticker2):
    """
    Helper function to simulate trades and calculate performance metrics for a given period (training or testing).

    Args:
        data_period (pd.DataFrame): DataFrame containing historical 'Close' price data for the period.
        hedge_ratio (float): The hedge ratio calculated from the training period OLS regression.
        ticker1 (str): The symbol of the first stock in the pair.
        ticker2 (str): The symbol of the second stock in the pair.

    Returns:
        tuple:
            - annualized_returns (float or None): Annualized return of the pairs trading strategy.
            - annualized_returns_ticker1 (float or None): Annualized return of ticker1 (buy and hold).
            - annualized_returns_ticker2 (float or None): Annualized return of ticker2 (buy and hold).
            - num_entry_trades (int): Number of entry trades made.
            - sharpe_ratio (float or None): Sharpe Ratio for the period.
            - max_drawdown (float or None): Maximum Drawdown for the period.
            - z_score (pd.Series): Z-score time series for the spread.
            - cumulative_returns (pd.Series): Cumulative returns of the pairs trading strategy.
            - cumulative_returns_ticker1 (pd.Series): Cumulative returns of ticker1 (buy and hold).
            - cumulative_returns_ticker2 (pd.Series): Cumulative returns of ticker2 (buy and hold).
            - portfolio_returns (pd.Series): Raw portfolio returns for each period.
    """
    # Calculate the spread: Ticker1's price minus hedge_ratio * Ticker2's price.
    spread = data_period[ticker1] - hedge_ratio * data_period[ticker2]
    # Calculate rolling mean and standard deviation for the spread.
    rolling_mean = spread.rolling(window=self.window).mean()
    rolling_std = spread.rolling(window=self.window).std()
    # Calculate the Z-score for the spread.
    z_score = (spread - rolling_mean) / rolling_std

    # Initialize position for the period. A value of 0 means no open position.
    position = pd.Series(0, index=data_period.index)
    num_entry_trades = 0

    # Simulate trades based on Z-score and thresholds.
    # This loop iterates through the Z-score time series to determine when to enter or exit trades.
    for i in range(1, len(z_score)):
      # Stop-loss condition: if the absolute Z-score exceeds the stop-loss threshold while in a trade,
      # close the position (set position to 0).
      if abs(z_score.iloc[i]) > self.stop_loss_threshold and position.iloc[i-1] != 0:
        position.iloc[i] = 0
      # Entry condition (short spread): if Z-score falls below negative entry threshold and no position is open,
      # go long Ticker1 and short Ticker2 (represented by position = 1).
      elif z_score.iloc[i] < -self.entry_threshold and position.iloc[i-1] == 0 and z_score.iloc[i] >= -self.enter_trade_max:
        position.iloc[i] = 1
        num_entry_trades += 1
      # Entry condition (long spread): if Z-score rises above positive entry threshold and no position is open,
      # short Ticker1 and long Ticker2 (represented by position = -1).
      elif z_score.iloc[i] > self.entry_threshold and position.iloc[i-1] == 0 and z_score.iloc[i] <= self.enter_trade_max:
        position.iloc[i] = -1
        num_entry_trades += 1
      # Exit condition: if the absolute Z-score falls below the exit threshold while in a trade,
      # close the position.
      elif abs(z_score.iloc[i]) < self.exit_threshold and position.iloc[i-1] != 0:
        position.iloc[i] = 0
      else:
        # If no entry, exit, or stop-loss condition is met, maintain the current position.
        position.iloc[i] = position.iloc[i-1]

    # Calculate percentage changes in prices for individual tickers.
    returns_ticker1 = data_period[ticker1].pct_change()
    returns_ticker2 = data_period[ticker2].pct_change()

    # Calculate portfolio returns for the pairs trading strategy.
    # The formula accounts for the hedge ratio and assumes that a position taken at time `t-1`
    # generates returns based on price changes from `t-1` to `t`.
    # The denominator (1 + abs(hedge_ratio)) normalizes the return by the total capital at risk.
    portfolio_returns = position.shift(1) * (returns_ticker1 - hedge_ratio * returns_ticker2) / (1 + abs(hedge_ratio))
    # Fill any NaN values (e.g., at the beginning of the series) with 0, as no trade occurred then.
    portfolio_returns = portfolio_returns.fillna(0)

    # Calculate cumulative returns for the pairs strategy and individual tickers (buy and hold).
    # Cumulative returns are calculated by compounding daily returns.
    cumulative_returns = (1 + portfolio_returns).cumprod()
    cumulative_returns_ticker1 = (1 + returns_ticker1).cumprod()
    cumulative_returns_ticker2 = (1 + returns_ticker2).cumprod()

    # Initialize performance metrics to None, in case they cannot be calculated.
    annualized_returns = None
    sharpe_ratio = None
    max_drawdown = None
    annualized_returns_ticker1 = None
    annualized_returns_ticker2 = None

    # Calculate annualized returns, Sharpe Ratio, and Maximum Drawdown if sufficient data exists.
    num_periods = len(data_period)
    if num_periods > 0:
      # Total return for the strategy: (final cumulative value - 1).
      total_return_strategy = cumulative_returns.iloc[-1] - 1
      # Annualize the total return. This converts the total return over the period into an equivalent annual rate.
      annualized_returns = (1 + total_return_strategy)**(self.annualization_factor / num_periods) - 1 if total_return_strategy is not None else None

      # Calculate and annualize buy-and-hold returns for Ticker1.
      total_return_tick1 = cumulative_returns_ticker1.iloc[-1] - 1
      annualized_returns_ticker1 = (1 + total_return_tick1)**(self.annualization_factor / num_periods) - 1 if total_return_tick1 is not None else None

      # Calculate and annualize buy-and-hold returns for Ticker2.
      total_return_tick2 = cumulative_returns_ticker2.iloc[-1] - 1
      annualized_returns_ticker2 = (1 + total_return_tick2)**(self.annualization_factor / num_periods) - 1 if total_return_tick2 is not None else None

      # Calculate Sharpe Ratio and Maximum Drawdown if portfolio returns are not empty and have variability.
      if not portfolio_returns.empty and portfolio_returns.std() != 0:
        # Sharpe Ratio measures risk-adjusted return: (mean daily return / standard deviation of daily returns) * sqrt(annualization factor).
        sharpe_ratio = (portfolio_returns.mean()) / (portfolio_returns.std()) * np.sqrt(self.annualization_factor)
        # Maximum Drawdown: measures the largest peak-to-trough decline in cumulative returns.
        # First, calculate the running maximum (peak) of the cumulative returns.
        peak = cumulative_returns.expanding(min_periods=1).max()
        # Then, calculate the drawdown from each peak.
        drawdown = (cumulative_returns / peak) - 1
        # The maximum drawdown is the minimum (most negative) value of the drawdown series.
        max_drawdown = drawdown.min()

    return (
        annualized_returns,
        annualized_returns_ticker1,
        annualized_returns_ticker2,
        num_entry_trades,
        sharpe_ratio,
        max_drawdown,
        z_score,
        cumulative_returns,
        cumulative_returns_ticker1,
        cumulative_returns_ticker2,
        portfolio_returns
    )



  def _analyze_pair_performance(self, df_AI_data, ticker1, ticker2, coint_pvalue, coint_score,
                                training_start_date, training_end_date,
                                testing_start_date, testing_end_date):
    """
    Tests a pairs trading thesis for two given tickers, calculating returns
    for both a training and a testing period using the helper function `_simulate_trades_and_calculate_returns`.

    This function isolates the training and testing phases to evaluate how well a pair's relationship (hedge ratio)
    identified in the training period holds up and performs in a subsequent, unseen testing period.

    Args:
        df_AI_data (pd.DataFrame): DataFrame with historical data for all tickers.
        ticker1 (str): The symbol of the first stock in the pair.
        ticker2 (str): The symbol of the second stock in the pair.
        coint_pvalue (float): The p-value from the cointegration test for the pair.
        coint_score (float): The test statistic from the cointegration test for the pair.
        training_start_date (str): Start date for the training period.
        training_end_date (str): End date for the training period.
        testing_start_date (str): Start date for the testing period.
        testing_end_date (str): End date for the testing period.

    Returns:
        tuple: A comprehensive set of performance metrics for both training and testing periods,
               including annualized returns, Sharpe ratios, max drawdowns, number of trades,
               and time series of cumulative returns, Z-scores, and portfolio returns for the testing period.
    """

    # --- Training Period Analysis ---
    # Slice df_AI_data for the training period to calibrate the model.
    data_training = df_AI_data.loc[training_start_date:training_end_date, [ticker1, ticker2]]
    data_training = data_training.dropna() # Remove any rows with missing data within the training window.

    # If training data is insufficient, return placeholder values.
    if data_training.empty or len(data_training) < 2:
      return None, None, None, None, None, None, coint_pvalue, coint_score, 0, 0, None, None, None, None, pd.Series(), pd.Series(), pd.Series()

    # Calculate the hedge ratio (beta) using Ordinary Least Squares (OLS) regression
    # on the training data. Ticker1 is the dependent variable (Y), and Ticker2 is the independent variable (X).
    X_train = sm.add_constant(data_training[ticker2]) # Add a constant to the independent variable for the regression.
    model_train = sm.OLS(data_training[ticker1], X_train)
    results_train = model_train.fit()
    hedge_ratio = results_train.params[ticker2] # Extract the hedge ratio (slope coefficient for Ticker2).

    # Simulate trades and calculate performance metrics for the training period using the derived hedge ratio.
    (annualized_returns_train,
     annualized_returns_ticker1_train,
     annualized_returns_ticker2_train,
     num_entry_trades_training,
     sharpe_ratio_training,
     max_drawdown_training,
     z_score_training,
     cumulative_returns_training,
     cumulative_returns_ticker1_training,
     cumulative_returns_ticker2_training,
     portfolio_returns_training) = \
        self._simulate_trades_and_calculate_returns(data_training, hedge_ratio, ticker1, ticker2)

    # --- Testing Period Analysis ---
    # Slice df_AI_data for the testing period. This data is 'unseen' by the model calibration.
    data_testing_raw = df_AI_data.loc[testing_start_date:testing_end_date, [ticker1, ticker2]]
    data_testing_raw = data_testing_raw.dropna() # Remove missing data.

    # Prepare `data_testing` by potentially adding `window - 1` data points from the end of `data_training`.
    # This is a common practice to 'warm up' rolling window calculations (like Z-score) in the testing period,
    # ensuring that the `window` size is fully populated from the start of the testing period.
    if len(data_training) >= self.window - 1:
      warmup_data = data_training[[ticker1, ticker2]].tail(self.window - 1)
      data_testing = pd.concat([warmup_data, data_testing_raw])
    else:
      # If training data is too short, just use the raw testing data.
      data_testing = data_testing_raw

    # Initialize testing period metrics to None or empty Series.
    annualized_returns_test = None
    annualized_returns_ticker1_test = None
    annualized_returns_ticker2_test = None
    num_entry_trades_testing = 0 # Placeholder for new entry trades in testing, if applicable.
    sharpe_ratio_testing = None
    max_drawdown_testing = None
    cumulative_returns_testing = pd.Series([1]) # Start cumulative returns at 1 (100% of initial capital).
    cumulative_returns_ticker1_testing = pd.Series([1])
    cumulative_returns_ticker2_testing = pd.Series([1])
    z_score_testing = pd.Series()
    portfolio_returns_testing = pd.Series()

    # If testing data is sufficient, simulate trades and calculate performance metrics.
    if not (data_testing_raw.empty or len(data_testing_raw) < 2):
      # Call the helper function for the testing period. The `hedge_ratio` from training is used.
      (annualized_returns_test,
       annualized_returns_ticker1_test,
       annualized_returns_ticker2_test,
       num_entry_trades_testing,
       sharpe_ratio_testing,
       max_drawdown_testing,
       z_score_testing_full,
       cumulative_returns_testing_full,
       cumulative_returns_ticker1_testing_full,
       cumulative_returns_ticker2_testing_full,
       portfolio_returns_testing_full) = \
          self._simulate_trades_and_calculate_returns(data_testing, hedge_ratio, ticker1, ticker2)
      # Extract only the actual testing period's data from the full simulation results (which included warmup data).
      cumulative_returns_testing = cumulative_returns_testing_full.loc[data_testing_raw.index]
      cumulative_returns_ticker1_testing = cumulative_returns_ticker1_testing_full.loc[data_testing_raw.index]
      cumulative_returns_ticker2_testing = cumulative_returns_ticker2_testing_full.loc[data_testing_raw.index]
      z_score_testing = z_score_testing_full.loc[data_testing_raw.index]
      portfolio_returns_testing = portfolio_returns_testing_full.loc[data_testing_raw.index]

    # Return all calculated metrics for both training and testing periods.
    return annualized_returns_train, \
           annualized_returns_test, \
           annualized_returns_ticker1_train, annualized_returns_ticker2_train, \
           annualized_returns_ticker1_test, \
           annualized_returns_ticker2_test, \
           coint_pvalue, coint_score, num_entry_trades_training, num_entry_trades_testing, \
           sharpe_ratio_training, max_drawdown_training, sharpe_ratio_testing, max_drawdown_testing, \
           cumulative_returns_testing, z_score_testing, portfolio_returns_testing

  def _select_optimized_pairs(self, stock_pairs_p_min, df_AI, training_start_date, training_end_date, testing_start_date, testing_end_date):
    """
    Analyzes a list of cointegrated pairs and selects the 'optimized' ones based on predefined performance criteria.

    This function evaluates each cointegrated pair's performance during both a training period
    (for model calibration) and a testing period (for out-of-sample validation). It then filters
    these pairs based on metrics like annualized returns, Sharpe ratio, and maximum drawdown
    to identify the most robust and profitable pairs.

    Args:
        stock_pairs_p_min (list): A list of tuples, each containing (Ticker1, Ticker2, pvalue, score, crit_value)
                                  for pairs that passed the initial cointegration test.
        df_AI (pd.DataFrame): The DataFrame containing all historical price data for the tickers.
        training_start_date (str): Start date for the training period.
        training_end_date (str): End date for the training period.
        testing_start_date (str): Start date for the testing period.
        testing_end_date (str): End date for the testing period.

    Returns:
        tuple:
            - df_optimized_pairs (pd.DataFrame): A DataFrame of pairs that meet all optimization criteria,
                                                 sorted by training return.
            - df_pair_results (pd.DataFrame): A DataFrame containing the performance results for ALL analyzed pairs.
    """
    all_pair_results = []
    # Iterate through each cointegrated pair to analyze its performance.
    for pair_data in stock_pairs_p_min:
      ticker1, ticker2, pvalue, score, crit_value = pair_data
      # Call `_analyze_pair_performance` to get detailed metrics for training and testing periods.
      initial_ret, new_ret, ret_tick1_train, ret_tick2_train, ret_tick1_test, ret_tick2_test, \
      coint_pvalue_returned, coint_score_returned, num_entry_trades_training, num_entry_trades_testing, \
      sharpe_ratio_training, max_drawdown_training, sharpe_ratio_testing, max_drawdown_testing, _, _, _ = \
      self._analyze_pair_performance(df_AI, ticker1, ticker2, coint_pvalue=pvalue, coint_score=score,
                                    training_start_date=training_start_date, training_end_date=training_end_date,
                                    testing_start_date=testing_start_date, testing_end_date=testing_end_date)

      # Store the results for each pair.
      all_pair_results.append({
          'Ticker1': ticker1, 'Ticker2': ticker2, 'p-value': coint_pvalue_returned,
          'Annualized Training Return': initial_ret, 'Annualized Testing Return': new_ret,
          'Training Sharpe Ratio': sharpe_ratio_training, 'Testing Sharpe Ratio': sharpe_ratio_testing,
          'Training Max Drawdown': max_drawdown_training, 'Testing Max Drawdown': max_drawdown_testing,
          'Training Entry Trades': num_entry_trades_training, 'Testing Entry Trades': num_entry_trades_testing
      })

    df_pair_results = pd.DataFrame(all_pair_results)
    # Save all individual pair results to a CSV file for detailed review.
    df_pair_results.to_csv(os.path.join(self.results_output_dir, f'all_pair_results_{testing_end_date}.csv'), index=False)

    # If no pairs were analyzed, return empty DataFrames.
    if df_pair_results.empty:
      empty_optimized_df = pd.DataFrame(columns=[
          'Ticker1', 'Ticker2', 'p-value', 'Annualized Training Return',
          'Annualized Testing Return', 'Training Sharpe Ratio', 'Testing Sharpe Ratio',
          'Training Max Drawdown', 'Testing Max Drawdown', 'Training Entry Trades', 'Testing Entry Trades'
      ])
      return empty_optimized_df, df_pair_results

    # Filter pairs based on the specified optimization criteria (min returns, min Sharpe, max drawdown, etc.).
    df_optimized_pairs = df_pair_results[
        (df_pair_results['Annualized Training Return'] > self.min_training_return) & \
        (df_pair_results['Training Sharpe Ratio'] > self.min_training_sharpe) & \
        (df_pair_results['Training Entry Trades'] > self.min_training_trades) & \
        (df_pair_results['Training Max Drawdown'] > self.max_training_drawdown) & \
        (df_pair_results['Annualized Testing Return'] > self.min_testing_return) & \
        (df_pair_results['Testing Sharpe Ratio'] > self.min_testing_sharpe) & \
        (df_pair_results['Training Entry Trades'] > 0) & # Ensure at least one trade was made in training.\
        (df_pair_results['Testing Max Drawdown'] > self.max_testing_drawdown) & \
        (df_pair_results['Testing Sharpe Ratio'] / df_pair_results['Training Sharpe Ratio'] > self.min_sharpe_ratio_stability)
    ]

    # Sort the optimized pairs by their annualized training return in descending order.
    return df_optimized_pairs.sort_values(by='Annualized Training Return', ascending=False), df_pair_results

  def _run_final_analysis(self, df_optimized_pairs, df_AI, training_start_date, training_end_date, testing_start_date, testing_end_date):
    """
    Runs a final, detailed performance analysis for the selected optimized pairs.

    This function takes the `df_optimized_pairs` (pairs that passed the `_select_optimized_pairs` criteria)
    and re-runs `_analyze_pair_performance` to retrieve the full time-series results (cumulative returns,
    Z-scores, and raw portfolio returns) for the testing period. These detailed time-series are crucial
    for plotting and in-depth performance evaluation of the final selected pairs.

    Args:
        df_optimized_pairs (pd.DataFrame): DataFrame of pairs previously identified as optimized.
        df_AI (pd.DataFrame): The full historical price data.
        training_start_date (str): Start date for the training period.
        training_end_date (str): End date for the training period.
        testing_start_date (str): Start date for the testing period.
        testing_end_date (str): End date for the testing period.

    Returns:
        pd.DataFrame: A DataFrame containing all performance metrics and time-series data for each
                      of the optimized pairs, specifically including cumulative returns, Z-scores,
                      and portfolio returns for the testing period.
    """
    optimized_pair_results = []
    # Iterate through each of the previously optimized pairs.
    for line in df_optimized_pairs.itertuples():
      ticker1, ticker2 = line.Ticker1, line.Ticker2

      # Re-run the `_analyze_pair_performance` function for these pairs.
      # This time, we specifically need the time-series outputs (cumulative_returns_testing,
      # z_score_testing, portfolio_returns_testing) for plotting and deeper analysis.
      # The cointegration p-value and score are passed as dummy values (0.01, 1) as they were already determined.
      initial_ret, new_ret, ret_tick1_train, ret_tick2_train, ret_tick1_test, ret_tick2_test, \
      coint_pvalue_returned, coint_score_returned, num_entry_trades_training, num_entry_trades_testing, \
      sharpe_ratio_training, max_drawdown_training, sharpe_ratio_testing, max_drawdown_testing, \
      cumulative_returns_testing, z_score_testing, portfolio_returns_testing = \
      self._analyze_pair_performance(df_AI, ticker1, ticker2, \
                                    coint_pvalue=0.01, coint_score=1,
                                    training_start_date=training_start_date,
                                    training_end_date=training_end_date,
                                    testing_start_date=testing_start_date,
                                    testing_end_date=testing_end_date)

      # Append all results, including the detailed time-series data, for each optimized pair.
      optimized_pair_results.append({
          'Ticker1': ticker1, 'Ticker2': ticker2,
          'Annualized Training Return': initial_ret, 'Annualized Testing Return': new_ret,
          'Training Sharpe Ratio': sharpe_ratio_training, 'Testing Sharpe Ratio': sharpe_ratio_testing,
          'Training Max Drawdown': max_drawdown_training, 'Testing Max Drawdown': max_drawdown_testing,
          'Training Entry Trades': num_entry_trades_training, 'Testing Entry Trades': num_entry_trades_testing,
          'Cumulative Returns Testing': cumulative_returns_testing, 'Z-Score Testing': z_score_testing,
          'Portfolio Returns Testing': portfolio_returns_testing
      })
    # Return a DataFrame containing the full analysis for all optimized pairs.
    return pd.DataFrame(optimized_pair_results)

  def update_trades_file(self, new_optimized_pairs_df):
    """
    Updates the trades.csv file by merging new optimized pairs
    with existing trades, preserving the state of open trades.
    """
    try:
      current_trades = pd.read_csv(self.trades_file)
      # Ensure 'pair_key' exists in current_trades for consistent merging logic
      if 'pair_key' not in current_trades.columns:
        current_trades['pair_key'] = current_trades['Ticker1'] + '/' + current_trades['Ticker2']
    except (FileNotFoundError, pd.errors.EmptyDataError):
      # Initialize with relevant columns.
      current_trades = pd.DataFrame(columns=[
          'pair_key', 'Ticker1', 'Ticker2', 'status', 'profit', 'intrade',
          'Ticker1 Buy Price', 'Ticker2 Buy Price', 'Two Months Profit',
          'p-value', 'Annualized Training Return', 'Annualized Testing Return',
          'Training Sharpe Ratio', 'Testing Sharpe Ratio', 'Training Max Drawdown', 'Testing Max Drawdown',
          'Training Entry Trades', 'Testing Entry Trades'])

    # Create pair_key for new_optimized_pairs_df
    new_optimized_pairs_df['pair_key'] = new_optimized_pairs_df['Ticker1'] + '/' + new_optimized_pairs_df['Ticker2']

    updated_trades_list = []
    current_pair_keys_set = set(current_trades['pair_key']) if not current_trades.empty else set()
    new_optimized_pair_keys_set = set(new_optimized_pairs_df['pair_key'])


    # First, process all existing trades from current_trades
    for idx, row in current_trades.iterrows():
      row_dict = row.to_dict() # Create a mutable copy of the row

      # Case 1: The existing pair is also in the newly optimized list
      if row_dict['pair_key'] in new_optimized_pair_keys_set:
        row_dict['Two Months Profit'] = 0
        # If it was previously 'inactive' reactivate it.
        if row_dict['status'] == 'inactive':
          row_dict['status'] = 'active'

      # Case 2: The existing pair is NO LONGER in the newly optimized list
      else:
        # If two months profit is more than min_two_months_profit_for_active keep it active
        if row_dict['Two Months Profit'] <= self.min_two_months_profit_for_active:
          row_dict['status'] = 'inactive'
        row_dict['Two Months Profit'] = 0

      updated_trades_list.append(row_dict)

    # Second, add any truly NEW optimized pairs (that were not in current_trades before)
    for idx, new_pair_row in new_optimized_pairs_df.iterrows():
      new_pair_key = new_pair_row['pair_key']
      if new_pair_key not in current_pair_keys_set:
        # This is a genuinely new optimized pair, add it with default active state and metrics.
        new_trade_entry = {
            'pair_key': new_pair_key,
            'Ticker1': new_pair_row['Ticker1'],
            'Ticker2': new_pair_row['Ticker2'],
            'status': 'active',
            'profit': 0.0,
            'intrade': 'no',
            'Ticker1 Buy Price': 0.0,
            'Ticker2 Buy Price': 0.0,
            'Two Months Profit': 0.0
        }
        # Add performance metrics from the new_pair_row
        for col in ['p-value', 'Annualized Training Return', 'Annualized Testing Return',
                    'Training Sharpe Ratio', 'Testing Sharpe Ratio', 'Training Max Drawdown',
                    'Testing Max Drawdown', 'Training Entry Trades', 'Testing Entry Trades']:
          if col in new_pair_row:
            new_trade_entry[col] = new_pair_row[col]
          else:
            new_trade_entry[col] = np.nan

        updated_trades_list.append(new_trade_entry)

    df_new_trades_history = pd.DataFrame(updated_trades_list)
    df_new_trades_history.to_csv(self.trades_file, index=False)


def training_and_optimization(ai_tickers, initial_date_ref_str='2026-01-01', trades_history_file_name = 'trades_history1.csv',
                 entry_threshold=1.5, exit_threshold=0.2, stop_loss_threshold=4.5,
                 window=90, p_min_coint=0.05, fee=0.005, enter_trade_max = 3.0,
                 min_training_return=1, min_training_sharpe=1, min_training_trades=0,
                 max_training_drawdown=-0.4, min_testing_return=0.8, min_testing_sharpe=1,
                 max_testing_drawdown=-0.4, min_sharpe_ratio_stability=0.3, min_two_months_profit_for_active=0.03, results_output_dir='.'):
    """
    Orchestrates the training and optimization process for pairs trading.
    This function identifies cointegrated pairs, evaluates their performance
    over training and testing periods, and selects optimized pairs based on
    predefined criteria.

    Args:
        ai_tickers (list): List of stock tickers to consider for pairs trading.
        initial_date_ref_str (str): Reference date for setting up training and testing periods.
        trades_history_file_name (str): File to store the history of all trades.
        entry_threshold (float): Z-score threshold for entering a trade.
        exit_threshold (float): Z-score threshold for exiting a trade.
        stop_loss_threshold (float): Z-score threshold for triggering a stop-loss.
        window (int): Rolling window size for Z-score calculation.
        p_min_coint (float): P-value threshold for cointegration test.
        fee (float): Transaction fee per trade.
        enter_trade_max (float): Maximum absolute Z-score to allow entering a new trade.
        min_training_return (float): Minimum annualized return for training period.
        min_training_sharpe (float): Minimum Sharpe ratio for training period.
        min_training_trades (int): Minimum number of trades in training period.
        max_training_drawdown (float): Maximum drawdown for training period.
        min_testing_return (float): Minimum annualized return for testing period.
        min_testing_sharpe (float): Minimum Sharpe ratio for testing period.
        max_testing_drawdown (float): Maximum drawdown for testing period.
        min_sharpe_ratio_stability (float): Minimum ratio of testing to training Sharpe ratio.
    """
    PairsTraining = PairsTradingManager(ai_tickers, trades_history_file_name,
                 entry_threshold=entry_threshold, exit_threshold=exit_threshold, stop_loss_threshold=stop_loss_threshold,
                 window=window, p_min_coint=p_min_coint, fee=fee, enter_trade_max=enter_trade_max,
                 min_training_return=min_training_return, min_training_sharpe=min_training_sharpe, min_training_trades=min_training_trades,
                 max_training_drawdown=max_training_drawdown, min_testing_return=min_testing_return, min_testing_sharpe=min_testing_sharpe,
                 max_testing_drawdown=max_testing_drawdown, min_sharpe_ratio_stability=min_sharpe_ratio_stability, min_two_months_profit_for_active=min_two_months_profit_for_active,
                 results_output_dir=results_output_dir)
    # --- Define Dates training and Optimization ---
    date_ref = pd.to_datetime(initial_date_ref_str)
    initial_testing_end_date = date_ref - timedelta(days=1)
    initial_testing_start_date = initial_testing_end_date - timedelta(days=30)
    initial_training_end_date = initial_testing_start_date - timedelta(days=1)
    initial_training_start_date = initial_training_end_date - timedelta(days=60)

    df_ai_full_history_end_date = initial_testing_end_date
    df_ai_full_history_start_date = initial_training_start_date - pd.DateOffset(years=1)

    # Download full historical data for the initial testing phase
    df_AI_full = yf.download(PairsTraining.ai_tickers,
                              start=df_ai_full_history_start_date.strftime('%Y-%m-%d'),
                              end=df_ai_full_history_end_date.strftime('%Y-%m-%d'),
                              interval='1h', auto_adjust=True)['Close']
    df_AI_full = df_AI_full.dropna()
    # 1. Perform initial cointegration test for 2025 period
    stock_pairs_p_min, initial_full_coint_results = PairsTraining._perform_coint_test(
        df_AI_full,
        initial_training_start_date.strftime('%Y-%m-%d'),
        initial_training_end_date.strftime('%Y-%m-%d')
    )
    # Save the full cointegration results to a CSV
    df_initial_full_coint = pd.DataFrame(initial_full_coint_results)
    df_initial_full_coint.to_csv(os.path.join(results_output_dir, 'full_coint_results_initial_testing.csv'), index=False)

    # 2. Select initial optimized pairs based on performance
    df_optimized_pairs, _ = PairsTraining._select_optimized_pairs(
        stock_pairs_p_min, df_AI_full,
        training_start_date=initial_training_start_date.strftime('%Y-%m-%d'),
        training_end_date=initial_training_end_date.strftime('%Y-%m-%d'),
        testing_start_date=initial_testing_start_date.strftime('%Y-%m-%d'),
        testing_end_date=initial_testing_end_date.strftime('%Y-%m-%d')
    )
    # 3. Initialize trades.csv with the initially optimized pairs
    PairsTraining.update_trades_file(df_optimized_pairs)
