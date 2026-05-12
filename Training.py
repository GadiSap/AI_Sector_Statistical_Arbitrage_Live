# Imports
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import os
import itertools # To handle permutations of parameters
from multiprocessing import Pool, cpu_count # Import for parallel processing
from tqdm import tqdm # Import tqdm for progress bar
from PairsTradingManager import PairsTradingManager

"""
This script is used for training the data
"""
# --- Function to run a single parameter combination --- #
def evaluate_single_combination(params, output_columns, ai_tickers, p_min_coint_param, fee_param, df_AI_full, coint_test_start, coint_test_end, training_period_start, testing_period_start, real_sim_start, real_sim_end, stock_pairs_p_min_once):
    (entry_th, exit_th, stop_loss_th, enter_trade_max_th, window_val, days_hedge_calc_val) = params

    # Instantiate PairsTradingManager with current parameters
    manager = PairsTradingManager(
        ai_tickers=ai_tickers,
        entry_threshold=entry_th,
        exit_threshold=exit_th,
        stop_loss_threshold=stop_loss_th,
        enter_trade_max=enter_trade_max_th,
        window=window_val,
        p_min_coint=p_min_coint_param,
        fee=fee_param
    )

    # Run the evaluation for the current parameter set, passing the pre-computed cointegrated pairs
    df_pair_results_for_this_combo = manager.evaluate_strategy_parameters(
        df_AI_data=df_AI_full, # df_AI_full is now filtered global variable
        coint_start_date=coint_test_start, # global variable
        coint_end_date=coint_test_end,     # global variable
        training_start_date=training_period_start, # global variable
        testing_start_date=testing_period_start,   # global variable
        real_scenario_start_date=real_sim_start,     # global variable
        real_scenario_end_date=real_sim_end,       # global variable
        days_for_hedge_calc=days_hedge_calc_val,
        pre_computed_stock_pairs_p_min=stock_pairs_p_min_once # Pass the pre-computed pairs
    )

    if not df_pair_results_for_this_combo.empty:
        # Add the current parameter values as new columns to each row of the results DataFrame
        df_pair_results_for_this_combo['Entry Threshold'] = entry_th
        df_pair_results_for_this_combo['Exit Threshold'] = exit_th
        df_pair_results_for_this_combo['Stop Loss Threshold'] = stop_loss_th
        df_pair_results_for_this_combo['Enter Trade Max'] = enter_trade_max_th
        df_pair_results_for_this_combo['Window'] = window_val
        df_pair_results_for_this_combo['Days for Hedge Calc'] = days_hedge_calc_val

        # Reorder columns to match the defined output_columns
        df_pair_results_for_this_combo = df_pair_results_for_this_combo[output_columns]
        return df_pair_results_for_this_combo
    else:
        return pd.DataFrame(columns=output_columns)
if __name__ == '__main__':
    # Define parameter ranges to test
    entry_thresholds_to_test = [1, 1.5, 2.0]
    exit_thresholds_to_test = [0.2, 0.5, 0.8]
    stop_loss_thresholds_to_test = [3, 3.5, 4, 5]
    enter_trade_max_to_test = [2.5, 3, 3.5]
    windows_to_test = [90, 140, 200]  # In hours
    days_for_hedge_calcs_to_test = [30, 60, 90]  # In days

    fee_param = 0.005
    p_min_coint_param = 0.05

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
    # Define file names for trading
    directory = 'files/training'
    trades_history_file_name = 'training_trades_history_hedge90.csv' # File to store the history of all trades made by the strategy.

    if not os.path.exists(directory):
        # if directory does not exist it will create it
        os.makedirs(directory)

    # Define fixed dates for evaluation for evaluate_strategy_parameters
    # These dates are for historical evaluation of the strategy parameters, not the live simulation dates.
    coint_test_start = '2025-01-01'
    coint_test_end = '2025-08-31' # End of cointegration period
    training_period_start = '2025-06-01'
    # training_end_date is internally derived as 1 hour before testing_period_start by evaluate_strategy_parameters
    testing_period_start = '2025-09-01'
    # testing_end_date is internally derived as 1 hour before real_sim_start by evaluate_strategy_parameters
    real_sim_start = '2025-11-01'
    real_sim_end = '2025-12-31'


    download_start_date = pd.to_datetime(coint_test_start).strftime('%Y-%m-%d')
    download_end_date = (pd.to_datetime(real_sim_end) + timedelta(days=1)).strftime('%Y-%m-%d') # +1 day to ensure end date is inclusive

    # Download full historical data once for all evaluations
    print(f"Downloading historical data from {download_start_date} to {download_end_date} for all tickers...")
    df_AI_full = yf.download(ai_tickers, start=download_start_date, end=download_end_date, interval='1h', auto_adjust=True)['Close']
    df_AI_full = df_AI_full.dropna(axis=1, how='all') # Drop columns that are all NaN
    df_AI_full = df_AI_full.dropna() # Drop rows with any NaN after cleaning columns
    print("Historical data download complete.")

    # Define output file path and ensure directory exists
    output_csv_file = os.path.join(directory, trades_history_file_name)
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)

    # --- Perform Cointegration Test Once ---
    # Instantiate PairsTradingManager to perform the initial cointegration test.
    # Use default parameters for the manager, as they won't affect the coint test itself.
    initial_manager = PairsTradingManager(ai_tickers=ai_tickers, p_min_coint=p_min_coint_param)
    print("\nPerforming initial cointegration test...")
    stock_pairs_p_min_once, _ = initial_manager._perform_coint_test(df_AI_full, coint_test_start, coint_test_end)
    print(f"Found {len(stock_pairs_p_min_once)} cointegrated pairs.")
    # -------------------------------------

    # --- Memory Optimization: Filter DF_AI_FULL to only include tickers in cointegrated pairs ---
    # Extract all unique tickers from the cointegrated pairs
    relevant_tickers = sorted(list(set([ticker for pair in stock_pairs_p_min_once for ticker in (pair[0], pair[1])])))
    df_AI_full = df_AI_full[relevant_tickers].copy() # Filter and make a copy to ensure it's not a view
    print(f"Filtered df_AI_full to only include {len(relevant_tickers)} relevant tickers from cointegrated pairs.")
    # ---------------------------------------------------------------------------------------------


    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        entry_thresholds_to_test,
        exit_thresholds_to_test,
        stop_loss_thresholds_to_test,
        enter_trade_max_to_test,
        windows_to_test,
        days_for_hedge_calcs_to_test
    ))

    # Define the full list of columns for the output CSV
    output_columns = [
        'Entry Threshold', 'Exit Threshold', 'Stop Loss Threshold', 'Enter Trade Max',
        'Window', 'Days for Hedge Calc',
        'Ticker1', 'Ticker2', 'Cointegration P-value', 'Cointegration Score', 'Crit Value',
        'Training Annualized Return', 'Testing Annualized Return', 'Simulation Annualized Return',
        'Training Sharpe Ratio', 'Testing Sharpe Ratio', 'Simulation Sharpe Ratio',
        'Training Max Drawdown', 'Testing Max Drawdown', 'Simulation Max Drawdown',
        'Training Entry Trades', 'Testing Entry Trades', 'Simulation Entry Trades',
        'Simulation Hedge Ratio'
    ]

    # Initialize CSV file with headers. Clear existing file if any.
    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)
    pd.DataFrame(columns=output_columns).to_csv(output_csv_file, index=False, header=True)



    # --- Run in parallel --- #
    print(f"\nStarting parallel evaluation of {len(param_combinations)} parameter combinations...")
    # Determine the number of CPU cores to use. Leave one core free for system processes.
    num_processes = max(1, cpu_count() - 1)

    with Pool(processes=num_processes) as pool:
        # Use starmap to pass each tuple of parameters from param_combinations to the function
        # Wrap with tqdm to show a progress bar
        all_results_dfs = list(tqdm(pool.starmap(evaluate_single_combination, [ (combo,  output_columns, ai_tickers, p_min_coint_param, fee_param, df_AI_full, coint_test_start, coint_test_end, training_period_start, testing_period_start, real_sim_start, real_sim_end, stock_pairs_p_min_once) for combo in param_combinations ]), total=len(param_combinations)))

    # Concatenate all results and save to CSV
    final_results_df = pd.concat(all_results_dfs, ignore_index=True)
    final_results_df.to_csv(output_csv_file, index=False, header=True) # Overwrite with the full, collected results

    print(f"\nParameter optimization complete. All results appended to {output_csv_file}")
    #print("Final summary of results:")
    df_summary_results = pd.read_csv(output_csv_file)
