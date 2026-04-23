import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 

def get_multivar_norm(mean: list, cov: pd.DataFrame, days: int) -> np.ndarray[np.float64]:
    mean = np.array(mean)
    cov = cov.to_numpy()
    return np.random.multivariate_normal(mean, cov, days)

def get_portfolio_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        ticks = ' '.join(tickers)
        data = yf.download(ticks, start=start_date,
                        end=end_date, group_by='tickers')
        return data
    except Exception as e:
        print(f'Not valid ticker information: {e}')
        return None

def get_portfolio_returns(stock_returns: pd.DataFrame, tickers: list, weights: list) -> list[float]:
    
    t_weight = {tickers[i]:weights[i] for i in range(len(tickers))}


    if isinstance(stock_returns.columns, pd.MultiIndex):
        close_returns = stock_returns.xs('Close', axis=1, level=1)
    else:
        close_returns = stock_returns.copy()

    close_returns = close_returns.reindex(columns=tickers)

    portfolio_returns = []
    for _, row in close_returns.iterrows():
        new_weights = _check_and_rebalance(row, t_weight) # Check if there are missing values
        portfolio_returns.append(sum([float(row[t]*w) for t,w in new_weights.items()]))
    return portfolio_returns

def _check_and_rebalance(data: pd.DataFrame, tick_weights: dict,) -> dict:
    ## If some of the tickers are missing on this date (i.e. not listed yet)
    ## remove it's weight and contribute it equally to others 
    t_weights = tick_weights.copy()
    removed_weight = 0

    for ticker, weight in tick_weights.items():
        if pd.isna(data.get(ticker, np.nan)):
            del t_weights[ticker]
            removed_weight += weight
    
    return {t : w/(1 - removed_weight) for t, w in t_weights.items()}

def get_stock_returns(data: pd.DataFrame, days: int)-> pd.DataFrame:
    ## Calculate the returns over 'days' period
    ## drop row where all the values are 
    stock_returns = data.pct_change(days)
    stock_returns = stock_returns.dropna(how='all') ## drop first days
    return stock_returns

def get_cov_matrix(data: pd.DataFrame) -> pd.DataFrame:
    # returns = get_stock_returns(data, 21)
    close_data = data.loc[:, pd.IndexSlice[:, 'Close']]
    return close_data.cov()

if __name__ == '__main__':

    tickers = ['GOOG', 'META', 'NVDA', 'COUR']
    weights = [0.45, 0.32, 0.1, 0.13]
    tot_val = 5 * 10**5

    start_date = '2015-03-19'
    end_date = '2025-03-19'
    
    df = get_portfolio_data(tickers, start_date, end_date)

    returns = get_stock_returns(df, 21)
    close_data = df.loc[:, pd.IndexSlice[:, 'Close']]
    stock_returns = get_stock_returns(close_data, 21) #one month returns

    PnL = get_portfolio_returns(stock_returns, tickers, weights)
    historic_var = np.percentile(PnL, 1)
    print(f'The simple historical VaR is {abs(historic_var*tot_val)} USD')

    ES = []
    for r in PnL:
        if r<=historic_var:
            ES.append(r)
    print(f'The historical expected shortfall is {abs(np.mean(ES)*tot_val)} USD')

    daily_returns = get_stock_returns(df, 1)
    daily_returns = daily_returns.loc[:, pd.IndexSlice[:, 'Close']] 
    daily_returns_mean = daily_returns.mean(axis=0)
    print(daily_returns_mean)
    daily_cov = get_cov_matrix(daily_returns)
    print(daily_returns)
    print(daily_cov)
    
    dist = []
    for i in range(10000):
        sim = pd.DataFrame(get_multivar_norm(daily_returns_mean, daily_cov, 21), columns = tickers)
        port_returns = np.array(get_portfolio_returns(sim, tickers, weights), dtype=float)
        # Compound 21 daily portfolio returns: (1+r1)*(1+r2)*...*(1+r21) - 1
        compounded_return = np.prod(1 + port_returns) - 1
        dist.append(compounded_return)

    dist = np.array(dist, dtype=float)

    plt.figure(figsize=(10, 6))
    plt.hist(dist, bins=30, edgecolor='black', alpha=0.75)
    plt.title('Distribution of Simulated 21-Day Portfolio Returns')
    plt.xlabel('Portfolio return')
    plt.ylabel('Frequency')

    # Show risk marker (1% quantile)
    var_1 = np.percentile(dist, 1)
    plt.axvline(var_1, color='red', linestyle='--', linewidth=2, label=f'1% VaR: {var_1:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Monte Carlo paths visualization
    num_paths = 100
    num_days = 21
    initial_value = 1.0  # normalized to 1

    paths = np.zeros((num_paths, num_days))
    paths[:, 0] = initial_value

    for path_idx in range(num_paths):
        sim = pd.DataFrame(
            get_multivar_norm(daily_returns_mean, daily_cov, num_days - 1),  # ← Generate 20 returns
            columns=tickers
        )
        port_daily_returns = np.array(get_portfolio_returns(sim, tickers, weights), dtype=float)
        
        # Compound daily returns to get cumulative portfolio value
        cumulative = np.cumprod(1 + port_daily_returns)  # Now 20 values
        paths[path_idx, 1:] = cumulative  # Fits into slots 1-20

    # Plot paths
    plt.figure(figsize=(12, 6))
    plt.plot(paths.T, alpha=0.3, color='blue', linewidth=1)

    # Add mean path
    mean_path = paths.mean(axis=0)
    plt.plot(mean_path, color='red', linewidth=2.5, label='Mean path')

    # Add percentiles
    percentile_5 = np.percentile(paths, 5, axis=0)
    percentile_95 = np.percentile(paths, 95, axis=0)
    plt.plot(percentile_5, color='orange', linestyle='--', linewidth=2, label='5th percentile')
    plt.plot(percentile_95, color='green', linestyle='--', linewidth=2, label='95th percentile')

    plt.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value (normalized)')
    plt.title(f'Monte Carlo Simulation: {num_paths} Portfolio Paths over {num_days} Days')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

