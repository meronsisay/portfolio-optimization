"""
Functions for data fetching, cleaning, analysis, and visualization
Preprocess and Explore the Data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")


# ============================================
# 1. DATA FETCHING
# ============================================


def fetch_stock_data(tickers, start, end):
    """
    Fetch stock data from yfinance

    Parameters:
    -----------
    tickers : list
        List of ticker symbols (e.g., ['TSLA', 'BND', 'SPY'])
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str
        End date in 'YYYY-MM-DD' format

    Returns:
    --------
    dict : Dictionary of DataFrames for each ticker

    Example:
    >>> data = fetch_stock_data(['TSLA', 'BND', 'SPY'], '2015-01-01', '2026-06-30')
    """
    data = {}
    print(f"Fetching data for {tickers}...")
    print("-" * 40)

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # FIX: Force auto_adjust=False to cleanly preserve standard column outputs
            df = stock.history(start=start, end=end, auto_adjust=False)
            data[ticker] = df
            print(f"  {ticker}: {len(df):,} rows")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")
            data[ticker] = pd.DataFrame()

    print("-" * 40)
    print(f"Data fetching complete!")
    return data


def combine_prices(data_dict, column="Adj Close"):
    """
    Combine closing prices from multiple tickers into one DataFrame

    Parameters:
    -----------
    data_dict : dict
        Dictionary of DataFrames from fetch_stock_data()
    column : str
        Column name to extract (e.g., 'Close', 'Adj Close')

    Returns:
    --------
    pd.DataFrame : Combined price data
    """
    combined = pd.DataFrame()
    for ticker, df in data_dict.items():
        if column in df.columns:
            combined[ticker] = df[column]
        else:
            print(f"Column '{column}' not found in {ticker}")

    return combined


def verify_data_types(data_dict):
    """
    Verify and report data types for all DataFrames

    Parameters:
    -----------
    data_dict : dict
        Dictionary of DataFrames

    Returns:
    --------
    pd.DataFrame : Summary of data types
    """
    print("\n" + "=" * 60)
    print("DATA TYPE VERIFICATION")
    print("=" * 60)

    type_summary = []

    for ticker, df in data_dict.items():
        print(f"\n {ticker}:")
        print("-" * 30)

        # Check index type
        print(f"  Index type: {type(df.index)}")
        print(f"  Index is DatetimeIndex: {isinstance(df.index, pd.DatetimeIndex)}")

        # Check each column
        for col in df.columns:
            dtype = df[col].dtype
            print(f"  {col}: {dtype}")

            # Check for potential issues
            if dtype == "object":
                print(f"    {col} is object type - may need conversion")
            elif dtype in ["int64", "float64"]:
                print(f"    {col} is numeric")

        # Store summary
        type_summary.append(
            {
                "Ticker": ticker,
                "Index_Type": str(type(df.index)),
                "Columns": ", ".join(df.columns.tolist()),
                "Data_Types": ", ".join(
                    [f"{col}: {df[col].dtype}" for col in df.columns]
                ),
            }
        )

    return pd.DataFrame(type_summary)


def get_data_summary(data_dict):
    """
    Get summary information about fetched data

    Parameters:
    -----------
    data_dict : dict
        Dictionary of DataFrames

    Returns:
    --------
    pd.DataFrame : Summary table
    """
    summary = []
    for ticker, df in data_dict.items():
        if not df.empty:
            summary.append(
                {
                    "Ticker": ticker,
                    "Start": df.index[0].date(),
                    "End": df.index[-1].date(),
                    "Rows": len(df),
                    "Columns": len(df.columns),
                }
            )
    return pd.DataFrame(summary)


# ============================================
# 2. DATA CLEANING
# ============================================


def clean_data(data_dict, method="ffill"):
    """
    Clean data by handling missing values

    Parameters:
    -----------
    data_dict : dict
        Dictionary of DataFrames
    method : str
        Cleaning method: 'ffill' (forward fill), 'bfill' (backward fill),
        'interpolate', or 'drop'

    Returns:
    --------
    dict : Cleaned DataFrames
    """
    cleaned = {}
    print(f"\n Cleaning data using method: {method}")
    print("-" * 40)

    for ticker, df in data_dict.items():
        df_clean = df.copy()

        if method == "ffill":
            df_clean = df_clean.ffill().bfill()
        elif method == "bfill":
            df_clean = df_clean.bfill().ffill()
        elif method == "interpolate":
            df_clean = df_clean.interpolate(method="linear")
        elif method == "drop":
            df_clean = df_clean.dropna()
        else:
            print(f"  Unknown method '{method}', using forward fill")
            # FIX: Removed deprecated positional fillna syntax to avoid future pandas warnings
            df_clean = df_clean.ffill().bfill()

        cleaned[ticker] = df_clean

        missing = df_clean.isnull().sum().sum()
        if missing == 0:
            print(f"    {ticker}: Cleaned (no missing values)")
        else:
            print(f"  {ticker}: {missing} missing values remain")

    return cleaned


def check_missing(data_dict):
    """
    Check and report missing values in data

    Parameters:
    -----------
    data_dict : dict
        Dictionary of DataFrames
    """
    print("\nChecking for missing values...")
    print("-" * 40)

    for ticker, df in data_dict.items():
        missing = df.isnull().sum()
        total_missing = missing.sum()

        if total_missing == 0:
            print(f"  {ticker}: No missing values")
        else:
            print(f"    {ticker}: {total_missing} missing values")
            print(f"     Columns: {missing[missing > 0].index.tolist()}")


def calculate_returns(price_df):
    """
    Calculate daily returns from price data

    Parameters:
    -----------
    price_df : pd.DataFrame
        Price data (single asset or multiple assets)

    Returns:
    --------
    pd.DataFrame : Returns data
    """
    returns = price_df.pct_change()
    return returns


# ============================================
# 3. VISUALIZATION FUNCTIONS
# ============================================


def plot_prices(price_df, title="Closing Prices", figsize=(14, 6)):
    """
    Plot closing prices for all assets

    Parameters:
    -----------
    price_df : pd.DataFrame
        Combined price data from combine_prices()
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    for col in price_df.columns:
        plt.plot(price_df.index, price_df[col], label=col, linewidth=1.5, alpha=0.8)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price ($)", fontsize=11)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_returns(returns_df, title="Daily Returns", figsize=(14, 3)):
    """
    Plot daily returns for all assets in subplots

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Returns data from calculate_returns()
    title : str
        Plot title
    figsize : tuple
        Base figure size (width, height per subplot)
    """
    n = len(returns_df.columns)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True)

    if n == 1:
        axes = [axes]

    for i, col in enumerate(returns_df.columns):
        ax = axes[i]
        ax.plot(
            returns_df.index,
            returns_df[col],
            linewidth=0.7,
            alpha=0.7,
            color="steelblue",
        )
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Return", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add statistics box
        mean = returns_df[col].mean() * 100
        std = returns_df[col].std() * 100
        ax.text(
            0.02,
            0.95,
            f"Mean: {mean:.3f}%\nStd: {std:.3f}%",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

    plt.xlabel("Date", fontsize=11)
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_rolling_volatility(price_df, window=30, figsize=(14, 4)):
    """
    Plot price with rolling mean and standard deviation bands

    Parameters:
    -----------
    price_df : pd.DataFrame
        Combined price data
    window : int
        Rolling window size in days
    figsize : tuple
        Base figure size (width, height per subplot)
    """
    n = len(price_df.columns)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True)

    if n == 1:
        axes = [axes]

    for i, col in enumerate(price_df.columns):
        ax = axes[i]
        prices = price_df[col]

        # Calculate rolling statistics
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()

        # Plot
        ax.plot(
            prices.index, prices, linewidth=1, alpha=0.5, label="Price", color="blue"
        )
        ax.plot(
            rolling_mean.index,
            rolling_mean,
            linewidth=2,
            label=f"{window}-day MA",
            color="orange",
        )
        ax.fill_between(
            rolling_mean.index,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.2,
            color="green",
            label="±1 Std",
        )

        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Price ($)", fontsize=9)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.xlabel("Date", fontsize=11)
    plt.suptitle(
        f"Price with {window}-Day Rolling Statistics",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(price_df, figsize=(8, 6)):
    """
    Plot correlation heatmap for all assets

    Parameters:
    -----------
    price_df : pd.DataFrame
        Combined price data
    figsize : tuple
        Figure size (width, height)
    """
    corr = price_df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Asset Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return corr


def plot_returns_distribution(returns_df, figsize=(14, 4)):
    """
    Plot distribution of returns with histogram and KDE

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Returns data
    figsize : tuple
        Base figure size (width, height per subplot)
    """
    n = len(returns_df.columns)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, col in enumerate(returns_df.columns):
        ax = axes[idx]
        returns = returns_df[col].dropna()

        # Histogram with KDE
        ax.hist(
            returns,
            bins=50,
            density=True,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        sns.kdeplot(returns, ax=ax, color="red", linewidth=2)

        # Add normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        mu, std = returns.mean(), returns.std()
        normal_dist = stats.norm.pdf(x, mu, std)
        ax.plot(x, normal_dist, "g--", linewidth=2, label="Normal")

        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Daily Return", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(returns_df.columns), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Returns Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


# ============================================
# 4. STATISTICAL TESTS
# ============================================


def test_stationarity(series, name, verbose=True):
    """
    Augmented Dickey-Fuller test for stationarity

    Parameters:
    -----------
    series : pd.Series
        Time series data
    name : str
        Name of the series for reporting
    verbose : bool
        Whether to print results

    Returns:
    --------
    dict : Test results
    """
    series_clean = series.dropna()

    if len(series_clean) < 10:
        print(f"Not enough data for {name}")
        return {"is_stationary": False}

    result = adfuller(series_clean, autolag="AIC")

    is_stationary = result[1] <= 0.05

    if verbose:
        print(f"\nADF Test - {name}")
        print("-" * 40)
        print(f"  ADF Statistic: {result[0]:.6f}")
        print(f"  p-value: {result[1]:.6f}")
        print(f"  {'STATIONARY' if is_stationary else ' NON-STATIONARY'}")
        print(f"  Critical values:")
        for key, value in result[4].items():
            print(f"    {key}: {value:.6f}")

    return {
        "name": name,
        "statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4],
        "is_stationary": is_stationary,
    }


def analyze_all_stationarity(price_df, returns_df):
    """
    Test stationarity for all price and returns data

    Parameters:
    -----------
    price_df : pd.DataFrame
        Combined price data
    returns_df : pd.DataFrame
        Returns data

    Returns:
    --------
    dict : All test results
    """
    results = {}

    print("\n" + "=" * 60)
    print("STATIONARITY ANALYSIS")
    print("=" * 60)

    # Test prices (expected: non-stationary)
    print("\n Testing CLOSING PRICES (should be non-stationary):")
    for ticker in price_df.columns:
        results[f"{ticker}_price"] = test_stationarity(
            price_df[ticker], f"{ticker} (Price)"
        )

    # Test returns (expected: stationary)
    print("\n Testing RETURNS (should be stationary):")
    for ticker in returns_df.columns:
        results[f"{ticker}_returns"] = test_stationarity(
            returns_df[ticker], f"{ticker} (Returns)"
        )

    return results


# ============================================
# 5. RISK METRICS
# ============================================


def calculate_risk_metrics(returns, confidence=0.95):
    """
    Calculate comprehensive risk metrics for a returns series

    Parameters:
    -----------
    returns : pd.Series
        Returns data
    confidence : float
        Confidence level for VaR (default: 0.95)

    Returns:
    --------
    dict : Risk metrics
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return {"Error": "No valid returns data"}

    # Value at Risk (Historical)
    var = returns.quantile(1 - confidence)

    # Expected Shortfall (CVaR)
    cvar = returns[returns <= var].mean()

    # Sharpe Ratio (assuming 0% risk-free rate)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Skewness and Kurtosis
    skew = returns.skew()
    kurt = returns.kurtosis()

    return {
        "Daily Mean (%)": returns.mean() * 100,
        "Daily Std (%)": returns.std() * 100,
        "Annual Return (%)": returns.mean() * 252 * 100,
        "Annual Volatility (%)": returns.std() * np.sqrt(252) * 100,
        "confidence_level": confidence,  # FIX: Save the confidence setting directly to bypass naming mismatch crashes
        f"VaR ({int(confidence*100)}%)": var * 100,
        "CVaR (%)": cvar * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd * 100,
        "Skewness": skew,
        "Kurtosis": kurt,
    }


def calculate_all_risk_metrics(returns_df, confidence=0.95):
    """
    Calculate risk metrics for all assets

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Returns data for all assets
    confidence : float
        Confidence level for VaR

    Returns:
    --------
    dict : Risk metrics for each asset
    """
    all_metrics = {}
    for ticker in returns_df.columns:
        all_metrics[ticker] = calculate_risk_metrics(returns_df[ticker], confidence)
    return all_metrics


def detect_outliers(returns, threshold=3):
    """
    Detect outlier days using Z-score method

    Parameters:
    -----------
    returns : pd.Series
        Returns data
    threshold : float
        Z-score threshold for outlier detection

    Returns:
    --------
    pd.Series : Outlier returns
    """
    returns_clean = returns.dropna()

    if len(returns_clean) == 0:
        return pd.Series()

    z_scores = np.abs(stats.zscore(returns_clean))
    outliers = returns_clean[z_scores > threshold]
    return outliers


def detect_all_outliers(returns_df, threshold=3):
    """
    Detect outliers for all assets

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Returns data for all assets
    threshold : float
        Z-score threshold

    Returns:
    --------
    dict : Outliers for each asset
    """
    all_outliers = {}
    for ticker in returns_df.columns:
        all_outliers[ticker] = detect_outliers(returns_df[ticker], threshold)
    return all_outliers


def analyze_extreme_days(returns_df, top_n=5, threshold=3):
    """
    Analyze days with unusually high or low returns

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Returns data
    top_n : int
        Number of extreme days to show
    threshold : float
        Z-score threshold for "extreme"

    Returns:
    --------
    dict : Extreme days for each asset
    """
    print("\n" + "=" * 60)
    print("EXTREME DAYS ANALYSIS")
    print("=" * 60)

    extreme_days = {}

    for ticker in returns_df.columns:
        returns = returns_df[ticker].dropna()

        # Find extreme days
        z_scores = np.abs(stats.zscore(returns))
        extreme_mask = z_scores > threshold
        extreme_returns = returns[extreme_mask]

        # Sort by absolute value
        extreme_sorted = extreme_returns.abs().sort_values(ascending=False)
        top_extreme = extreme_sorted.head(top_n)

        print(f"\n {ticker} - Top {top_n} Extreme Days:")
        print("-" * 40)

        if len(top_extreme) == 0:
            print("  No extreme days found")
        else:
            for date, abs_return in top_extreme.items():
                actual_return = returns[date] * 100
                direction = "📈 UP" if actual_return > 0 else "📉 DOWN"
                print(f"  {date.date()}: {direction} {actual_return:.2f}%")

                # Add context
                if abs(actual_return) > 5:
                    print(f"    Very extreme (>5%)")
                elif abs(actual_return) > 3:
                    print(f"     Extreme (>3%)")

        extreme_days[ticker] = {
            "count": len(extreme_returns),
            "dates": extreme_returns.index.tolist(),
            "values": extreme_returns.tolist(),
        }

    return extreme_days


# ============================================
# 6. REPORTING
# ============================================


def generate_summary_report(prices, returns, risk_metrics, stationarity_results):
    """
    Generate a comprehensive summary report

    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    returns : pd.DataFrame
        Returns data
    risk_metrics : dict
        Risk metrics from calculate_all_risk_metrics()
    stationarity_results : dict
        Stationarity test results
    """
    print("\n" + "=" * 60)
    print("EDA SUMMARY REPORT")
    print("=" * 60)

    # Price trends
    print("\nPRICE TRENDS:")
    for ticker in prices.columns:
        start = prices[ticker].iloc[0]
        end = prices[ticker].iloc[-1]
        change = ((end - start) / start) * 100
        direction = "🟢 UP" if change > 0 else "🔴 DOWN"
        print(f"  {ticker}: {direction} {change:+.1f}% (${start:.2f} → ${end:.2f})")

    # Volatility
    print("\nVOLATILITY (Daily):")
    for ticker in returns.columns:
        std = returns[ticker].std() * 100
        print(f"  {ticker}: ±{std:.2f}%")

    # Risk metrics
    print("\nRISK METRICS:")
    for ticker, metrics in risk_metrics.items():
        # FIX: Dynamically generate target key from dictionary metadata to prevent KeyError crash
        conf_pct = int(metrics.get("confidence_level", 0.95) * 100)
        var_key = f"VaR ({conf_pct}%)"

        print(f"  {ticker}:")
        print(f"    VaR ({conf_pct}%): {metrics[var_key]:.2f}% daily loss")
        print(f"    Sharpe: {metrics['Sharpe Ratio']:.3f}")
        print(f"    Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%")

    # Correlation
    print("\nCORRELATION:")
    corr = prices.corr()
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            col1, col2 = corr.columns[i], corr.columns[j]
            print(f"  {col1} ↔ {col2}: {corr.iloc[i, j]:.3f}")

    # Stationarity
    print("\n STATIONARITY:")
    for ticker in prices.columns:
        price_key = f"{ticker}_price"
        ret_key = f"{ticker}_returns"
        if price_key in stationarity_results:
            price_stat = (
                "✅" if stationarity_results[price_key]["is_stationary"] else "❌"
            )
            ret_stat = "✅" if stationarity_results[ret_key]["is_stationary"] else "❌"
            print(f"  {ticker}: Price {price_stat} | Returns {ret_stat}")
