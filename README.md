# Portfolio Optimization - Time Series Forecasting

##  Project Overview

A comprehensive financial analysis project applying time series forecasting and portfolio optimization to three key assets: **TSLA** (Tesla), **BND** (Bond ETF), and **SPY** (S&P 500 ETF).

**My Role**: Financial Analyst at GMF Investments analyzing historical data, building predictive models, and recommending portfolio adjustments based on forecasted trends.

---

## Project Structure
```
portfolio-optimization/
├── .github/workflows/unittests.yml # CI/CD pipeline
├── data/processed/ # Cleaned data files
├── notebooks/
│ ├── eda_and_preprocessing.ipynb # EDA
│ ├── model_training.ipynb #  Forecasting
│ ├── future_forcasting.ipynb # Future horizon forecasting
│ ├── portfolio_optimization.ipynb # MPT portfolio optimization
│ └── backtesting.ipynb # Strategy backtest vs. benchmark
├── src/
│ ├── init.py
│ ├── data_processing.py # Core utilities
│ ├── forecasting.py # ARIMA & LSTM models + future forecasting
│ ├── portfolio_optimization.py # PyPortfolioOpt-based MPT optimizer
│ └── backtesting.py # Backtester vs. 60/40 benchmark
├── tests/
│ └── test_data_processing.py # Unit tests
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Data

### Assets Analyzed

| Asset | Ticker | Risk Profile |
|-------|--------|--------------|
| Tesla | TSLA | High Risk |
| Vanguard Bond ETF | BND | Low Risk |
| S&P 500 ETF | SPY | Moderate Risk |

- **Period**: Jan 2015 - Jun 2026 (11.5 years)
- **Data Points**: 2,888 daily observations per asset
- **Fields**: Open, High, Low, Close, Adj Close, Volume

---

##  Data Preprocessing & EDA 

- Fetched data from yfinance
- Cleaned and validated data (no missing values)
- Performed exploratory data analysis
- Tested stationarity (ADF test)
- Calculated risk metrics (VaR, Sharpe Ratio)
- Created 5+ visualizations

### Key Findings:

| Asset | Total Return | Daily Volatility | Max Drawdown | Sharpe Ratio |
|-------|--------------|------------------|--------------|--------------|
| TSLA | +2,716% | ±3.60% | -73.63% | 0.794 |
| SPY | +337% | ±1.11% | -33.72% | 0.818 |
| BND | +24% | ±0.33% | -18.58% | 0.381 |

**Price Correlation Matrix** (Adj Close levels — inflated by shared upward trend, not a risk-modeling input):
- TSLA ↔ SPY: 0.919 (Very Strong)
- BND ↔ SPY: 0.656 (Moderate)
- TSLA ↔ BND: 0.649 (Moderate)

**Stationarity Results**:
- **Prices**: NON-STATIONARY → Need differencing (d=1) for ARIMA
- **Returns**: STATIONARY → Ready for modeling

---

## Time Series Forecasting 

### Models Implemented

| Model | Parameters | Description |
|-------|------------|-------------|
| **ARIMA** | (0, 0, 0) | Random walk model - no autocorrelation in returns |
| **LSTM** | 2 layers, 32 units | Stacked deep learning model with dropout |

### Key Results (TSLA Price Forecasts)

| Metric | ARIMA | LSTM |
|--------|-------|------|
| **MAE** | $8.73 | $8.76 |
| **RMSE** | $11.69 | $11.72 |
| **MAPE** | 2.78% | 2.78% |
| **Directional Accuracy** | 49.83% | 49.65% |

### Key Insights

1. **ARIMA(0,0,0) Confirms Random Walk**
   - Tesla's returns show no autocorrelation
   - Past prices do NOT predict future prices
   - Consistent with Efficient Market Hypothesis

2. **LSTM Learned Successfully**
   - Loss reduced by 76.5% during training
   - Validation loss decreased (no overfitting)
   - Complex DL matches simpler ARIMA

3. **Low MAPE (2.78%)**
   - Predictions track actual prices accurately
   - Structural stability confirmed

4. **Directional Accuracy ~50%**
   - Expected for random walk
   - No predictive edge from historical data alone

### Price Reconstruction

- Models forecast **returns** (stationary), then convert to **prices**
- Formula: `Price[t] = Price[t-1] × (1 + Return[t])`
- Final evaluation on dollar prices for business relevance

---

## Future Market Trend Forecasting

Using the best-performing model from Task 2 (**ARIMA(0,0,0)**), refit on the full price history, to project TSLA prices 6 and 12 months beyond the end of the dataset, with confidence intervals built via **Monte Carlo simulation** of the return-generating process rather than a naive compounding of per-step bounds.

### Model & Data Overview

| Item | Details |
|---|---|
| Data Period | 2015-01-02 to 2026-06-29 (2,888 days) |
| Target | Tesla (TSLA) |
| Current Price | $411.84 |
| Optimal Model | ARIMA(0,0,0) — Random Walk |
| Training Period | 2015-01-05 to 2024-03-07 (2,309 days) |
| Testing Period | 2024-03-08 to 2026-06-29 (578 days) |

### Forecast Results

| Horizon | End Price | Change | 95% CI Range | Reliability |
|---|---|---|---|---|
| 6 Months | $516.42 | +25.39% | $213 – $1,060 | Low |
| 12 Months | $647.56 | +57.24% | $182 – $1,684 | Very Low |

### Confidence Interval Growth

| Metric | 6-Month | 12-Month |
|---|---|---|
| Initial CI Width | $59.65 | $58.99 |
| Final CI Width | $846.91 | $1,502.51 |
| Growth Factor | 14.2x | 25.5x |
| Range/Price | 205.6% | 364.8% |

### Market Outlook

| Aspect | 6-Month | 12-Month |
|---|---|---|
| Upside | +157.5% → $1,060 | +308.9% → $1,684 |
| Expected (point est.) | +25.4% → $516 | +57.2% → $648 |
| Downside | -48.2% → $213 | -55.9% → $182 |
| Uncertainty | HIGH (205.6%) | HIGH (364.8%) |
| Directional lean | Bullish | Bullish |

### Key Insights

1. **Upward point forecast is a drift artifact, not a detected signal**
   - ARIMA selected order (0,0,0) — i.e., **no autocorrelation** in TSLA's returns, a textbook random walk consistent with the Efficient Market Hypothesis.
   - The +25%/+57% point forecasts come from the model projecting forward the **historical mean daily return**, which is pulled sharply positive by TSLA's 2020–2021 rally embedded in the training window — not from any detected momentum or pattern.

2. **Confidence intervals widen sharply with horizon**
   - CI width grows from ~$60 (day 1) to **$847 by 6 months (14.2x)** and **$1,503 by 12 months (25.5x)**.
   - By 12 months the band spans **365% of the forecast price** ($182–$1,684) — a near-halving to a near-quadrupling of the current price.
   - Consistent with random-walk uncertainty compounding over a multiplicative price path.

3. **Reliability decreases with horizon**

   | Horizon | Reliability | Implication |
   |---|---|---|
   | 1 Month | Moderate | Directionally usable with caution |
   | 3 Months | Low | Limited confidence |
   | 6 Months | Low–Very Low | Signal increasingly overwhelmed by uncertainty |
   | 12 Months | Very Low | Not usable for high-confidence decisions |

4. **"Bullish" describes direction, not confidence**
   - The point estimate leans bullish at both horizons, but the width of the confidence band means this should be read as one plausible central path among many, not a reliable price target — especially at the 12-month mark.


---
## Portfolio Optimization

Using Modern Portfolio Theory (MPT), TSLA's forecasted return (from Task 3's ARIMA
12-month projection) was combined with BND and SPY's historical average returns to
construct an optimized three-asset portfolio.

##  Expected Returns & Risk

### Expected Returns (Annualized)

| Asset | Source | Expected Return |
|---|---|---|
| TSLA | ARIMA Forecast (12-month) | 57.24% |
| SPY | Historical Average | 14.43% |
| BND | Historical Average | 2.03% |

### Covariance Matrix (Annualized)

| | TSLA | BND | SPY |
|---|---|---|---|
| **TSLA** | 0.326926 | 0.001789 | 0.049820 |
| **BND** | 0.001789 | 0.002822 | 0.001086 |
| **SPY** | 0.049820 | 0.001086 | 0.031169 |

**Key Observations:**
- TSLA carries by far the highest risk (variance 0.327 → ~57.2% annual volatility) — coincidentally close in magnitude to its expected return, underscoring how much uncertainty accompanies that upside.
- BND is the lowest-risk asset (variance 0.003 → ~5.3% annual volatility).
- Stock–bond covariances (TSLA↔BND: 0.0018, SPY↔BND: 0.0011) are very low relative to stock–stock covariance (TSLA↔SPY: 0.050) — meaningful diversification benefit from holding bonds alongside equities. Note: the **return-based** TSLA↔SPY correlation implied by this matrix is ~0.49 — the 0.919 figure quoted above under EDA is a *price-level* correlation (inflated by both assets simply trending upward over 11 years) and is not the number that drives this optimization.

###  Portfolio Comparison

| Metric | Max Sharpe | Min Volatility |
|---|---|---|
| TSLA Weight | 15.25% | 0.00% |
| BND Weight | 60.08% | 94.54% |
| SPY Weight | 24.67% | 5.46% |
| Expected Return | 13.50% | 2.70% |
| Volatility | 12.21% | 5.22% |
| Sharpe Ratio | 1.106 | 0.517 |

### Recommended Portfolio: Maximum Sharpe Ratio (Tangency Portfolio)

| Asset | Weight | Purpose |
|---|---|---|
| BND | 60.08% | Stability and income |
| SPY | 24.67% | Broad market growth |
| TSLA | 15.25% | High return potential |

**Expected Performance**
- Annual Return: **13.50%**
- Annual Volatility: **12.21%**
- Sharpe Ratio: **1.106** *(0% risk-free rate assumption)*

###  Why This Portfolio?

| Reason | Explanation |
|---|---|
| Balanced | 15% TSLA captures upside exposure without concentrating the portfolio's risk |
| Diversified | Combines equities (TSLA + SPY) with bonds (BND) across different risk profiles |
| Risk-Adjusted | Sharpe Ratio of 1.106 (>1.0) indicates strong return per unit of risk taken |
| Practical | A 15/60/25 split is realistic and implementable, not a corner-case allocation |

**Note on TSLA's weight:** despite carrying the highest expected return input (57.24%) of the three assets, the optimizer allocated only 15.25% to TSLA rather than concentrating heavily in it. TSLA's variance (0.327) is so large relative to its return that adding more TSLA increases portfolio risk faster than it improves risk-adjusted return — the optimizer is weighing the *forecasted* return against actual *historical* volatility rather than blindly maximizing return. Reassuring, given the "very low reliability" rating on the 12-month TSLA forecast above: the portfolio construction step doesn't over-commit to a forecast that carries substantial uncertainty.

---

## Strategy Backtesting

The Max Sharpe portfolio (15% TSLA / 60% BND / 25% SPY) was simulated over the final 12 months of the dataset (**Jun 29, 2025 – Jun 29, 2026**) — data the models never trained on — and compared against a static **60% SPY / 40% BND** passive benchmark.

### Performance Summary

| Metric | Strategy (15/60/25) | Benchmark (60/40) |
|---|---|---|
| Total Return | 11.36% | 13.64% |
| Annualized Volatility | 9.61% | 8.07% |
| Sharpe Ratio *(0% rf)* | 1.19 | 1.71 |
| Max Drawdown | -6.63% | -5.84% |

*(Simulation assumes weights are held constant and implicitly rebalanced daily; no transaction costs modeled.)*

### Did the Strategy Outperform?

**No.** The MPT-optimized portfolio underperformed the simple passive benchmark on every metric — lower return, higher volatility, deeper drawdown. The main driver: TSLA's forecasted 57.24% annual return (the input that shaped the optimizer's allocation) did not materialize over the actual backtest window, and the position still carried its full historical volatility — a cost the optimizer paid for upside that didn't show up.

### Viability & Limitations

This single 12-month test doesn't invalidate the model-driven approach outright, but it's a clear caution: an optimizer is only as good as its return assumptions, and a forecast already flagged as "very low reliability" in Task 3 flowed directly into a real allocation decision here. Key limitations: one backtest year is too short to judge across market regimes, no rebalancing costs or taxes are modeled, and only three assets limits true diversification. Before any real deployment, this strategy would need capping TSLA's forecast-driven weight, testing over multiple historical windows, and modeling realistic trading frictions.

---

## Environment Setup

### Prerequisites
* **Python**: `3.11`
* **Version Control**: `Git`

### Setup Instructions

Follow these step-by-step instructions to prepare your local development environment.

#### 1. Clone the Repository

```bash
git clone https://github.com/meronsisay/portfolio-optimization.git
cd portfolio-optimization

# Create the virtual environment using Python 3.11
python3.11 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate  

# On Windows (Command Prompt):
# venv\Scripts\activate.bat

# On Windows (PowerShell):
# venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

# running jupyter notebook
jupyter notebook notebooks/eda_and_preprocessing.ipynb
```