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
│ └── future_forcasting.ipynb # Future horizon forecasting 
├── src/
│ ├── init.py
│ ├── data_processing.py # Core utilities
│ └── forecasting.py # ARIMA & LSTM models + future forecasting
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

**Correlation Matrix**:
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