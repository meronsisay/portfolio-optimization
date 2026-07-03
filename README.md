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
│ └── eda_and_preprocessing.ipynb # Task 1: EDA
├── src/
│ ├── init.py
│ └── data_processing.py # Core utilities
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

##  Data Preprocessing & EDA (Complete)

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

## Environment Setup

### Prerequisites
* **Python**: `3.11`
* **Version Control**: `Git`

### Setup Instructions

Follow these step-by-step instructions to prepare your local development environment.

#### 1. Clone the Repository

```bash
git clone [https://github.com/meronsisay/portfolio-optimization.git](https://github.com/meronsisay/portfolio-optimization.git)
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
jupyter notebook notebooks/01_data_preprocessing_eda.ipynb