"""
tests for data_processing functions
Run with: pytest tests/
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions
from src.data_processing import (
    fetch_stock_data,
    combine_prices,
    clean_data,
    check_missing,
    calculate_returns,
    calculate_risk_metrics,
    detect_outliers
)


def create_sample_data():
    """Create sample price data"""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = {
        'TSLA': [100 + i*2 for i in range(10)],
        'BND': [50 + i*0.5 for i in range(10)],
        'SPY': [200 + i*1.5 for i in range(10)]
    }
    return pd.DataFrame(data, index=dates)


def create_sample_dict():
    """Create sample dictionary"""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data_dict = {}
    for ticker in ['TSLA', 'BND', 'SPY']:
        df = pd.DataFrame({
            'Open': [100 + i*2 for i in range(10)],
            'High': [105 + i*2 for i in range(10)],
            'Low': [95 + i*2 for i in range(10)],
            'Close': [100 + i*2 for i in range(10)],
            'Adj Close': [100 + i*2 for i in range(10)],
            'Volume': [1000000 + i*10000 for i in range(10)]
        }, index=dates)
        data_dict[ticker] = df
    return data_dict


class TestDataProcessing:
    """Test data processing functions"""

    def test_calculate_returns(self):
        """Test returns calculation"""
        prices = create_sample_data()
        returns = calculate_returns(prices)
        
        # Check shape
        assert returns.shape == (10, 3)
        
        # Check first row is NaN (no previous day)
        assert pd.isna(returns.iloc[0, 0])
        
        # Check calculation
        expected = (prices.iloc[1, 0] - prices.iloc[0, 0]) / prices.iloc[0, 0]
        assert abs(returns.iloc[1, 0] - expected) < 0.0001

    def test_combine_prices(self):
        """Test combining prices from dictionary"""
        data_dict = create_sample_dict()
        prices = combine_prices(data_dict, column='Close')
        
        assert prices.shape == (10, 3)
        assert list(prices.columns) == ['TSLA', 'BND', 'SPY']
        assert not prices.isnull().any().any()

    def test_clean_data(self):
        """Test data cleaning"""
        # Create data with missing values
        data_dict = create_sample_dict()
        data_dict['TSLA'].loc[data_dict['TSLA'].index[0], 'Close'] = np.nan
        
        cleaned = clean_data(data_dict, method='ffill')
        
        # Check missing values are filled
        assert cleaned['TSLA']['Close'].iloc[0] == cleaned['TSLA']['Close'].iloc[1]
        assert cleaned['TSLA'].isnull().sum().sum() == 0

    def test_check_missing(self):
        """Test missing value check"""
        data_dict = create_sample_dict()
        check_missing(data_dict)
        assert True

    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        metrics = calculate_risk_metrics(returns, confidence=0.95)
        
        assert 'Sharpe Ratio' in metrics
        assert 'VaR (95%)' in metrics
        assert 'Max Drawdown (%)' in metrics
        assert 'Daily Mean (%)' in metrics

    def test_detect_outliers(self):
        """Test outlier detection"""
        # Create data with clear outlier
        returns = pd.Series([0.001] * 99 + [10.0])
        outliers = detect_outliers(returns, threshold=3)
        
        assert len(outliers) == 1
        assert outliers.iloc[0] == 10.0


if __name__ == "__main__":
    print("Running tests manually...")
    test = TestDataProcessing()
    
    test.test_calculate_returns()
    print(" test_calculate_returns passed")
    
    test.test_combine_prices()
    print(" test_combine_prices passed")
    
    test.test_clean_data()
    print(" test_clean_data passed")
    
    test.test_check_missing()
    print(" test_check_missing passed")
    
    test.test_calculate_risk_metrics()
    print(" test_calculate_risk_metrics passed")
    
    test.test_detect_outliers()
    print(" test_detect_outliers passed")
    
    print("\nAll tests passed!")