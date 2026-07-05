"""
Simple tests for forecasting module
Run with: pytest tests/
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting import DataPreparer, ARIMAModel, ModelEvaluator


# ============================================
# SAMPLE DATA
# ============================================

def create_sample_prices():
    """Create sample price data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
    return pd.DataFrame({'TSLA': prices}, index=dates)


# ============================================
# TESTS
# ============================================

class TestDataPreparer:
    """Test DataPreparer class"""
    
    def test_split_data(self):
        prices = create_sample_prices()
        preparer = DataPreparer(prices, target_col='TSLA', window_size=10)
        preparer.split_data(test_size=0.2)
        
        assert len(preparer.train_returns) > 0
        assert len(preparer.actual_test_prices) > 0
        assert len(preparer.train_dates) > 0
        assert len(preparer.test_dates) > 0
    
    def test_prepare_lstm_data(self):
        prices = create_sample_prices()
        preparer = DataPreparer(prices, target_col='TSLA', window_size=10)
        X_train, X_test, y_train, y_test, scaler = preparer.prepare_lstm_data()
        
        assert X_train.shape[1] == 10
        assert X_test.shape[1] == 10
        assert len(y_train) > 0
        assert len(y_test) > 0


class TestARIMAModel:
    """Test ARIMAModel class"""
    
    def test_auto_fit(self):
        prices = create_sample_prices()
        returns = prices.pct_change().dropna()
        
        arima = ARIMAModel(returns['TSLA'])
        arima.auto_fit(seasonal=False)
        
        assert arima.order is not None
        assert len(arima.order) == 3
    
    def test_rolling_forecast(self):
        prices = create_sample_prices()
        returns = prices.pct_change().dropna()
        
        train = returns['TSLA'].iloc[:50]
        test = returns['TSLA'].iloc[50:70].values
        
        arima = ARIMAModel(train)
        arima.auto_fit(seasonal=False)
        forecast = arima.rolling_forecast(test)
        
        assert len(forecast) == len(test)


class TestModelEvaluator:
    """Test ModelEvaluator class"""
    
    def test_add_predictions(self):
        y_true = np.array([100, 101, 102])
        evaluator = ModelEvaluator(y_true)
        evaluator.add_predictions('Test', np.array([99, 100, 101]))
        
        assert 'Test' in evaluator.predictions
        assert len(evaluator.predictions['Test']) == 3
    
    def test_compare(self):
        y_true = np.array([100, 101, 102])
        evaluator = ModelEvaluator(y_true)
        evaluator.add_predictions('Model1', np.array([99, 100, 101]))
        evaluator.add_predictions('Model2', np.array([100, 101, 102]))
        
        df = evaluator.compare()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'Model' in df.columns
        assert 'MAE ($)' in df.columns


# ============================================
# RUN TESTS
# ============================================

if __name__ == "__main__":
    test = TestDataPreparer()
    test.test_split_data()
    print( "test_split_data passed")
    test.test_prepare_lstm_data()
    print(" test_prepare_lstm_data passed")
    
    test2 = TestARIMAModel()
    test2.test_auto_fit()
    print(" test_auto_fit passed")
    test2.test_rolling_forecast()
    print(" test_rolling_forecast passed")
    
    test3 = TestModelEvaluator()
    test3.test_add_predictions()
    print(" test_add_predictions passed")
    test3.test_compare()
    print(" test_compare passed")
    
    print("\n All tests passed!")