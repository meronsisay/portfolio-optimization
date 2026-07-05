"""
Time Series Forecasting Module
Production-ready ARIMA and LSTM models with walk-forward validation.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Classical Modeling
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler

# Deep Learning Framework
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA

# ============================================
# 1. ROBUST DATA PREPARER (PRICE & RETURN ALIGNMENT)
# ============================================


class DataPreparer:
    """
    Handles data transformations without leakage. Processes stationary returns
    internally for training while preserving raw price arrays for final reconstruction.
    """

    def __init__(self, price_df, target_col="TSLA", window_size=60):
        self.price_df = price_df.copy()
        self.target_col = target_col
        self.window_size = window_size

        # Compute daily arithmetic returns to achieve stationarity
        self.returns_df = self.price_df[[self.target_col]].pct_change().dropna()

        self.train_returns = None
        self.test_returns = None
        self.train_dates = None
        self.test_dates = None

        self.actual_test_prices = None
        self.baseline_prices = None
        self.scaler = MinMaxScaler(feature_range=(0.1, 0.9))

    def split_data(self, test_size=0.2):
        """Chronologically segments data into train/test splits without shuffling."""
        target_returns = self.returns_df[self.target_col]
        target_prices = self.price_df[self.target_col].loc[target_returns.index]

        split_idx = int(len(target_returns) * (1 - test_size))

        # Train partition metrics (clean returns history)
        self.train_returns = target_returns.iloc[:split_idx]
        self.train_dates = target_returns.index[:split_idx]

        # Test partition metrics (includes lookback window for LSTM sequence tracking)
        self.test_returns = target_returns.iloc[split_idx - self.window_size :]
        self.test_dates = target_returns.index[split_idx:]

        # Ground truth structures for price evaluation
        self.actual_test_prices = target_prices.iloc[split_idx:].values
        # Baseline prices P_(t-1) used to mathematically reconstruct daily price from return
        self.baseline_prices = target_prices.iloc[split_idx - 1 : -1].values

        print(f"\n[Data Split Summary]")
        print(
            f"  Training Period: {self.train_dates[0].date()} to {self.train_dates[-1].date()} ({len(self.train_returns):,} days)"
        )
        print(
            f"  Testing Period:  {self.test_dates[0].date()} to {self.test_dates[-1].date()} ({len(self.actual_test_prices):,} days)"
        )
        return self

    def prepare_lstm_data(self):
        if self.train_returns is None:
            self.split_data()

        train_arr = self.train_returns.values.reshape(-1, 1)
        test_arr = self.test_returns.values.reshape(-1, 1)

        # Scale parameters bound strictly to the isolated training partition
        train_scaled = self.scaler.fit_transform(train_arr).flatten()
        test_scaled = self.scaler.transform(test_arr).flatten()

        X_train, y_train = self._create_sequences(train_scaled)
        X_test, y_test = self._create_sequences(test_scaled)

        return X_train, X_test, y_train, y_test, self.scaler

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i - self.window_size : i])
            y.append(data[i])
        return np.array(X).reshape(-1, self.window_size, 1), np.array(y).reshape(-1, 1)


class ARIMAModel:
    def __init__(self, train_returns):
        self.train_returns = train_returns
        self.order = (1, 0, 1)
        self.model = None

    def auto_fit(self, seasonal=False, seasonal_period=5):
        print("\nSearching optimal ARIMA parameters on returns space...")
        model = auto_arima(
            self.train_returns,
            start_p=0,
            max_p=5,
            start_d=0,
            max_d=2,
            start_q=0,
            max_q=5,
            seasonal=seasonal,
            m=seasonal_period if seasonal else 1,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        self.order = model.order
        self.model = model
        print(f"  Optimal Parameter Set Chosen: ARIMA{self.order}")
        return self

    def rolling_forecast(self, test_returns):
        """Applies walk-forward validation updates across the test horizon."""
        history = list(self.train_returns.values)
        predictions = []

        for t in range(len(test_returns)):
            model = ARIMA(history, order=self.order)
            model_fitted = model.fit()
            yhat = model_fitted.forecast()[0]
            predictions.append(yhat)
            history.append(test_returns[t])

        return np.array(predictions)


class LSTMModel:
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.history = None

    def build(self, lstm_units=64, dropout=0.2, learning_rate=0.001, num_layers=2):
        model = Sequential()
        input_shape = (self.X_train.shape[1], 1)

        for i in range(num_layers):
            is_last = i == num_layers - 1
            units = max(lstm_units // (2**i), 16)

            if i == 0:
                model.add(
                    LSTM(units, input_shape=input_shape, return_sequences=not is_last)
                )
            else:
                model.add(LSTM(units, return_sequences=not is_last))
            model.add(Dropout(dropout))

        model.add(Dense(1))
        model.compile(
            optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"]
        )
        self.model = model
        return self

    def train(self, epochs=20, batch_size=64, patience=4, verbose=0):
        callbacks = [
            EarlyStopping(
                patience=patience,
                restore_best_weights=True,
                monitor="val_loss" if self.X_test is not None else "loss",
            )
        ]
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                (self.X_test, self.y_test) if self.X_test is not None else None
            ),
            callbacks=callbacks,
            verbose=verbose,
        )
        return self

    def optimize_hyperparameters(self, param_grid, validation_split=0.1, epochs=10):
        """Performs grid search tuning using a localized internal training split."""
        grid = ParameterGrid(param_grid)
        best_loss = float("inf")
        best_params = None

        split_idx = int(len(self.X_train) * (1 - validation_split))
        X_t, y_t = self.X_train[:split_idx], self.y_train[:split_idx]
        X_v, y_v = self.X_train[split_idx:], self.y_train[split_idx:]

        print(f"Tuning LSTM parameters over {len(grid)} optimization candidates...")
        for params in grid:
            temp_model = LSTMModel(X_t, y_t, X_v, y_v)
            temp_model.build(
                lstm_units=params.get("lstm_units", 64),
                dropout=params.get("dropout", 0.2),
                learning_rate=params.get("learning_rate", 0.001),
                num_layers=params.get("num_layers", 2),
            )
            temp_model.train(epochs=epochs, batch_size=64, patience=2, verbose=0)
            val_loss = min(temp_model.history.history["val_loss"])

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params

        print(f"  Optimal Hyperparameters Selected: {best_params}")
        return best_params

    def predict(self, X=None):
        X_pred = X if X is not None else self.X_test
        return self.model.predict(X_pred, verbose=0)


class ModelEvaluator:
    def __init__(self, y_true_prices):
        self.y_true = np.array(y_true_prices).flatten()
        self.predictions = {}

    def add_predictions(self, name, y_pred_prices):
        self.predictions[name] = np.array(y_pred_prices).flatten()

    def compare(self):
        results = []
        print("\n" + "=" * 65)
        print("FINAL DOLLAR-PRICE EVALUATION METRICS")
        print("=" * 65)

        for name, pred in self.predictions.items():
            truncated_pred = pred[: len(self.y_true)]
            y_true_truncated = self.y_true[: len(truncated_pred)]

            # Evaluate real dollar variance errors
            mae = mean_absolute_error(y_true_truncated, truncated_pred)
            rmse = np.sqrt(mean_squared_error(y_true_truncated, truncated_pred))

            with np.errstate(divide="ignore", invalid="ignore"):
                mape = (
                    np.mean(
                        np.abs((y_true_truncated - truncated_pred) / y_true_truncated)
                    )
                    * 100
                )
            mape = np.nan_to_num(mape)

            dir_true = np.sign(
                np.diff(np.insert(y_true_truncated, 0, y_true_truncated[0]))
            )
            dir_pred = np.sign(
                np.diff(np.insert(truncated_pred, 0, y_true_truncated[0]))
            )
            dir_acc = np.mean(dir_true == dir_pred) * 100

            results.append(
                {
                    "Model": name,
                    "MAE ($)": round(mae, 4),
                    "RMSE ($)": round(rmse, 4),
                    "MAPE": f"{mape:.2f}%",
                    "Directional Accuracy": f"{dir_acc:.2f}%",
                }
            )

        df = pd.DataFrame(results).sort_values("RMSE ($)")
        print(df.to_string(index=False))
        print("=" * 65)
        return df

    def plot(self, title="Stock Price Target Forecast", dates=None):
        fig, ax = plt.subplots(figsize=(14, 6))
        x = dates if dates is not None else range(len(self.y_true))

        ax.plot(
            x, self.y_true, color="black", linewidth=2, label="Actual Closing Price"
        )
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, (name, pred) in enumerate(self.predictions.items()):
            ax.plot(
                x[: len(pred)],
                pred,
                color=colors[i % len(colors)],
                linewidth=1.5,
                linestyle="--",
                label=name,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date Timeline")
        ax.set_ylabel("Price in USD ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================
# 5. ORCHESTRATION PIPELINE
# ============================================


class ForecastingPipeline:
    def __init__(self, price_df, target_col="TSLA", window_size=60, test_size=0.2):
        self.price_df = price_df
        self.target_col = target_col
        self.window_size = window_size
        self.test_size = test_size

    def run(self, optimize_lstm=True, arima_seasonal=False, lstm_epochs=20):
        # 1. Pipeline partitioning setup
        self.preparer = DataPreparer(self.price_df, self.target_col, self.window_size)
        self.preparer.split_data(self.test_size)
        X_train, X_test, y_train, y_test, scaler = self.preparer.prepare_lstm_data()

        actual_test_returns = self.preparer.test_returns.iloc[self.window_size :].values
        baseline_prices = self.preparer.baseline_prices
        actual_test_prices = self.preparer.actual_test_prices

        # 2. Run Classical Engine on Return Arrays
        self.arima = ARIMAModel(self.preparer.train_returns)
        self.arima.auto_fit(seasonal=arima_seasonal)
        arima_return_forecast = self.arima.rolling_forecast(actual_test_returns)

        # 3. Run Deep Learning Engine on Sequences
        self.lstm = LSTMModel(X_train, y_train, X_test, y_test)
        if optimize_lstm:
            param_grid = {
                "lstm_units": [32, 64],
                "num_layers": [1, 2],
                "learning_rate": [0.005, 0.001],
            }
            best_params = self.lstm.optimize_hyperparameters(param_grid, epochs=8)
            self.lstm.build(
                lstm_units=best_params["lstm_units"],
                num_layers=best_params["num_layers"],
                learning_rate=best_params["learning_rate"],
            )
        else:
            self.lstm.build(lstm_units=64, num_layers=2, learning_rate=0.001)

        self.lstm.train(epochs=lstm_epochs, batch_size=64, verbose=0)
        lstm_preds_scaled = self.lstm.predict()
        lstm_return_forecast = scaler.inverse_transform(lstm_preds_scaled).flatten()

        # =========================================================
        # INVERSE-DIFFERENCING: RECONSTRUCT RAW DOLLAR STOCK PRICES
        # Formula: P_t = P_(t-1) * (1 + R_t)
        # =========================================================
        arima_price_forecast = baseline_prices * (1 + arima_return_forecast)
        lstm_price_forecast = baseline_prices * (1 + lstm_return_forecast)

        # 4. Align and compute errors relative to true Dollar Price trajectories
        self.evaluator = ModelEvaluator(actual_test_prices)
        self.evaluator.add_predictions(
            "ARIMA (Walk-Forward Price)", arima_price_forecast
        )
        self.evaluator.add_predictions("LSTM (Optimized Price)", lstm_price_forecast)

        summary_df = self.evaluator.compare()
        self.evaluator.plot(
            title=f"Tesla ({self.target_col}) Stock Price Forecasting Evaluation",
            dates=self.preparer.test_dates,
        )

        return summary_df
