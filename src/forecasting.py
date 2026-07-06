"""
Time Series Forecasting Module
Production-ready ARIMA and LSTM models with walk-forward validation,
hyperparameter optimization, and out-of-sample future horizon forecasting with confidence intervals.
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


# ============================================
# 2. ADVANCED WALK-FORWARD ARIMA
# ============================================


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

    # ========================================================
    # TASK 3 FUNCTIONALITY: FUTURE HORIZON SYSTEM
    # ========================================================
    def generate_future_horizon(self, full_returns_series, steps=126, alpha=0.05):
        """
        Generates out-of-sample forecasts into the absolute future with analytical
        per-step confidence intervals (return space) AND the fitted residual std
        needed for a proper Monte Carlo price-level band.

        Default steps=126 simulates ~6 months of trading days (21 days/month).
        alpha=0.05 generates a 95% Confidence Interval.

        FIX: fit on the pandas Series directly (not .values) so all downstream
        outputs stay well-typed, and conf_int() is accessed with array
        indexing (works whether it comes back as ndarray or DataFrame).
        """
        print(f"\nFitting full history into ARIMA{self.order} engine for out-of-sample generation...")

        full_model = ARIMA(np.asarray(full_returns_series), order=self.order)
        fitted_model = full_model.fit()

        # Extract out-of-sample forecast arrays
        forecast_results = fitted_model.get_forecast(steps=steps)
        mean_forecast_returns = np.asarray(forecast_results.predicted_mean)

        # Pull structural confidence framework bounds -- use positional indexing
        # so this works regardless of whether statsmodels returns a DataFrame
        # or a plain ndarray (depends on whether the input had pandas metadata).
        conf_int_arr = np.asarray(forecast_results.conf_int(alpha=alpha))
        lower_returns = conf_int_arr[:, 0]
        upper_returns = conf_int_arr[:, 1]

        # Residual std -- needed for Monte Carlo simulation of the PRICE-level
        # band, since naively compounding the per-step RETURN bound day-over-day
        # assumes the extreme return happens every single day and explodes.
        resid_std = np.std(fitted_model.resid)

        return {
            "mean_returns": mean_forecast_returns,
            "lower_returns": lower_returns,
            "upper_returns": upper_returns,
            "resid_std": resid_std,
        }


# ============================================
# 3. TYPE-SAFE DEEP STACKED LSTM
# ============================================


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


# ============================================
# 4. FINANCIAL DOLLAR-PRICE EVALUATOR
# ============================================


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

    # =========================================================
    # VISUAL FUTURE HORIZON PIPELINE
    # =========================================================
    def generate_future_forecast(self, months=6, alpha=0.05, n_sims=3000, seed=42):
        """
        Orchestrates full out-of-sample future price tracking with an
        expanding, statistically valid risk boundary region built via
        Monte Carlo simulation of the return-generating process.
        """
        # ~21 trading days per month
        steps = int(months * 21)

        # Pull complete series histories
        full_prices = self.price_df[self.target_col].dropna()
        full_returns = full_prices.pct_change().dropna()

        # 1. Run full-horizon ARIMA forecasting
        if not hasattr(self, 'arima'):
            self.arima = ARIMAModel(full_returns)
            self.arima.auto_fit()

        horizon = self.arima.generate_future_horizon(full_returns, steps=steps, alpha=alpha)
        mean_returns = horizon["mean_returns"]
        resid_std = horizon["resid_std"]

        # 2. Build Date Objects for the Future
        last_date = full_prices.index[-1]
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq="B")[1:]

        # 3. Point forecast: standard multiplicative inverse-differencing
        last_known_price = full_prices.iloc[-1]
        future_prices = last_known_price * np.cumprod(1 + mean_returns)

        # 4. Price-level CI via Monte Carlo (replaces the naive cumprod-of-bounds)
        rng = np.random.default_rng(seed)
        sim_returns = mean_returns.reshape(1, -1) + rng.normal(0, resid_std, size=(n_sims, steps))
        sim_price_paths = last_known_price * np.cumprod(1 + sim_returns, axis=1)

        lower_bound = np.percentile(sim_price_paths, alpha / 2 * 100, axis=0)
        upper_bound = np.percentile(sim_price_paths, (1 - alpha / 2) * 100, axis=0)

        # 5. Structured DataFrame Output Conversion
        forecast_df = pd.DataFrame({
            "Forecasted Price": future_prices,
            "Lower Bound ($)": lower_bound,
            "Upper Bound ($)": upper_bound
        }, index=future_dates)

        # 6. Production Visual Plot Generation
        self._plot_future_horizon(full_prices, forecast_df, months, alpha)

        return forecast_df

    def _plot_future_horizon(self, historical_prices, forecast_df, months, alpha):
        """Generates the official Task 3 structural visualization chart with uncertainty bands."""
        plt.subplots(figsize=(15, 6))

        # Slice historical window to make the chart readable (e.g., last 1.5 years)
        plot_history = historical_prices.iloc[-350:]

        # Plot historical prices
        plt.plot(plot_history.index, plot_history.values, color="black", linewidth=2, label="Historical Closing Price")

        # Plot future forecast path
        plt.plot(forecast_df.index, forecast_df["Forecasted Price"], color="#1f77b4", linewidth=2, linestyle="-", label=f"Future Forecast ({months} Months)")

        # Generate the expanding statistical risk envelope
        plt.fill_between(
            forecast_df.index,
            forecast_df["Lower Bound ($)"],
            forecast_df["Upper Bound ($)"],
            color="#1f77b4",
            alpha=0.15,
            label=f"{int((1-alpha)*100)}% Confidence Interval Envelope"
        )

        plt.title(f"Tesla (TSLA) Out-of-Sample {months}-Month Strategic Price Forecast", fontsize=14, fontweight="bold")
        plt.xlabel("Timeline Horizon")
        plt.ylabel("Asset Valuation in USD ($)")
        # FIX: label the boundary dynamically from the actual last historical date
        # instead of a hardcoded string that goes stale on re-runs.
        plt.axvline(
            historical_prices.index[-1],
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Forecast Horizon Boundary ({historical_prices.index[-1].strftime('%B %Y')})",
        )
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    # Add to ForecastingPipeline class (after _plot_future_horizon)

    def print_forecast_summary(self, forecast_df, months, last_price=None):
        """
        Print comprehensive forecast summary statistics.
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            DataFrame from generate_future_forecast()
        months : int or str
            Forecast horizon in months
        last_price : float, optional
            Last known price. If None, uses the last price from price_df.
        """
        if last_price is None:
            last_price = self.price_df[self.target_col].iloc[-1]
        
        start_price = last_price
        end_price = forecast_df["Forecasted Price"].iloc[-1]
        change_pct = (end_price / start_price - 1) * 100
        
        lower = forecast_df["Lower Bound ($)"].iloc[-1]
        upper = forecast_df["Upper Bound ($)"].iloc[-1]
        range_width = upper - lower
        
        print(f"\n{'-'*50}")
        print(f"{months}-MONTH FORECAST SUMMARY")
        print(f"{'-'*50}")
        print(f"Start Price:     ${start_price:.2f}")
        print(f"End Price:       ${end_price:.2f}")
        print(f"Expected Change: {change_pct:+.2f}%")
        print(f"\n95% Confidence Interval:")
        print(f"  Lower:         ${lower:.2f}")
        print(f"  Upper:         ${upper:.2f}")
        print(f"  Range Width:   ${range_width:.2f}")
        print(f"  Range/Price:   {range_width/start_price*100:.1f}%")
        
        # Directional assessment
        if abs(change_pct) < 1:
            direction = "STABLE (random walk)"
        elif change_pct > 0:
            direction = "UPWARD"
        else:
            direction = "DOWNWARD"
        print(f"\nPredicted Trend: {direction}")
        
        return {
            "start_price": start_price,
            "end_price": end_price,
            "change_pct": change_pct,
            "lower": lower,
            "upper": upper,
            "range_width": range_width,
            "direction": direction
        }

    def analyze_confidence_interval_growth(self, forecast_df, label):
        """
        Analyze how confidence intervals grow over the forecast horizon.
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            DataFrame from generate_future_forecast()
        label : str
            Label for the forecast (e.g., "6-Month")
        
        Returns:
        --------
        dict : Growth analysis metrics
        """
        initial_width = forecast_df["Upper Bound ($)"].iloc[0] - forecast_df["Lower Bound ($)"].iloc[0]
        final_width = forecast_df["Upper Bound ($)"].iloc[-1] - forecast_df["Lower Bound ($)"].iloc[-1]
        growth_factor = final_width / initial_width
        
        short_term_width = forecast_df["Upper Bound ($)"].iloc[20] - forecast_df["Lower Bound ($)"].iloc[20]  # ~1 month
        long_term_width = final_width
        
        print(f"\n{label} Forecast:")
        print(f"  Initial CI Width:  ${initial_width:.2f}")
        print(f"  Final CI Width:    ${final_width:.2f}")
        print(f"  Growth Factor:     {growth_factor:.1f}x")
        print(f"  Short-term (1 mo): ${short_term_width:.2f}")
        print(f"  Long-term (full):  ${long_term_width:.2f}")
        print(f"  Long/Short Ratio:  {long_term_width/short_term_width:.1f}x")
        
        reliability = "Low reliability" if growth_factor > 3 else "Moderate reliability"
        if growth_factor < 2:
            reliability = "High reliability"
        print(f"  Implication:       {reliability}")
        
        return {
            "initial_width": initial_width,
            "final_width": final_width,
            "growth_factor": growth_factor,
            "short_term_width": short_term_width,
            "long_term_width": long_term_width,
            "reliability": reliability
        }

    def generate_forecast_summary_report(self, forecasts, last_price=None):
        """
        Generate a complete forecast summary report for multiple horizons.
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary of {months: forecast_df} pairs
        last_price : float, optional
            Last known price. If None, uses the last price from price_df.
        
        Returns:
        --------
        pd.DataFrame : Summary DataFrame
        """
        if last_price is None:
            last_price = self.price_df[self.target_col].iloc[-1]
        
        print("\n" + "="*60)
        print("COMPLETE FORECAST SUMMARY REPORT")
        print("="*60)
        
        summary_data = []
        
        for months, forecast_df in forecasts.items():
            print(f"\n{'='*60}")
            print(f"{months}-MONTH FORECAST")
            print(f"{'='*60}")
            
            summary = self.print_forecast_summary(forecast_df, months, last_price)
            growth = self.analyze_confidence_interval_growth(forecast_df, f"{months}-Month")
            
            summary_data.append({
                "Horizon (months)": months,
                "Start Price": summary["start_price"],
                "End Price": summary["end_price"],
                "Change (%)": summary["change_pct"],
                "95% CI Lower": summary["lower"],
                "95% CI Upper": summary["upper"],
                "Range Width": summary["range_width"],
                "Range/Price (%)": summary["range_width"] / summary["start_price"] * 100,
                "Trend": summary["direction"],
                "CI Growth Factor": growth["growth_factor"],
                "Reliability": growth["reliability"]
            })
        
        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        return summary_df

    def analyze_market_opportunities_risks(self, forecast_df, months):
        """
        Analyze market opportunities and risks from the forecast.
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            DataFrame from generate_future_forecast()
        months : int
            Forecast horizon in months
        
        Returns:
        --------
        dict : Opportunities and risks assessment
        """
        last_price = self.price_df[self.target_col].iloc[-1]
        end_price = forecast_df["Forecasted Price"].iloc[-1]
        lower = forecast_df["Lower Bound ($)"].iloc[-1]
        upper = forecast_df["Upper Bound ($)"].iloc[-1]
        change_pct = (end_price / last_price - 1) * 100
        
        upside = (upper / last_price - 1) * 100
        downside = (lower / last_price - 1) * 100
        range_ratio = (upper - lower) / last_price * 100
        
        print(f"\n{'-'*50}")
        print(f"{months}-MONTH MARKET OUTLOOK")
        print(f"{'-'*50}")
        
        print(f"\n OPPORTUNITIES:")
        if upside > 20:
            print(f"  • Significant upside potential: +{upside:.1f}%")
            print(f"  • Bullish scenario: Price could reach ${upper:.2f}")
        else:
            print(f"  • Limited upside potential: +{upside:.1f}%")
        
        print(f"\n  RISKS:")
        if downside < -20:
            print(f"  • Significant downside risk: {downside:.1f}%")
            print(f"  • Bearish scenario: Price could drop to ${lower:.2f}")
        else:
            print(f"  • Limited downside risk: {downside:.1f}%")
        
        print(f"\n VOLATILITY & UNCERTAINTY:")
        if range_ratio > 40:
            print(f"  • HIGH uncertainty: {range_ratio:.1f}% price range")
        elif range_ratio > 20:
            print(f"  • MODERATE uncertainty: {range_ratio:.1f}% price range")
        else:
            print(f"  • LOW uncertainty: {range_ratio:.1f}% price range")
        
        print(f"\n RECOMMENDATION:")
        if abs(change_pct) < 2:
            print("  • NEUTRAL - No clear directional signal")
            print("  • Random walk behavior suggests efficient market")
        elif change_pct > 0:
            print(f"  • BULLISH - Expected upside of {change_pct:.1f}%")
        else:
            print(f"  • BEARISH - Expected downside of {abs(change_pct):.1f}%")
        
        return {
            "upside": upside,
            "downside": downside,
            "range_ratio": range_ratio,
            "recommendation": "NEUTRAL" if abs(change_pct) < 2 else ("BULLISH" if change_pct > 0 else "BEARISH")
        }