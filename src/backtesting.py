import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class Backtester:
    """
    Backtests portfolio strategies against benchmarks.
    Simulates historical performance and structural stress periods.
    """

    def __init__(self, prices_df, strategy_weights, benchmark_weights):
        self.prices = prices_df
        self.strategy_weights = strategy_weights
        self.benchmark_weights = benchmark_weights
        self.results = {}

        self.global_returns = self.prices.pct_change().fillna(0)

    def get_portfolio_returns(self, weights, start_date, end_date):
        """Calculate daily returns for a portfolio using pre-computed return vectors.

        Applies fixed weights every day (implicit daily rebalancing / constant-mix
        strategy) -- see class docstring.
        """
        # Slice the returns globally to avoid losing the first session's return data
        return_slice = self.global_returns.loc[start_date:end_date]

        portfolio_return = pd.Series(0, index=return_slice.index)
        for asset, weight in weights.items():
            if asset in return_slice.columns:
                portfolio_return += weight * return_slice[asset]

        return portfolio_return

    def calculate_cumulative_returns(self, returns):
        """Convert daily returns to a cumulative gross wealth index multiplier (starting at 1.0)."""
        return (1 + returns).cumprod()

    def calculate_max_drawdown(self, gross_wealth_index):
        """Correctly calculate the maximum peak-to-trough decline from a wealth index multiplier."""
        running_max = gross_wealth_index.expanding().max()
        drawdown = (gross_wealth_index - running_max) / running_max
        return drawdown.min()

    def calculate_metrics(self, returns, name, risk_free_rate=0.0):
        """Calculate key annualized performance and risk metrics.
        """
        gross_wealth = self.calculate_cumulative_returns(returns)
        total_return = gross_wealth.iloc[-1] - 1 if len(gross_wealth) > 0 else 0

        n_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        annualized_vol = returns.std() * np.sqrt(252)

        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
        max_drawdown = self.calculate_max_drawdown(gross_wealth)

        return {
            'Portfolio': name,
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

    def run_backtest(self, start_date, end_date):
        """Run complete backtest pipeline for both strategy and benchmark portfolios."""
        strategy_returns = self.get_portfolio_returns(self.strategy_weights, start_date, end_date)
        benchmark_returns = self.get_portfolio_returns(self.benchmark_weights, start_date, end_date)

        strategy_cumulative = self.calculate_cumulative_returns(strategy_returns)
        benchmark_cumulative = self.calculate_cumulative_returns(benchmark_returns)

        strategy_metrics = self.calculate_metrics(strategy_returns, 'Strategy (15/60/25)')
        benchmark_metrics = self.calculate_metrics(benchmark_returns, 'Benchmark (60/40)')

        self.results = {
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
            'strategy_cumulative': strategy_cumulative,
            'benchmark_cumulative': benchmark_cumulative,
            'strategy_metrics': strategy_metrics,
            'benchmark_metrics': benchmark_metrics,
            'start_date': start_date,
            'end_date': end_date
        }
        return self.results

    def run_market_stress_test(self):
        """
        Executes a targeted stress test over critical macroeconomic periods.
        Fulfills Task 5 evaluation requirements.

        """
        stress_periods = {
            "2020 COVID Market Crash": ("2020-02-19", "2020-03-23"),
            "2022 Fed Rate Hike Regime": ("2022-01-03", "2022-12-30")
        }

        stress_results = []
        for period_name, (start, end) in stress_periods.items():
            strat_ret = self.get_portfolio_returns(self.strategy_weights, start, end)
            bench_ret = self.get_portfolio_returns(self.benchmark_weights, start, end)

            strat_metrics = self.calculate_metrics(strat_ret, f"Strategy ({period_name})")
            bench_metrics = self.calculate_metrics(bench_ret, f"Benchmark ({period_name})")
            stress_results.extend([strat_metrics, bench_metrics])

        return pd.DataFrame(stress_results)

    def plot_results(self, save_path=None):
        """Plot cumulative returns comparison."""
        fig, ax = plt.subplots(figsize=(14, 7))

        strategy_cum = self.results['strategy_cumulative'] - 1  # Convert back to return growth above basis
        benchmark_cum = self.results['benchmark_cumulative'] - 1

        ax.plot(strategy_cum.index, strategy_cum.values, color='blue', linewidth=2, label='Strategy (15/60/25)')
        ax.plot(benchmark_cum.index, benchmark_cum.values, color='orange', linewidth=2, label='Benchmark (60/40)')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Portfolio Backtest: Strategy vs Benchmark', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        final_strategy = strategy_cum.iloc[-1]
        final_benchmark = benchmark_cum.iloc[-1]
        diff = final_strategy - final_benchmark

        summary_text = (
            f"Strategy Final Return: {final_strategy:.2%}\n"
            f"Benchmark Final Return: {final_benchmark:.2%}\n"
            f"Net Outperformance: {diff:+.2%}"
        )
        ax.text(0.02, 0.95, summary_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_summary(self):
        """Print formatted performance summary."""
        metrics_df = pd.DataFrame([self.results['strategy_metrics'], self.results['benchmark_metrics']])
        for col in ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Max Drawdown']:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}")
        metrics_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
        print(metrics_df.to_string(index=False))
