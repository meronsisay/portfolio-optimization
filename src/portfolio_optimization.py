"""
Portfolio Optimization Module
Modern Portfolio Theory (MPT) implementation with PyPortfolioOpt.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt import plotting
import warnings
warnings.filterwarnings("ignore")


class PortfolioOptimizer:
    """
    Builds optimal portfolios using Modern Portfolio Theory.
    
    Combines forecasted returns (TSLA) with historical returns (BND, SPY).
    """
    
    def __init__(self, returns_df, forecasted_return_tsla=None):
        """
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Historical daily returns for all assets
        forecasted_return_tsla : float
            Annualized forecasted return for TSLA (from ARIMA model)
        """
        self.returns_df = returns_df
        self.forecasted_return_tsla = forecasted_return_tsla
        
    def prepare_expected_returns(self):
        """
        Prepare expected returns vector:
        - TSLA: Use forecasted return
        - BND, SPY: Use historical average returns (annualized)
        """
        # Historical average daily returns
        hist_returns = self.returns_df.mean() * 252  # Annualized
        
        # If forecasted return provided, override TSLA
        if self.forecasted_return_tsla is not None:
            hist_returns['TSLA'] = self.forecasted_return_tsla
            print(f"✓ Using forecasted return for TSLA: {hist_returns['TSLA']:.2%}")
        else:
            print(" No forecasted return provided. Using historical for TSLA.")
        
        self.expected_returns = hist_returns
        return hist_returns
    
    def compute_covariance_matrix(self):
        """Compute annualized covariance matrix from historical returns."""
        cov_matrix = self.returns_df.cov() * 252
        self.cov_matrix = cov_matrix
        return cov_matrix
    
    def optimize_max_sharpe(self):
        """Find portfolio with maximum Sharpe ratio."""
        ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(verbose=False)
        
        return {
            'weights': cleaned_weights,
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2]
        }
    
    def optimize_min_volatility(self):
        """Find portfolio with minimum volatility."""
        ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(verbose=False)
        
        return {
            'weights': cleaned_weights,
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2]
        }
    
    def generate_efficient_frontier(self, points=50):
        """Generate efficient frontier points within strictly feasible mathematical bounds."""
        # 1. Establish the base boundary limit (Minimum Volatility node)
        ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
        ef.min_volatility()
        min_ret = ef.portfolio_performance(verbose=False)[0]
        
        # 2. Establish the top boundary limit (Maximum possible single asset target return)
        max_ret = self.expected_returns.max()
        
        # 3. Space points safely across the verified feasible return spectrum
        target_returns = np.linspace(min_ret, max_ret, points)
        frontier_vols = []
        frontier_rets = []
        
        for target_ret in target_returns:
            # We must re-instantiate the EF object for every step because they hold optimization state
            ef = EfficientFrontier(self.expected_returns, self.cov_matrix)
            try:
                ef.efficient_return(target_ret)
                perf = ef.portfolio_performance(verbose=False)
                frontier_rets.append(perf[0])
                frontier_vols.append(perf[1])
            except Exception:
                continue
        
        return pd.DataFrame({
            'volatility': frontier_vols,
            'return': frontier_rets
        })
    def plot_efficient_frontier(self, max_sharpe_port=None, min_vol_port=None):
        """Plot efficient frontier with key portfolios marked."""
        # Generate frontier
        frontier = self.generate_efficient_frontier()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot efficient frontier
        ax.plot(frontier['volatility'], frontier['return'], 
                color='blue', linewidth=2, label='Efficient Frontier')
        
        # Plot individual assets
        for asset in self.expected_returns.index:
            vol = np.sqrt(self.cov_matrix.loc[asset, asset])
            ret = self.expected_returns[asset]
            ax.scatter(vol, ret, s=100, label=asset, zorder=5)
            ax.annotate(asset, (vol, ret), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        # Mark optimal portfolios
        colors = ['red', 'green']
        labels = ['Max Sharpe Ratio', 'Min Volatility']
        portfolios = [max_sharpe_port, min_vol_port]
        
        for i, port in enumerate(portfolios):
            if port:
                ax.scatter(port['volatility'], port['expected_return'], 
                          s=150, color=colors[i], marker='*', 
                          label=labels[i], zorder=10)
                ax.annotate(labels[i], 
                           (port['volatility'], port['expected_return']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, color=colors[i], fontweight='bold')
        
        ax.set_xlabel('Volatility (Annualized)', fontsize=12)
        ax.set_ylabel('Expected Return (Annualized)', fontsize=12)
        ax.set_title('Efficient Frontier - Portfolio Optimization', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_covariance_heatmap(self):
        """Plot covariance matrix heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cov_matrix, annot=True, fmt='.4f', 
                   cmap='coolwarm', square=True, 
                   linewidths=0.5)
        plt.title('Covariance Matrix (Annualized)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def get_portfolio_summary(self, portfolio, name):
        """Print formatted portfolio summary."""
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        print(f"Weights:")
        for asset, weight in portfolio['weights'].items():
            print(f"  {asset}: {weight:.2%}")
        print(f"\nExpected Annual Return: {portfolio['expected_return']:.2%}")
        print(f"Expected Volatility:    {portfolio['volatility']:.2%}")
        print(f"Sharpe Ratio:           {portfolio['sharpe_ratio']:.3f}")
        print(f"{'='*50}")
        
        return portfolio