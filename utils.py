"""
Utility functions for options trading RL system.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

logger = logging.getLogger(__name__)

def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('options_trading_rl.log')
        ]
    )
    return logging.getLogger()

def calculate_option_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculate option price using Black-Scholes model.
    
    Args:
        S (float): Spot price
        K (float): Strike price
        T (float): Time to expiry (in years)
        r (float): Risk-free interest rate
        sigma (float): Implied volatility
        option_type (str): "call" or "put"
        
    Returns:
        float: Option price
    """
    # Ensure time to expiry is positive
    T = max(T, 0.00001)
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return max(price, 0.01)  # Ensure price is positive

def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate option greeks using Black-Scholes model.
    
    Args:
        S (float): Spot price
        K (float): Strike price
        T (float): Time to expiry (in years)
        r (float): Risk-free interest rate
        sigma (float): Implied volatility
        option_type (str): "call" or "put"
        
    Returns:
        dict: Option greeks
    """
    # Ensure time to expiry is positive
    T = max(T, 0.00001)
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate delta
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
    else:  # put
        delta = norm.cdf(d1) - 1
    
    # Calculate gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Calculate theta
    if option_type.lower() == "call":
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    # Calculate vega (same for calls and puts)
    vega = S * np.sqrt(T) * norm.pdf(d1)
    
    # Calculate rho
    if option_type.lower() == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365,  # Convert to daily theta
        "vega": vega / 100,    # Convert to 1% change in IV
        "rho": rho / 100       # Convert to 1% change in interest rate
    }

def calculate_implied_volatility(option_price, S, K, T, r, option_type="call", precision=0.00001, max_iterations=100):
    """
    Calculate implied volatility using binary search method.
    
    Args:
        option_price (float): Market price of the option
        S (float): Spot price
        K (float): Strike price
        T (float): Time to expiry (in years)
        r (float): Risk-free interest rate
        option_type (str): "call" or "put"
        precision (float): Desired precision for IV
        max_iterations (int): Maximum number of iterations
        
    Returns:
        float: Implied volatility
    """
    # Initial bounds for IV search
    sigma_low = 0.001
    sigma_high = 5.0
    
    for i in range(max_iterations):
        sigma_mid = (sigma_low + sigma_high) / 2
        price = calculate_option_price(S, K, T, r, sigma_mid, option_type)
        
        if abs(price - option_price) < precision:
            return sigma_mid
        
        if price > option_price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    
    return (sigma_low + sigma_high) / 2

def plot_option_chain(option_chain_df, spot_price, filename=None):
    """
    Plot option chain data visualization.
    
    Args:
        option_chain_df (pd.DataFrame): Option chain dataframe
        spot_price (float): Current spot price
        filename (str): Filename to save plot, or None for display
        
    Returns:
        None
    """
    try:
        # Set styling
        sns.set(style="darkgrid")
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Split into calls and puts
        calls = option_chain_df[option_chain_df['option_type'] == 'CALL']
        puts = option_chain_df[option_chain_df['option_type'] == 'PUT']
        
        # Plot call prices
        if not calls.empty:
            plt.plot(calls['strike_price'], calls['last_price'], 'b-o', label='Call Prices')
        
        # Plot put prices
        if not puts.empty:
            plt.plot(puts['strike_price'], puts['last_price'], 'r-o', label='Put Prices')
        
        # Add vertical line for spot price
        plt.axvline(x=spot_price, color='green', linestyle='--', label=f'Spot Price ({spot_price})')
        
        # Add title and labels
        plt.title('Option Chain Prices', fontsize=16)
        plt.xlabel('Strike Price', fontsize=12)
        plt.ylabel('Option Price', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        # Save or display
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting option chain: {e}")

def plot_learning_curve(episode_rewards, moving_avg_window=10, filename=None):
    """
    Plot learning curve from training.
    
    Args:
        episode_rewards (list): List of episode rewards
        moving_avg_window (int): Window size for moving average
        filename (str): Filename to save plot, or None for display
        
    Returns:
        None
    """
    try:
        # Calculate moving average
        moving_avg = []
        for i in range(len(episode_rewards)):
            if i < moving_avg_window:
                moving_avg.append(np.mean(episode_rewards[:i+1]))
            else:
                moving_avg.append(np.mean(episode_rewards[i-moving_avg_window+1:i+1]))
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot episode rewards
        plt.plot(episode_rewards, 'b-', alpha=0.3, label='Episode Rewards')
        
        # Plot moving average
        plt.plot(moving_avg, 'r-', label=f'{moving_avg_window}-Episode Moving Average')
        
        # Add title and labels
        plt.title('Training Learning Curve', fontsize=16)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        # Save or display
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting learning curve: {e}")

def plot_position_analysis(trades_df, position_name, filename=None):
    """
    Plot detailed analysis of a specific trading position.
    
    Args:
        trades_df (pd.DataFrame): Dataframe of trades
        position_name (str): Name of the position to analyze
        filename (str): Filename to save plot, or None for display
        
    Returns:
        None
    """
    try:
        # Filter trades for the specified position
        position_trades = trades_df[trades_df['symbol'] == position_name]
        
        if position_trades.empty:
            logger.warning(f"No trades found for position: {position_name}")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot price and position over time
        position_trades.plot(x='timestamp', y='price', ax=ax1, marker='o', linestyle='-', color='blue')
        
        # Add buy/sell markers
        buys = position_trades[position_trades['action'] == 'BUY']
        sells = position_trades[position_trades['action'] == 'SELL']
        
        ax1.scatter(buys['timestamp'], buys['price'], color='green', marker='^', s=100, label='Buy')
        ax1.scatter(sells['timestamp'], sells['price'], color='red', marker='v', s=100, label='Sell')
        
        ax1.set_title(f'Price Action and Trades for {position_name}', fontsize=16)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend()
        ax1.grid(True)
        
        # Plot position size over time
        position_size = []
        current_size = 0
        
        for _, row in position_trades.iterrows():
            if row['action'] == 'BUY':
                current_size += row['quantity']
            else:  # SELL
                current_size -= row['quantity']
            position_size.append(current_size)
        
        position_trades['position_size'] = position_size
        position_trades.plot(x='timestamp', y='position_size', ax=ax2, marker='o', linestyle='-', color='purple')
        
        ax2.set_title('Position Size Over Time', fontsize=16)
        ax2.set_ylabel('Contracts', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='--')
        ax2.grid(True)
        
        # Save or display
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting position analysis: {e}")

def calculate_kelly_criterion(win_rate, win_loss_ratio):
    """
    Calculate Kelly Criterion for optimal position sizing.
    
    Args:
        win_rate (float): Probability of winning (0-1)
        win_loss_ratio (float): Average win / Average loss
        
    Returns:
        float: Kelly percentage
    """
    if win_rate <= 0 or win_loss_ratio <= 0:
        return 0
    
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Constrain Kelly to reasonable range
    kelly = max(0, min(kelly, 0.5))
    
    return kelly

def get_market_hours(market="NSE"):
    """
    Get market trading hours.
    
    Args:
        market (str): Market name
        
    Returns:
        tuple: (market_open, market_close) datetime objects for today
    """
    today = datetime.now().date()
    
    if market == "NSE":
        # NSE trading hours: 9:15 AM to 3:30 PM IST
        market_open = datetime.combine(today, datetime.strptime("09:15:00", "%H:%M:%S").time())
        market_close = datetime.combine(today, datetime.strptime("15:30:00", "%H:%M:%S").time())
    else:
        # Default trading hours
        market_open = datetime.combine(today, datetime.strptime("09:00:00", "%H:%M:%S").time())
        market_close = datetime.combine(today, datetime.strptime("17:00:00", "%H:%M:%S").time())
    
    return market_open, market_close

def is_market_open(market="NSE"):
    """
    Check if market is currently open.
    
    Args:
        market (str): Market name
        
    Returns:
        bool: True if market is open
    """
    now = datetime.now()
    market_open, market_close = get_market_hours(market)
    
    # Check if today is a weekday (0=Monday, 6=Sunday)
    is_weekday = now.weekday() < 5
    
    # Check if current time is within market hours
    is_within_hours = market_open <= now <= market_close
    
    return is_weekday and is_within_hours

def load_config(config_file="config.json"):
    """
    Load configuration from a JSON file.
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        dict: Configuration parameters
    """
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            return config_data
        else:
            logger.warning(f"Config file {config_file} not found. Using defaults.")
            return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}

def save_config(config_data, config_file="config.json"):
    """
    Save configuration to a JSON file.
    
    Args:
        config_data (dict): Configuration parameters
        config_file (str): Path to configuration file
        
    Returns:
        bool: True if successful
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving config file: {e}")
        return False
