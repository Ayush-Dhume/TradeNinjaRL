"""
Configuration settings for the RL options trading system.
"""

# API Configuration
API_RETRY_ATTEMPTS = 3
API_TIMEOUT = 30  # seconds

# Data Collection
DATA_COLLECTION_INTERVAL = 60  # seconds
ERROR_RETRY_INTERVAL = 30  # seconds
HISTORICAL_DATA_DAYS = 90  # days of historical data to fetch
OPTION_CHAIN_DEPTH = 10  # Number of strikes above and below current price

# Trading Environment
MAX_POSITION_SIZE = 5  # Maximum number of contracts
MAX_TRADE_VALUE = 100000  # Maximum trade value in INR
TRADING_FEES = 0.0005  # 0.05% of trade value
TRANSACTION_FEES = 0.0005  # 0.05% for STT, exchange fees, etc.
MAX_STEPS_PER_EPISODE = 100  # Maximum number of time steps in one episode
MAX_SLIPPAGE = 0.005  # Maximum slippage percentage
TIME_STEP_SIZE = 5  # minutes between steps

# Reinforcement Learning
LEARNING_RATE = 0.0003
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TRAINING_EPOCHS = 10
N_STEPS = 2048
ENT_COEF = 0.01
PPO_STEPS = 2048
PPO_BATCH_SIZE = 64
MODEL_SAVE_FREQUENCY = 10  # Save model every 10 episodes

# Features
FEATURES = [
    "spot_price", "strike_price", "time_to_expiry", "interest_rate",
    "implied_volatility", "delta", "gamma", "theta", "vega", "rho",
    "open", "high", "low", "close", "volume", "open_interest"
]

# Indices
AVAILABLE_INDICES = ["NIFTY", "SENSEX", "BANKNIFTY"]

# Web Dashboard
CHART_UPDATE_INTERVAL = 10  # seconds
DASHBOARD_REFRESH_RATE = 10  # seconds
