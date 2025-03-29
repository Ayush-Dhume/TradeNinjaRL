"""
Custom trading environment for options trading.
This is a simplified version for demonstration.
"""
import logging
import datetime
import random
import math
import json

logger = logging.getLogger(__name__)

class OptionsEnv:
    """
    A custom environment for options trading.
    """
    
    def __init__(self, data_processor, render_mode=None):
        """
        Initialize the options trading environment.
        
        Args:
            data_processor: Data processor for market data
            render_mode (str): Rendering mode (human or None)
        """
        self.data_processor = data_processor
        self.render_mode = render_mode
        
        # Environment state
        self.current_step = 0
        self.max_steps = 1000
        self.data = None
        self.current_option_idx = 0
        self.current_observation = None
        self.positions = {}  # Symbol -> {quantity, entry_price, entry_time}
        self.balance = 100000  # Starting capital
        self.initial_balance = self.balance
        self.portfolio_value = self.balance
        self.trades = []
        self.history = []
        
        # Action and observation space
        # Action: (0: do nothing, 1: buy, 2: sell)
        # Position size: (0.0 to 1.0) * balance
        self.action_space = 3
        
        # Load historical data for backtesting
        self._load_historical_data()
        
        logger.info("Initialized options trading environment")
    
    def reset(self):
        """
        Reset the environment to an initial state.
        
        Returns:
            tuple: (observation, info)
        """
        logger.info("Resetting environment")
        
        # Reset state
        self.current_step = 0
        self.positions = {}
        self.balance = self.initial_balance
        self.portfolio_value = self.balance
        self.trades = []
        self.history = []
        
        # Get initial observation
        self._next_time_step()
        observation = self._get_observation()
        
        info = {
            "balance": self.balance,
            "portfolio_value": self.portfolio_value,
            "positions": len(self.positions),
            "step": self.current_step
        }
        
        return observation, info
    
    def step(self, action):
        """
        Execute one time step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        prev_portfolio_value = self.portfolio_value
        
        # Parse action
        action_type, option_idx, position_size = self._parse_action(action)
        
        # Execute action
        reward, trade_info = self._execute_action(action_type, option_idx, position_size)
        
        # Move to next time step
        self._next_time_step()
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        terminated = (self.current_step >= self.max_steps or 
                     self.portfolio_value <= 0.1 * self.initial_balance)  # Stop if 90% is lost
        
        truncated = False
        
        # Calculate portfolio value
        self.portfolio_value = self.balance + self._calculate_position_value()
        
        # Update history
        self.history.append({
            "step": self.current_step,
            "balance": self.balance,
            "portfolio_value": self.portfolio_value,
            "positions": len(self.positions),
            "action": action_type,
            "reward": reward
        })
        
        # Save trade if executed
        if trade_info:
            self.trades.append(trade_info)
        
        # Calculate reward as portfolio change
        portfolio_change = self.portfolio_value - prev_portfolio_value
        reward = portfolio_change / self.initial_balance * 100  # Reward as percentage of initial balance
        
        info = {
            "balance": self.balance,
            "portfolio_value": self.portfolio_value,
            "positions": len(self.positions),
            "unrealized_pnl": self._calculate_unrealized_pnl(),
            "step": self.current_step,
            "trade": trade_info
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            None
        """
        if self.render_mode != "human":
            return
        
        # Print current state
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Positions: {len(self.positions)}")
        print(f"Unrealized P&L: ${self._calculate_unrealized_pnl():.2f}")
        
        # Print recent trades
        if self.trades:
            print("\nRecent Trades:")
            for trade in self.trades[-5:]:
                print(f"{trade['timestamp']} - {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
        
        print("\n" + "-" * 50 + "\n")
    
    def close(self):
        """
        Close the environment.
        
        Returns:
            None
        """
        logger.info("Closing environment")
    
    def _parse_action(self, action):
        """
        Parse the action from the action space.
        
        Args:
            action: Raw action from the agent
            
        Returns:
            tuple: (action_type, option_idx, position_size)
        """
        # In a real implementation, this would parse the action from RL model
        # For this demonstration, we'll assume a simple action structure
        
        if isinstance(action, (list, tuple)):
            # Assume action is [action_type, option_idx, position_size]
            if len(action) >= 3:
                action_type = int(action[0])
                option_idx = int(action[1])
                position_size = float(action[2])
            elif len(action) == 2:
                action_type = int(action[0])
                option_idx = int(action[1])
                position_size = 0.1  # Default
            else:
                action_type = int(action[0])
                option_idx = 0  # Default
                position_size = 0.1  # Default
        else:
            # Assume action is just the action type
            action_type = int(action)
            option_idx = 0  # Default
            position_size = 0.1  # Default
        
        # Ensure action_type is in valid range
        action_type = max(0, min(action_type, 2))
        
        # Ensure option_idx is in valid range
        if self.data is not None:
            num_options = len(self.data.get("calls", [])) + len(self.data.get("puts", []))
            option_idx = max(0, min(option_idx, num_options - 1))
        
        # Ensure position_size is in valid range
        position_size = max(0.0, min(position_size, 1.0))
        
        return action_type, option_idx, position_size
    
    def _execute_action(self, action_type, option_idx, position_size):
        """
        Execute a trading action.
        
        Args:
            action_type (int): 0 (do nothing), 1 (buy), 2 (sell)
            option_idx (int): Index of the option in the current data
            position_size (float): % of capital to allocate
            
        Returns:
            tuple: (reward, trade_info)
        """
        trade_info = None
        reward = 0
        
        # If action is do nothing, return immediately
        if action_type == 0:
            return reward, trade_info
        
        # Get current option data
        option = self._get_option_by_idx(option_idx)
        if not option:
            return reward, trade_info
        
        symbol = option.get("symbol", "UNKNOWN")
        price = option.get("last_price", 0)
        
        # Calculate trade size
        trade_value = self.balance * position_size
        quantity = int(trade_value / price) if price > 0 else 0
        
        if quantity <= 0:
            return reward, trade_info
        
        # Execute trade
        timestamp = datetime.datetime.now().isoformat()
        
        if action_type == 1:  # Buy
            # Check if we have enough balance
            cost = quantity * price
            if cost > self.balance:
                quantity = int(self.balance / price)
                cost = quantity * price
            
            if quantity <= 0:
                return reward, trade_info
            
            # Update balance
            self.balance -= cost
            
            # Add to positions
            if symbol in self.positions:
                # Average down/up existing position
                existing_qty = self.positions[symbol]["quantity"]
                existing_cost = existing_qty * self.positions[symbol]["entry_price"]
                new_cost = existing_cost + cost
                new_qty = existing_qty + quantity
                avg_price = new_cost / new_qty if new_qty > 0 else 0
                
                self.positions[symbol] = {
                    "quantity": new_qty,
                    "entry_price": avg_price,
                    "entry_time": timestamp
                }
            else:
                # Create new position
                self.positions[symbol] = {
                    "quantity": quantity,
                    "entry_price": price,
                    "entry_time": timestamp
                }
            
            trade_info = {
                "symbol": symbol,
                "action": "BUY",
                "quantity": quantity,
                "price": price,
                "cost": cost,
                "timestamp": timestamp
            }
            
            logger.info(f"Bought {quantity} {symbol} @ ${price:.2f}")
            
        elif action_type == 2:  # Sell
            # Check if we have the position
            existing_qty = self.positions.get(symbol, {}).get("quantity", 0)
            
            if existing_qty > 0:
                # Sell existing position
                sell_qty = min(existing_qty, quantity)
                proceeds = sell_qty * price
                
                # Update balance
                self.balance += proceeds
                
                # Update position
                if sell_qty >= existing_qty:
                    # Close position completely
                    entry_price = self.positions[symbol]["entry_price"]
                    profit = proceeds - (sell_qty * entry_price)
                    
                    # Apply reward for profitable trade
                    reward += profit / self.initial_balance * 100
                    
                    del self.positions[symbol]
                else:
                    # Reduce position
                    self.positions[symbol]["quantity"] -= sell_qty
                
                trade_info = {
                    "symbol": symbol,
                    "action": "SELL",
                    "quantity": sell_qty,
                    "price": price,
                    "proceeds": proceeds,
                    "timestamp": timestamp
                }
                
                logger.info(f"Sold {sell_qty} {symbol} @ ${price:.2f}")
            else:
                # Short selling (not implemented in this simplified version)
                pass
        
        return reward, trade_info
    
    def _next_time_step(self):
        """
        Move to the next time step and update market data.
        
        Returns:
            None
        """
        self.current_step += 1
        
        # If we have historical data loaded, use it
        if hasattr(self, 'historical_data') and self.historical_data:
            if self.current_step < len(self.historical_data):
                self.data = self.historical_data[self.current_step]
            else:
                # End of historical data, generate random
                self._generate_synthetic_data()
        else:
            # No historical data, generate synthetic
            self._generate_synthetic_data()
        
        # Update current observation
        self.current_observation = self._get_observation()
    
    def _get_observation(self):
        """
        Get the current environment observation.
        
        Returns:
            dict: Current observation
        """
        if not self.data:
            return {
                "market_data": {},
                "portfolio": {
                    "balance": self.balance,
                    "portfolio_value": self.portfolio_value,
                    "positions": {}
                }
            }
        
        # Construct observation with market data and portfolio state
        observation = {
            "market_data": {
                "index": self.data.get("index", ""),
                "spot_price": self.data.get("spot_price", 0),
                "timestamp": self.data.get("timestamp", ""),
                "calls": self.data.get("calls", []),
                "puts": self.data.get("puts", [])
            },
            "portfolio": {
                "balance": self.balance,
                "portfolio_value": self.portfolio_value,
                "positions": self.positions
            }
        }
        
        return observation
    
    def _get_option_by_idx(self, option_idx):
        """
        Get option data by index.
        
        Args:
            option_idx (int): Index of the option
            
        Returns:
            dict: Option data or None
        """
        if not self.data:
            return None
        
        calls = self.data.get("calls", [])
        puts = self.data.get("puts", [])
        
        all_options = calls + puts
        
        if 0 <= option_idx < len(all_options):
            return all_options[option_idx]
        
        return None
    
    def _calculate_position_value(self):
        """
        Calculate the total value of all open positions.
        
        Returns:
            float: Total position value
        """
        total_value = 0
        
        if not self.data:
            return total_value
        
        calls = self.data.get("calls", [])
        puts = self.data.get("puts", [])
        
        all_options = calls + puts
        
        for symbol, position in self.positions.items():
            # Find the current price of this option
            option_price = 0
            for option in all_options:
                if option.get("symbol", "") == symbol:
                    option_price = option.get("last_price", 0)
                    break
            
            # Calculate value
            position_value = position["quantity"] * option_price
            total_value += position_value
        
        return total_value
    
    def _calculate_unrealized_pnl(self):
        """
        Calculate unrealized profit/loss for all positions.
        
        Returns:
            float: Total unrealized P&L
        """
        total_pnl = 0
        
        if not self.data:
            return total_pnl
        
        calls = self.data.get("calls", [])
        puts = self.data.get("puts", [])
        
        all_options = calls + puts
        
        for symbol, position in self.positions.items():
            # Find the current price of this option
            current_price = 0
            for option in all_options:
                if option.get("symbol", "") == symbol:
                    current_price = option.get("last_price", 0)
                    break
            
            # Calculate unrealized P&L
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            pnl = quantity * (current_price - entry_price)
            total_pnl += pnl
        
        return total_pnl
    
    def _load_historical_data(self):
        """
        Load historical data for backtesting.
        
        Returns:
            None
        """
        # In a real implementation, this would load historical data
        # For this demonstration, we'll generate synthetic data
        self.historical_data = []
        
        # Generate data for 100 steps
        for i in range(100):
            self.historical_data.append(self._generate_synthetic_data(i))
        
        logger.info(f"Loaded {len(self.historical_data)} historical data points")
    
    def _generate_synthetic_data(self, step_idx=None):
        """
        Generate synthetic data for testing.
        
        Args:
            step_idx (int): Step index for deterministic generation
            
        Returns:
            dict: Synthetic option chain data
        """
        # If step_idx is provided, use it for deterministic generation
        if step_idx is None:
            step_idx = self.current_step
        
        # Generate a base spot price with trending behavior
        trend = math.sin(step_idx / 20) * 1000  # Oscillating trend
        random_variation = random.uniform(-200, 200)  # Random noise
        spot_price = 22000 + trend + random_variation
        
        # Generate strike prices around the spot price
        atm_strike = round(spot_price / 50) * 50  # Round to nearest 50
        strikes = [atm_strike + (i * 50) for i in range(-10, 11)]
        
        # Generate expiry dates
        expiry_days = [7, 14, 30, 60, 90]
        current_date = datetime.datetime.now()
        expiry_dates = [(current_date + datetime.timedelta(days=d)).strftime("%Y-%m-%d") for d in expiry_days]
        
        # Select an expiry date
        expiry_date = expiry_dates[min(len(expiry_dates) - 1, step_idx // 20)]
        
        # Days to expiry
        days_to_expiry = (datetime.datetime.strptime(expiry_date, "%Y-%m-%d") - current_date).days
        days_to_expiry = max(1, days_to_expiry)
        
        # Generate call options
        calls = []
        for strike in strikes:
            # Calculate implied volatility based on distance from spot
            distance_from_spot = abs(strike - spot_price) / spot_price
            base_iv = 0.20 + (distance_from_spot * 0.8)  # IV increases with distance from spot
            implied_volatility = base_iv + random.uniform(-0.05, 0.05)  # Add some randomness
            
            # Calculate synthetic option price
            intrinsic_value = max(0, spot_price - strike)
            time_value = spot_price * implied_volatility * (days_to_expiry / 365) ** 0.5
            option_price = intrinsic_value + time_value * 0.8  # Apply a discount to theoretical time value
            
            # Add some randomness to price
            option_price = max(0.1, option_price * (1 + random.uniform(-0.02, 0.02)))
            
            # Calculate synthetic greeks
            delta = 0.5 + 0.5 * (1 - distance_from_spot) if strike <= spot_price else 0.5 * (1 - distance_from_spot)
            gamma = 0.08 / (1 + 10 * distance_from_spot)
            theta = -option_price * 0.1 / days_to_expiry
            vega = option_price * days_to_expiry / 365
            
            # Generate symbol
            symbol = f"NIFTY_{expiry_date}_{strike}_CALL"
            
            calls.append({
                "symbol": symbol,
                "strike_price": strike,
                "option_type": "CALL",
                "expiry_date": expiry_date,
                "last_price": round(option_price, 2),
                "volume": int(random.uniform(1000, 5000)),
                "open_interest": int(random.uniform(5000, 20000)),
                "implied_volatility": round(implied_volatility, 4),
                "delta": round(delta, 4),
                "gamma": round(gamma, 4),
                "theta": round(theta, 4),
                "vega": round(vega, 4),
                "days_to_expiry": days_to_expiry
            })
        
        # Generate put options
        puts = []
        for strike in strikes:
            # Calculate implied volatility based on distance from spot
            distance_from_spot = abs(strike - spot_price) / spot_price
            base_iv = 0.20 + (distance_from_spot * 0.8)  # IV increases with distance from spot
            implied_volatility = base_iv + random.uniform(-0.05, 0.05)  # Add some randomness
            
            # Calculate synthetic option price
            intrinsic_value = max(0, strike - spot_price)
            time_value = spot_price * implied_volatility * (days_to_expiry / 365) ** 0.5
            option_price = intrinsic_value + time_value * 0.8  # Apply a discount to theoretical time value
            
            # Add some randomness to price
            option_price = max(0.1, option_price * (1 + random.uniform(-0.02, 0.02)))
            
            # Calculate synthetic greeks
            delta = -0.5 - 0.5 * (1 - distance_from_spot) if strike >= spot_price else -0.5 * (1 - distance_from_spot)
            gamma = 0.08 / (1 + 10 * distance_from_spot)
            theta = -option_price * 0.1 / days_to_expiry
            vega = option_price * days_to_expiry / 365
            
            # Generate symbol
            symbol = f"NIFTY_{expiry_date}_{strike}_PUT"
            
            puts.append({
                "symbol": symbol,
                "strike_price": strike,
                "option_type": "PUT",
                "expiry_date": expiry_date,
                "last_price": round(option_price, 2),
                "volume": int(random.uniform(1000, 5000)),
                "open_interest": int(random.uniform(5000, 20000)),
                "implied_volatility": round(implied_volatility, 4),
                "delta": round(delta, 4),
                "gamma": round(gamma, 4),
                "theta": round(theta, 4),
                "vega": round(vega, 4),
                "days_to_expiry": days_to_expiry
            })
        
        # Combine into a single data structure
        data = {
            "index": "NIFTY 50",
            "spot_price": round(spot_price, 2),
            "expiry_date": expiry_date,
            "timestamp": datetime.datetime.now().isoformat(),
            "calls": calls,
            "puts": puts
        }
        
        # Store the data
        self.data = data
        
        return data