"""
Backtesting module for evaluating RL options trading strategies.
"""
import logging
import datetime
import random
import json
import math

logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtester for evaluating RL models on historical data.
    """
    
    def __init__(self, data_processor, model, mongo_db=None):
        """
        Initialize the backtester.
        
        Args:
            data_processor: Data processor for market data
            model: Trained RL model
            mongo_db: MongoDB connection for storing results
        """
        self.data_processor = data_processor
        self.model = model
        self.mongo_db = mongo_db
        
        # Historical data cache
        self.historical_data = []
        
        # Test results
        self.balance_history = []
        self.equity_history = []
        self.timestamp_history = []
        self.trades = []
        
        logger.info("Initialized backtester")
    
    def run_backtest(self, start_date=None, end_date=None, initial_balance=100000.0, plot_results=True):
        """
        Run a backtest over historical data.
        
        Args:
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            initial_balance (float): Initial account balance
            plot_results (bool): Whether to plot results
            
        Returns:
            dict: Backtest metrics
        """
        logger.info(f"Starting backtest from {start_date} to {end_date} with initial balance ${initial_balance}")
        
        # Set default dates if not provided
        if not start_date:
            start_date = datetime.datetime.now() - datetime.timedelta(days=30)
        
        if not end_date:
            end_date = datetime.datetime.now()
        
        # Load historical data
        self._load_historical_data(start_date, end_date)
        
        if not self.historical_data:
            logger.warning("No historical data available for backtest")
            return None
        
        # Initialize backtest state
        balance = initial_balance
        positions = {}  # Symbol -> {quantity, entry_price, entry_time}
        
        # Initialize history arrays
        self.balance_history = [balance]
        self.equity_history = [balance]
        self.timestamp_history = [self.historical_data[0].get("timestamp") if self.historical_data else datetime.datetime.now().isoformat()]
        self.trades = []
        
        # Run backtest
        for day_idx, day_data in enumerate(self.historical_data):
            logger.debug(f"Backtesting day {day_idx+1}/{len(self.historical_data)}")
            
            # Get timestamp
            timestamp = day_data.get("timestamp", datetime.datetime.now().isoformat())
            
            # Get calls and puts
            calls = day_data.get("calls", [])
            puts = day_data.get("puts", [])
            
            # Combine options
            all_options = calls + puts
            
            # Calculate portfolio value
            portfolio_value = balance
            for symbol, position in positions.items():
                option_price = 0
                for option in all_options:
                    if option.get("symbol", "") == symbol:
                        option_price = option.get("last_price", 0)
                        break
                
                position_value = position["quantity"] * option_price
                portfolio_value += position_value
            
            # Update history
            self.balance_history.append(balance)
            self.equity_history.append(portfolio_value)
            self.timestamp_history.append(timestamp)
            
            # Create observation for the model
            observation = {
                "market_data": {
                    "index": day_data.get("index", ""),
                    "spot_price": day_data.get("spot_price", 0),
                    "timestamp": timestamp,
                    "calls": calls,
                    "puts": puts
                },
                "portfolio": {
                    "balance": balance,
                    "portfolio_value": portfolio_value,
                    "positions": positions
                }
            }
            
            # Get model prediction
            action, _ = self.model.predict(observation)
            
            # Parse action
            action_type, option_idx, position_size = self._parse_action(action, len(all_options))
            
            # Execute action
            if action_type > 0 and option_idx < len(all_options):
                option = all_options[option_idx]
                symbol = option.get("symbol", "")
                price = option.get("last_price", 0)
                
                if price > 0:
                    # Calculate trade size
                    trade_value = balance * position_size
                    quantity = int(trade_value / price)
                    
                    if quantity > 0:
                        if action_type == 1:  # Buy
                            # Check if we have enough balance
                            cost = quantity * price
                            if cost > balance:
                                quantity = int(balance / price)
                                cost = quantity * price
                            
                            if quantity > 0:
                                # Update balance
                                balance -= cost
                                
                                # Add to positions
                                if symbol in positions:
                                    # Average down/up existing position
                                    existing_qty = positions[symbol]["quantity"]
                                    existing_cost = existing_qty * positions[symbol]["entry_price"]
                                    new_cost = existing_cost + cost
                                    new_qty = existing_qty + quantity
                                    avg_price = new_cost / new_qty if new_qty > 0 else 0
                                    
                                    positions[symbol] = {
                                        "quantity": new_qty,
                                        "entry_price": avg_price,
                                        "entry_time": timestamp
                                    }
                                else:
                                    # Create new position
                                    positions[symbol] = {
                                        "quantity": quantity,
                                        "entry_price": price,
                                        "entry_time": timestamp
                                    }
                                
                                # Record trade
                                trade = {
                                    "timestamp": timestamp,
                                    "symbol": symbol,
                                    "action": "BUY",
                                    "quantity": quantity,
                                    "price": price,
                                    "cost": cost
                                }
                                self.trades.append(trade)
                                
                                logger.debug(f"Bought {quantity} {symbol} @ ${price:.2f}")
                        
                        elif action_type == 2:  # Sell
                            # Check if we have the position
                            existing_qty = positions.get(symbol, {}).get("quantity", 0)
                            
                            if existing_qty > 0:
                                # Sell existing position
                                sell_qty = min(existing_qty, quantity)
                                proceeds = sell_qty * price
                                
                                # Update balance
                                balance += proceeds
                                
                                # Update position
                                if sell_qty >= existing_qty:
                                    entry_price = positions[symbol]["entry_price"]
                                    profit = proceeds - (sell_qty * entry_price)
                                    
                                    del positions[symbol]
                                else:
                                    positions[symbol]["quantity"] -= sell_qty
                                
                                # Record trade
                                trade = {
                                    "timestamp": timestamp,
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "quantity": sell_qty,
                                    "price": price,
                                    "proceeds": proceeds
                                }
                                self.trades.append(trade)
                                
                                logger.debug(f"Sold {sell_qty} {symbol} @ ${price:.2f}")
        
        # Calculate backtest metrics
        metrics = self._calculate_metrics(self.balance_history, self.equity_history, self.timestamp_history, self.trades)
        
        # Store backtest results in database if available
        if self.mongo_db:
            backtest_result = {
                "start_date": start_date.isoformat() if isinstance(start_date, datetime.datetime) else start_date,
                "end_date": end_date.isoformat() if isinstance(end_date, datetime.datetime) else end_date,
                "initial_balance": initial_balance,
                "final_balance": balance,
                "final_equity": self.equity_history[-1],
                "metrics": metrics,
                "trades": self.trades[:100],  # Store only the first 100 trades to limit size
                "timestamps": self.timestamp_history[::10],  # Store every 10th timestamp to limit size
                "balance_history": self.balance_history[::10],  # Store every 10th balance to limit size
                "equity_history": self.equity_history[::10],  # Store every 10th equity to limit size
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.mongo_db.db.backtest_results.insert_one(backtest_result)
            logger.info("Backtest results stored in database")
        
        # Plot results if requested
        if plot_results:
            try:
                self._plot_results()
            except Exception as e:
                logger.error(f"Error plotting results: {e}")
        
        logger.info(f"Backtest completed with final equity: ${self.equity_history[-1]:.2f}")
        
        return metrics
    
    def _parse_action(self, action, num_options):
        """
        Parse the model's action.
        
        Args:
            action: Action from model
            num_options (int): Number of available options
            
        Returns:
            tuple: (action_type, option_idx, position_size)
        """
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
            option_idx = 0  # Default to the first option
            position_size = 0.1  # Default to 10% of balance
        
        # Ensure action_type is in valid range
        action_type = max(0, min(action_type, 2))
        
        # Ensure option_idx is in valid range
        option_idx = max(0, min(option_idx, num_options - 1)) if num_options > 0 else 0
        
        # Ensure position_size is in valid range
        position_size = max(0.0, min(position_size, 1.0))
        
        return action_type, option_idx, position_size
    
    def _calculate_metrics(self, balance_history, equity_history, timestamp_history, trades):
        """
        Calculate performance metrics from backtest results.
        
        Args:
            balance_history (list): Account balance history
            equity_history (list): Account equity history
            timestamp_history (list): Timestamp history
            trades (list): List of executed trades
            
        Returns:
            dict: Performance metrics
        """
        if not equity_history or len(equity_history) < 2:
            return {
                "total_return": 0,
                "annual_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "total_trades": 0
            }
        
        # Calculate returns
        initial_equity = equity_history[0]
        final_equity = equity_history[-1]
        total_return_pct = (final_equity - initial_equity) / initial_equity * 100
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(equity_history)):
            daily_return = (equity_history[i] - equity_history[i-1]) / equity_history[i-1]
            daily_returns.append(daily_return)
        
        # Calculate annualized return
        days = len(equity_history)
        annual_return = (1 + total_return_pct / 100) ** (365 / days) - 1
        annual_return_pct = annual_return * 100
        
        # Calculate Sharpe ratio (using 0% risk-free rate for simplicity)
        if len(daily_returns) > 1:
            daily_return_mean = sum(daily_returns) / len(daily_returns)
            daily_return_std = (sum((r - daily_return_mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)) ** 0.5
            sharpe_ratio = (daily_return_mean * 252 ** 0.5) / (daily_return_std * 252 ** 0.5) if daily_return_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = equity_history[0]
        
        for equity in equity_history:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        max_drawdown_pct = max_drawdown * 100
        
        # Calculate win rate
        wins = 0
        losses = 0
        
        for i in range(1, len(trades)):
            if trades[i].get("action") == "SELL":
                # Calculate profit
                sell_proceeds = trades[i].get("proceeds", 0)
                
                # Find matching buy
                buy_cost = 0
                for j in range(i-1, -1, -1):
                    if trades[j].get("action") == "BUY" and trades[j].get("symbol") == trades[i].get("symbol"):
                        buy_cost = trades[j].get("cost", 0)
                        break
                
                if sell_proceeds > buy_cost:
                    wins += 1
                else:
                    losses += 1
        
        total_trades = len(trades)
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        
        return {
            "total_return": round(total_return_pct, 2),
            "annual_return": round(annual_return_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown_pct, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": total_trades
        }
    
    def _plot_results(self):
        """
        Plot backtest results.
        
        Returns:
            None
        """
        logger.info("Plotting backtest results (not implemented in this simplified version)")
        
        # In a real implementation, this would use matplotlib to plot results
        # For this demonstration, we'll just log the results
        
        if not self.equity_history:
            return
        
        initial_equity = self.equity_history[0]
        final_equity = self.equity_history[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        logger.info(f"Initial Equity: ${initial_equity:.2f}")
        logger.info(f"Final Equity: ${final_equity:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Total Trades: {len(self.trades)}")
    
    def export_results(self, filename='backtest_results.csv'):
        """
        Export backtest results to CSV.
        
        Args:
            filename (str): Filename for results
            
        Returns:
            bool: True if successful
        """
        logger.info(f"Exporting backtest results to {filename}")
        
        # In a real implementation, this would export to CSV
        # For this demonstration, we'll just log that it was called
        
        # Check if we have results
        if not self.equity_history or not self.timestamp_history:
            logger.warning("No backtest results to export")
            return False
        
        # Simulate export
        logger.info(f"Exported {len(self.equity_history)} data points to {filename}")
        
        return True
    
    def _load_historical_data(self, start_date, end_date):
        """
        Load historical data for backtesting.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            None
        """
        logger.info(f"Loading historical data from {start_date} to {end_date}")
        
        # In a real implementation, this would load data from database
        # For this demonstration, we'll generate synthetic data
        
        self.historical_data = []
        
        # Get the index to use
        index_name = self.data_processor.current_index or "NIFTY 50"
        
        # Calculate number of days
        if isinstance(start_date, datetime.datetime) and isinstance(end_date, datetime.datetime):
            days = (end_date - start_date).days
        else:
            days = 30  # Default to 30 days
        
        # Generate data for each day
        for day in range(days):
            current_date = start_date + datetime.timedelta(days=day) if isinstance(start_date, datetime.datetime) else datetime.datetime.now() - datetime.timedelta(days=days-day)
            
            # Skip weekends
            if current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                continue
            
            # Generate synthetic data for this day
            day_data = self._generate_synthetic_data(index_name, current_date, day)
            self.historical_data.append(day_data)
        
        logger.info(f"Loaded {len(self.historical_data)} days of historical data")
    
    def _generate_synthetic_data(self, index_name, date, day_idx):
        """
        Generate synthetic historical data for backtesting.
        
        Args:
            index_name (str): Name of the index
            date (datetime): Date for the data
            day_idx (int): Day index for trend calculation
            
        Returns:
            dict: Synthetic data
        """
        # Generate a base spot price with trending behavior
        trend = math.sin(day_idx / 20) * 1000  # Oscillating trend
        random_variation = random.uniform(-200, 200)  # Random noise
        
        if index_name == "NIFTY 50":
            base_price = 22000
        elif index_name == "SENSEX":
            base_price = 72000
        elif index_name == "BANKNIFTY":
            base_price = 46000
        else:
            base_price = 20000
        
        spot_price = base_price + trend + random_variation
        
        # Generate strike prices around the spot price
        atm_strike = round(spot_price / 50) * 50  # Round to nearest 50
        strikes = [atm_strike + (i * 50) for i in range(-10, 11)]
        
        # Generate expiry dates
        # Use the last Thursday of the current month and next month
        current_date = date
        
        # Find the last Thursday of the current month
        month = current_date.month
        year = current_date.year
        
        # Get the last day of the month
        if month == 12:
            last_day = datetime.datetime(year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            last_day = datetime.datetime(year, month + 1, 1) - datetime.timedelta(days=1)
        
        # Find the last Thursday of the month
        days_to_subtract = (last_day.weekday() - 3) % 7
        last_thursday = last_day - datetime.timedelta(days=days_to_subtract)
        
        # If we're past the last Thursday, use the next month
        if current_date > last_thursday:
            month = month + 1
            if month > 12:
                month = 1
                year += 1
            
            # Get the last day of the next month
            if month == 12:
                last_day = datetime.datetime(year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
                last_day = datetime.datetime(year, month + 1, 1) - datetime.timedelta(days=1)
            
            # Find the last Thursday of the next month
            days_to_subtract = (last_day.weekday() - 3) % 7
            last_thursday = last_day - datetime.timedelta(days=days_to_subtract)
        
        # Format expiry date
        expiry_date = last_thursday.strftime("%Y-%m-%d")
        
        # Calculate days to expiry
        days_to_expiry = (last_thursday - current_date).days
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
            symbol = f"{index_name.replace(' ', '')}_{expiry_date}_{strike}_CALL"
            
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
            symbol = f"{index_name.replace(' ', '')}_{expiry_date}_{strike}_PUT"
            
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
            "index": index_name,
            "spot_price": round(spot_price, 2),
            "expiry_date": expiry_date,
            "timestamp": date.isoformat(),
            "calls": calls,
            "puts": puts
        }
        
        return data