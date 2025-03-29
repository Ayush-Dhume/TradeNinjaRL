"""
Data processing module for options trading data.
Handles data collection, feature extraction, and preprocessing.
"""
import logging
import datetime
import json
import os
import random

logger = logging.getLogger(__name__)

class DataProcessor:
    
    def __init__(self, upstox_api, mongo_db=None):
        self.upstox_api = upstox_api
        self.mongo_db = mongo_db
        self.current_index = "NIFTY 50"
        self.current_expiry = None
        
        # Cache for processed data
        self._processed_data_cache = None
        self._last_update_time = None
    
    def collect_live_data(self):
        logger.info(f"Collecting live data for {self.current_index}")
        
        try:
            # Get spot price
            index_quote = self.upstox_api.get_index_quote(self.current_index)
            spot_price = index_quote.get("last_price", 0)
            
            # Get option chain
            option_chain = self.upstox_api.get_option_chain(self.current_index, self.current_expiry)
            
            # Process option chain
            processed_data = self._process_option_chain(option_chain, spot_price)
            
            # Store data if MongoDB is available
            if self.mongo_db:
                self._store_data(processed_data)
            
            # Update cache
            self._processed_data_cache = processed_data
            self._last_update_time = datetime.datetime.now()
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error collecting live data: {e}")
            return None
    
    def _process_option_chain(self, option_chain, spot_price):
        """
        Process option chain data and extract features.
        
        Args:
            option_chain (dict): Raw option chain data from Upstox API
            spot_price (float): Current spot price of the index
            
        Returns:
            dict: Processed options data with features
        """
        current_time = datetime.datetime.now()
        
        calls = option_chain.get("calls", [])
        puts = option_chain.get("puts", [])
        
        # Process call options
        processed_calls = []
        for call in calls:
            processed_call = self._extract_option_features(call, spot_price, "CALL", current_time)
            processed_calls.append(processed_call)
        
        # Process put options
        processed_puts = []
        for put in puts:
            processed_put = self._extract_option_features(put, spot_price, "PUT", current_time)
            processed_puts.append(processed_put)
        
        # Combine and add technical indicators
        combined_data = {
            "index": self.current_index,
            "spot_price": spot_price,
            "expiry_date": self.current_expiry,
            "timestamp": current_time.isoformat(),
            "calls": processed_calls,
            "puts": processed_puts
        }
        
        return combined_data
    
    def _extract_option_features(self, option, spot_price, option_type, current_time):
        """
        Extract features from a single option contract.
        
        Args:
            option (dict): Option contract data
            spot_price (float): Current spot price of the index
            option_type (str): 'CALL' or 'PUT'
            current_time (datetime): Current timestamp
            
        Returns:
            dict: Extracted features for the option
        """
        # Basic option data
        strike_price = option.get("strike_price", 0)
        last_price = option.get("last_price", 0)
        volume = option.get("volume", 0)
        open_interest = option.get("open_interest", 0)
        
        # Calculate moneyness
        moneyness = (spot_price - strike_price) / spot_price if spot_price > 0 else 0
        if option_type == "PUT":
            moneyness = -moneyness
        
        # Calculate time to expiry in days
        days_to_expiry = 30  # Default
        if self.current_expiry:
            try:
                expiry_date = datetime.datetime.strptime(self.current_expiry, "%Y-%m-%d")
                days_to_expiry = (expiry_date - current_time).days + 1
                days_to_expiry = max(1, days_to_expiry)  # At least 1 day
            except:
                pass
        
        # Extract existing greeks or use provided values
        implied_volatility = option.get("implied_volatility", 0)
        delta = option.get("delta", 0)
        gamma = option.get("gamma", 0)
        theta = option.get("theta", 0)
        vega = option.get("vega", 0)
        
        # Calculate additional features
        put_call_ratio = 1.0  # Default (calculated later at the index level)
        spread = random.uniform(0.5, 2.0)  # Simulated bid-ask spread
        volume_oi_ratio = volume / open_interest if open_interest > 0 else 0
        
        return {
            **option,  # Include all original data
            "moneyness": round(moneyness, 4),
            "days_to_expiry": days_to_expiry,
            "put_call_ratio": put_call_ratio,
            "spread": round(spread, 2),
            "volume_oi_ratio": round(volume_oi_ratio, 4)
        }
    
    def _store_data(self, data):
        """
        Store processed data in MongoDB.
        
        Args:
            data (dict): Processed options data
        """
        if not self.mongo_db:
            return
        
        try:
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = datetime.datetime.now().isoformat()
            
            # Store in MongoDB
            self.mongo_db.insert_option_data(data)
            logger.debug("Option data stored in MongoDB")
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
    
    def get_latest_data(self):
        """
        Get the latest processed options data.
        
        Returns:
            dict: Latest processed options data
        """
        # Check if we need to refresh data (older than 1 minute)
        current_time = datetime.datetime.now()
        if (not self._last_update_time or 
            (current_time - self._last_update_time).total_seconds() > 60):
            return self.collect_live_data()
        
        return self._processed_data_cache
    
    def set_index(self, index_name):
        """
        Set the current index to track.
        
        Args:
            index_name (str): Index name
            
        Returns:
            bool: True if successful
        """
        # Validate index name
        valid_indices = [idx["symbol"] for idx in self.upstox_api.get_indices()]
        if index_name not in valid_indices:
            logger.warning(f"Invalid index name: {index_name}")
            return False
        
        logger.info(f"Setting current index to {index_name}")
        self.current_index = index_name
        
        # Clear cached data
        self._processed_data_cache = None
        self._last_update_time = None
        
        return True
    
    def set_expiry(self, expiry_date):
        """
        Set the expiry date to track.
        
        Args:
            expiry_date (str): Expiry date in YYYY-MM-DD format
            
        Returns:
            bool: True if successful
        """
        # Validate expiry date format
        try:
            if expiry_date:
                datetime.datetime.strptime(expiry_date, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid expiry date format: {expiry_date}")
            return False
        
        logger.info(f"Setting current expiry to {expiry_date}")
        self.current_expiry = expiry_date
        
        # Clear cached data
        self._processed_data_cache = None
        self._last_update_time = None
        
        return True
    
    def get_historical_option_data(self, symbol, days=30):
        """
        Get historical data for an option.
        
        Args:
            symbol (str): Option symbol
            days (int): Number of days of historical data
            
        Returns:
            list: Historical option data
        """
        if not self.mongo_db:
            # Generate synthetic data for demonstration
            return self._generate_synthetic_historical_data(symbol, days)
        
        # Retrieve from MongoDB if available
        return self.mongo_db.get_historical_option_data(symbol, days)
    
    def _generate_synthetic_historical_data(self, symbol, days):
        """
        Generate synthetic historical data for demonstration.
        
        Args:
            symbol (str): Option symbol
            days (int): Number of days
            
        Returns:
            list: Synthetic historical data
        """
        # Parse symbol to get details (format: INDEX_EXPIRY_STRIKE_TYPE)
        parts = symbol.split('_')
        if len(parts) < 4:
            return []
        
        index_name = parts[0]
        expiry_date = parts[1]
        strike_price = float(parts[2])
        option_type = "CALL" if parts[3] == "C" else "PUT"
        
        # Get current option data to use as a base
        current_data = self.upstox_api.get_option_chain(index_name, expiry_date)
        
        # Find the specific option
        option_list = current_data["calls"] if option_type == "CALL" else current_data["puts"]
        base_option = None
        for opt in option_list:
            if opt["strike_price"] == strike_price:
                base_option = opt
                break
        
        if not base_option:
            return []
        
        # Generate historical data based on the current option
        historical_data = []
        base_price = base_option["last_price"]
        
        # Start from days ago and move forward
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        current_date = start_date
        price = base_price * (1 - random.uniform(0.1, 0.3))  # Start slightly lower
        
        while current_date <= end_date:
            # Don't generate data for weekends
            if current_date.weekday() < 5:  # 0-4 is Monday-Friday
                # Calculate days to expiry
                try:
                    expiry = datetime.datetime.strptime(expiry_date, "%Y-%m-%d")
                    days_to_expiry = max(1, (expiry - current_date).days)
                except:
                    days_to_expiry = 30
                
                # Generate price with some randomness (trending up slightly overall)
                change = price * random.uniform(-0.05, 0.06)
                price += change
                price = max(0.1, price)  # Ensure minimum price
                
                # Calculate greeks based on current price and days to expiry
                implied_volatility = base_option["implied_volatility"] * (1 + random.uniform(-0.1, 0.1))
                delta = base_option["delta"] * (1 + random.uniform(-0.05, 0.05))
                gamma = base_option["gamma"] * (1 + random.uniform(-0.05, 0.05)) * days_to_expiry / 30
                theta = base_option["theta"] * (1 + random.uniform(-0.05, 0.05)) * 30 / days_to_expiry
                vega = base_option["vega"] * (1 + random.uniform(-0.05, 0.05)) * days_to_expiry / 30
                
                # Generate volume and open interest
                volume = int(base_option["volume"] * (1 + random.uniform(-0.3, 0.3)))
                open_interest = int(base_option["open_interest"] * (1 + random.uniform(-0.1, 0.1)))
                
                data_point = {
                    "symbol": symbol,
                    "timestamp": current_date.isoformat(),
                    "strike_price": strike_price,
                    "option_type": option_type,
                    "expiry_date": expiry_date,
                    "last_price": round(price, 2),
                    "volume": volume,
                    "open_interest": open_interest,
                    "implied_volatility": round(implied_volatility, 4),
                    "delta": round(delta, 4),
                    "gamma": round(gamma, 4),
                    "theta": round(theta, 4),
                    "vega": round(vega, 4),
                    "days_to_expiry": days_to_expiry
                }
                
                historical_data.append(data_point)
            
            # Move to next day
            current_date += datetime.timedelta(days=1)
        
        return historical_data
    
    def get_training_data(self, lookback_days=30):
        """
        Get formatted training data for the RL model.
        
        Args:
            lookback_days (int): Number of days to look back
            
        Returns:
            tuple: (observations, features) training data
        """
        # In a real implementation, this would fetch and format data for training
        # For this demonstration, we'll return simulated data
        
        # Get current option chain data
        current_data = self.get_latest_data()
        if not current_data:
            return None, None
        
        # Extract calls and puts
        calls = current_data.get("calls", [])
        puts = current_data.get("puts", [])
        
        # Combine into a single list of options
        all_options = []
        for call in calls:
            call["option_type"] = "CALL"
            all_options.append(call)
        
        for put in puts:
            put["option_type"] = "PUT"
            all_options.append(put)
        
        # Sort by strike price
        all_options.sort(key=lambda x: x.get("strike_price", 0))
        
        # For each option, generate synthetic historical data
        observations = []
        features = []
        
        for option in all_options:
            symbol = option.get("symbol", "")
            historical_data = self.get_historical_option_data(symbol, lookback_days)
            
            if historical_data:
                # Create observation window (most recent data first)
                obs = []
                for day_data in reversed(historical_data[-5:]):  # Use last 5 days
                    # Create feature vector for this observation
                    feature = [
                        day_data.get("last_price", 0),
                        day_data.get("volume", 0) / 1000,  # Scale volume
                        day_data.get("open_interest", 0) / 1000,  # Scale OI
                        day_data.get("implied_volatility", 0),
                        day_data.get("delta", 0),
                        day_data.get("gamma", 0) * 100,  # Scale gamma
                        day_data.get("theta", 0) / 10,  # Scale theta
                        day_data.get("vega", 0) * 10,  # Scale vega
                        day_data.get("days_to_expiry", 30) / 30  # Normalize days
                    ]
                    obs.append(feature)
                
                if len(obs) == 5:  # Only include complete sequences
                    observations.append(obs)
                    
                    # Extract label features (whether the option will be profitable)
                    # In real implementation, this would be calculated from historical returns
                    is_profitable = random.random() > 0.5  # Random for demo
                    features.append(1 if is_profitable else 0)
        
        return observations, features