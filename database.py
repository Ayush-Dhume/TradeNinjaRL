"""
Database module for storing options trading data and model states.
Supports both MongoDB and PostgreSQL.
"""
import logging
import datetime
import json
import os
import random
import time
import pickle
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import dependencies conditionally to handle missing packages gracefully
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    MONGODB_AVAILABLE = True
    logger.info("MongoDB support enabled.")
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("pymongo package not found. MongoDB support will be disabled.")

try:
    import sqlalchemy
    from sqlalchemy import Column, String, Float, Integer, DateTime, Text, LargeBinary, ForeignKey, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship
    POSTGRESQL_AVAILABLE = True
    Base = declarative_base()
    
    # Define SQLAlchemy models for PostgreSQL
    class OptionData(Base):
        __tablename__ = 'option_data'
        
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, index=True)
        symbol = Column(String(50), index=True)
        spot_price = Column(Float)
        index_name = Column(String(50))
        expiry_date = Column(String(20))
        data_json = Column(Text)  # Store JSON serialized option data
        
    class Trade(Base):
        __tablename__ = 'trades'
        
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, index=True)
        symbol = Column(String(50), index=True)
        option_type = Column(String(10))
        strike_price = Column(Float)
        quantity = Column(Integer)
        price = Column(Float)
        trade_type = Column(String(10))  # 'BUY' or 'SELL'
        profit_loss = Column(Float, nullable=True)
        
    class ModelMetric(Base):
        __tablename__ = 'model_metrics'
        
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, index=True)
        model_name = Column(String(100), index=True)
        accuracy = Column(Float, nullable=True)
        loss = Column(Float, nullable=True)
        reward = Column(Float, nullable=True)
        metrics_json = Column(Text)  # Additional metrics as JSON
        
    class ModelState(Base):
        __tablename__ = 'model_states'
        
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, index=True)
        model_name = Column(String(100), index=True)
        model_binary = Column(LargeBinary)
        metadata_json = Column(Text)  # Metadata as JSON
        
    class BacktestResult(Base):
        __tablename__ = 'backtest_results'
        
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, index=True)
        model_name = Column(String(100), index=True)
        start_date = Column(DateTime)
        end_date = Column(DateTime)
        initial_balance = Column(Float)
        final_balance = Column(Float)
        total_trades = Column(Integer)
        profitable_trades = Column(Integer)
        max_drawdown = Column(Float)
        sharpe_ratio = Column(Float)
        results_json = Column(Text)  # Additional results as JSON
        
except ImportError:
    POSTGRESQL_AVAILABLE = False
    logger.warning("sqlalchemy package not found. PostgreSQL support will be limited.")

class DatabaseHandler:
    """
    Database handler for storing options trading data.
    Supports MongoDB with fallback to PostgreSQL or in-memory storage.
    """
    
    def __init__(self, connection_string=None, db_name=None):
        """
        Initialize database connection with fallback mechanisms.
        Tries MongoDB first, then PostgreSQL, and finally in-memory storage.
        
        Args:
            connection_string (str): MongoDB connection string
            db_name (str): Name of the database
        """
        self.connection_string = connection_string or os.environ.get("MONGO_URI", "")
        self.db_name = db_name or "options_trading"
        
        # Flags for database types
        self.use_real_mongo = False
        self.use_postgresql = False
        self.use_in_memory = False
        
        # Try PostgreSQL first since it's available in the Replit environment
        if POSTGRESQL_AVAILABLE:
            try:
                # Get PostgreSQL connection string from environment
                pg_uri = os.environ.get("DATABASE_URL")
                
                if pg_uri:
                    # Create SQLAlchemy engine and session
                    self.engine = sqlalchemy.create_engine(pg_uri)
                    Base.metadata.create_all(self.engine)
                    Session = sessionmaker(bind=self.engine)
                    self.session = Session()
                    
                    logger.info(f"Connected to PostgreSQL database")
                    
                    # Create a pseudo-MongoDB style interface for compatibility with the rest of the code
                    self.db = type('obj', (object,), {
                        'backtest_results': type('obj', (object,), {
                            'find': lambda *a, **k: self._pg_find_backtest_results(*a, **k),
                            'find_one': lambda *a, **k: self._pg_find_one_backtest_results(*a, **k),
                            'insert_one': lambda *a, **k: self._pg_insert_one_backtest_results(*a, **k)
                        })
                    })
                    
                    # Set flag
                    self.use_postgresql = True
                else:
                    logger.warning("PostgreSQL connection string not found in environment.")
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                logger.warning("Trying MongoDB as a fallback...")
        
        # Try MongoDB if PostgreSQL is not available or failed
        if not self.use_postgresql and MONGODB_AVAILABLE and self.connection_string:
            try:
                # Connect to real MongoDB
                self.client = MongoClient(self.connection_string)
                self.mongodb = self.client[self.db_name]
                
                # Create indices for common queries
                self.mongodb.option_data.create_index([("timestamp", DESCENDING)])
                self.mongodb.trades.create_index([("timestamp", DESCENDING)])
                self.mongodb.model_metrics.create_index([("timestamp", DESCENDING)])
                
                # Test the connection
                self.client.admin.command('ping')
                logger.info(f"Connected to MongoDB database: {self.db_name}")
                
                # Create a db object for direct access to collections
                self.db = type('obj', (object,), {
                    'backtest_results': self.mongodb.backtest_results
                })
                
                # Set flag
                self.use_real_mongo = True
                
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                logger.warning("Falling back to in-memory storage...")
        
        # If neither MongoDB nor PostgreSQL is available, use in-memory storage
        if not self.use_real_mongo and not self.use_postgresql:
            # Set up in-memory storage
            self.collections = {
                "option_data": [],
                "trades": [],
                "model_metrics": [],
                "model_states": [],
                "backtest_results": []
            }
            
            # Create a db object for direct access
            self.db = type('obj', (object,), {
                'backtest_results': type('obj', (object,), {
                    'find': lambda *a, **k: self._find("backtest_results", *a, **k),
                    'find_one': lambda *a, **k: self._find_one("backtest_results", *a, **k),
                    'insert_one': lambda *a, **k: self._insert_one("backtest_results", *a, **k)
                })
            })
            
            # Set flag
            self.use_in_memory = True
            
            logger.info(f"Initialized in-memory storage for options trading data")
    
    def _pg_find_backtest_results(self, filter=None, sort=None, limit=None):
        """
        Find backtest results in PostgreSQL.
        
        Args:
            filter (dict): Filter criteria
            sort (list): Sort criteria as a list of tuples
            limit (int): Maximum number of documents to return
            
        Returns:
            list: List of matching documents
        """
        try:
            query = self.session.query(BacktestResult)
            
            # Apply filter
            if filter:
                for key, value in filter.items():
                    if key == '_id':
                        query = query.filter(BacktestResult.id == value)
                    elif hasattr(BacktestResult, key):
                        query = query.filter(getattr(BacktestResult, key) == value)
            
            # Apply sort
            if sort:
                for sort_item in sort:
                    if isinstance(sort_item, tuple):
                        field, direction = sort_item
                        if hasattr(BacktestResult, field):
                            if direction == -1:
                                query = query.order_by(sqlalchemy.desc(getattr(BacktestResult, field)))
                            else:
                                query = query.order_by(getattr(BacktestResult, field))
            
            # Apply limit
            if limit:
                query = query.limit(limit)
                
            # Execute query
            results = query.all()
            
            # Convert to list of dicts
            result_list = []
            for result in results:
                result_dict = {
                    "_id": str(result.id),
                    "timestamp": result.timestamp,
                    "model_name": result.model_name,
                    "start_date": result.start_date,
                    "end_date": result.end_date,
                    "initial_balance": result.initial_balance,
                    "final_balance": result.final_balance,
                    "total_trades": result.total_trades,
                    "profitable_trades": result.profitable_trades,
                    "max_drawdown": result.max_drawdown,
                    "sharpe_ratio": result.sharpe_ratio
                }
                
                # Add results_json if present
                if result.results_json:
                    try:
                        additional_results = json.loads(result.results_json)
                        result_dict.update(additional_results)
                    except:
                        pass
                
                result_list.append(result_dict)
            
            return result_list
            
        except Exception as e:
            logger.error(f"Error in _pg_find_backtest_results: {e}")
            return []
    
    def _pg_find_one_backtest_results(self, filter=None):
        """
        Find a single backtest result in PostgreSQL.
        
        Args:
            filter (dict): Filter criteria
            
        Returns:
            dict: Matching document or None
        """
        results = self._pg_find_backtest_results(filter=filter, limit=1)
        return results[0] if results else None
    
    def _pg_insert_one_backtest_results(self, document):
        """
        Insert a backtest result document into PostgreSQL.
        
        Args:
            document (dict): Document to insert
            
        Returns:
            dict: Inserted document with _id
        """
        try:
            # Extract fields from document
            timestamp = document.get("timestamp", datetime.datetime.now())
            model_name = document.get("model_name", "unknown")
            start_date = document.get("start_date")
            end_date = document.get("end_date")
            initial_balance = document.get("initial_balance")
            final_balance = document.get("final_balance")
            total_trades = document.get("total_trades")
            profitable_trades = document.get("profitable_trades")
            max_drawdown = document.get("max_drawdown")
            sharpe_ratio = document.get("sharpe_ratio")
            
            # Create a copy of the document without standard fields for JSON storage
            json_doc = document.copy()
            for field in ["timestamp", "model_name", "start_date", "end_date", 
                         "initial_balance", "final_balance", "total_trades", 
                         "profitable_trades", "max_drawdown", "sharpe_ratio"]:
                if field in json_doc:
                    del json_doc[field]
            
            # Convert to JSON
            results_json = json.dumps(json_doc)
            
            # Create new BacktestResult record
            backtest_result = BacktestResult(
                timestamp=timestamp,
                model_name=model_name,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance,
                final_balance=final_balance,
                total_trades=total_trades,
                profitable_trades=profitable_trades,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                results_json=results_json
            )
            
            # Add to session and commit
            self.session.add(backtest_result)
            self.session.commit()
            
            # Update document with id
            document["_id"] = str(backtest_result.id)
            
            return document
            
        except Exception as e:
            logger.error(f"Error in _pg_insert_one_backtest_results: {e}")
            return None
    
    def _find(self, collection_name, filter=None, sort=None, limit=None):
        """
        Find documents in a collection.
        
        Args:
            collection_name (str): Name of the collection
            filter (dict): Filter criteria
            sort (tuple): Sort criteria
            limit (int): Maximum number of documents to return
            
        Returns:
            list: List of matching documents
        """
        collection = self.collections.get(collection_name, [])
        
        # Apply filter (simplified)
        results = collection
        if filter:
            filtered_results = []
            for doc in collection:
                match = True
                for key, value in filter.items():
                    if key not in doc or doc[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append(doc)
            results = filtered_results
        
        # Apply sort (simplified)
        if sort:
            field, direction = sort
            reverse = direction == -1
            results.sort(key=lambda x: x.get(field, 0), reverse=reverse)
        
        # Apply limit
        if limit:
            results = results[:limit]
        
        return results
    
    def _find_one(self, collection_name, filter=None):
        """
        Find a single document in a collection.
        
        Args:
            collection_name (str): Name of the collection
            filter (dict): Filter criteria
            
        Returns:
            dict: Matching document or None
        """
        results = self._find(collection_name, filter, limit=1)
        return results[0] if results else None
    
    def _insert_one(self, collection_name, document):
        """
        Insert a document into a collection.
        
        Args:
            collection_name (str): Name of the collection
            document (dict): Document to insert
            
        Returns:
            dict: Inserted document
        """
        # Add _id if not present
        if "_id" not in document:
            document["_id"] = str(random.randint(10000, 99999))
        
        # Add timestamp if not present
        if "timestamp" not in document:
            document["timestamp"] = datetime.datetime.now().isoformat()
        
        # Insert into collection
        collection = self.collections.get(collection_name, [])
        collection.append(document)
        
        return document
    
    def insert_option_data(self, data):
        """
        Insert option chain data into the database.
        
        Args:
            data (dict): Option chain data
            
        Returns:
            bool: True if successful
        """
        try:
            # Ensure timestamp is present
            if "timestamp" not in data:
                data["timestamp"] = datetime.datetime.now()
            
            if self.use_real_mongo:
                # Use real MongoDB
                result = self.mongodb.option_data.insert_one(data)
                return result.acknowledged
                
            elif self.use_postgresql:
                # Use PostgreSQL
                # Extract key data
                spot_price = data.get("spot_price", 0.0)
                index_name = data.get("index_name", "NIFTY 50")
                expiry_date = data.get("expiry_date", "")
                
                # Serialize the complete data as JSON for storage
                data_json = json.dumps(data)
                
                # Create new OptionData record
                option_data = OptionData(
                    timestamp=data["timestamp"],
                    symbol=index_name,  # Use index as main symbol
                    spot_price=spot_price,
                    index_name=index_name,
                    expiry_date=expiry_date,
                    data_json=data_json
                )
                
                # Add to session and commit
                self.session.add(option_data)
                self.session.commit()
                return True
                
            else:
                # Use in-memory storage
                self._insert_one("option_data", data)
                return True
                
        except Exception as e:
            logger.error(f"Error inserting option data: {e}")
            return False
    
    def get_latest_option_data(self):
        """
        Get the latest option data from MongoDB.
        
        Returns:
            dict: Latest option data or None
        """
        try:
            if self.use_real_mongo:
                # Use real MongoDB
                result = self.mongodb.option_data.find_one(
                    sort=[("timestamp", -1)]
                )
                return result
            else:
                # Use simulated MongoDB
                results = self._find("option_data", sort=("timestamp", -1), limit=1)
                return results[0] if results else None
                
        except Exception as e:
            logger.error(f"Error getting latest option data: {e}")
            return None
    
    def get_historical_option_data(self, symbol, days=30):
        """
        Get historical option data for a specific symbol.
        
        Args:
            symbol (str): Option symbol
            days (int): Number of days to look back
            
        Returns:
            list: Historical option data
        """
        try:
            # Calculate cutoff timestamp
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            if self.use_real_mongo:
                # Use real MongoDB
                pipeline = [
                    {"$match": {"timestamp": {"$gte": cutoff_date}}},
                    {"$unwind": "$calls"},
                    {"$match": {"calls.symbol": symbol}},
                    {"$project": {
                        "timestamp": 1,
                        "symbol": "$calls.symbol",
                        "strike_price": "$calls.strike_price",
                        "option_type": "$calls.option_type",
                        "last_price": "$calls.last_price",
                        "change": "$calls.change",
                        "volume": "$calls.volume",
                        "open_interest": "$calls.open_interest",
                        "implied_volatility": "$calls.implied_volatility",
                        "delta": "$calls.delta",
                        "gamma": "$calls.gamma",
                        "theta": "$calls.theta",
                        "vega": "$calls.vega"
                    }},
                    {"$sort": {"timestamp": 1}}
                ]
                
                call_results = list(self.mongodb.option_data.aggregate(pipeline))
                
                # Repeat for puts
                pipeline = [
                    {"$match": {"timestamp": {"$gte": cutoff_date}}},
                    {"$unwind": "$puts"},
                    {"$match": {"puts.symbol": symbol}},
                    {"$project": {
                        "timestamp": 1,
                        "symbol": "$puts.symbol",
                        "strike_price": "$puts.strike_price",
                        "option_type": "$puts.option_type",
                        "last_price": "$puts.last_price",
                        "change": "$puts.change",
                        "volume": "$puts.volume",
                        "open_interest": "$puts.open_interest",
                        "implied_volatility": "$puts.implied_volatility",
                        "delta": "$puts.delta",
                        "gamma": "$puts.gamma",
                        "theta": "$puts.theta",
                        "vega": "$puts.vega"
                    }},
                    {"$sort": {"timestamp": 1}}
                ]
                
                put_results = list(self.mongodb.option_data.aggregate(pipeline))
                
                # Combine and sort results
                results = call_results + put_results
                results.sort(key=lambda x: x.get("timestamp"))
                
                return results
            else:
                # Use simulated MongoDB
                cutoff_timestamp = cutoff_date.isoformat()
                all_data = self._find("option_data")
                
                results = []
                for data in all_data:
                    if data.get("timestamp", "") >= cutoff_timestamp:
                        # Extract the option data from calls or puts
                        for call in data.get("calls", []):
                            if call.get("symbol") == symbol:
                                results.append({**call, "timestamp": data.get("timestamp")})
                        
                        for put in data.get("puts", []):
                            if put.get("symbol") == symbol:
                                results.append({**put, "timestamp": data.get("timestamp")})
                
                # Sort by timestamp
                results.sort(key=lambda x: x.get("timestamp", ""))
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting historical option data: {e}")
            return []
    
    def record_trade(self, trade_data):
        """
        Record a trade in the database.
        
        Args:
            trade_data (dict): Trade details
            
        Returns:
            bool: True if successful
        """
        try:
            # Ensure timestamp is present
            if "timestamp" not in trade_data:
                trade_data["timestamp"] = datetime.datetime.now()
            
            if self.use_real_mongo:
                # Use real MongoDB
                result = self.mongodb.trades.insert_one(trade_data)
                return result.acknowledged
            else:
                # Use simulated MongoDB
                self._insert_one("trades", trade_data)
                return True
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False
    
    def get_trades(self, start_date=None, end_date=None, symbol=None):
        """
        Get trades from the database with optional filtering.
        
        Args:
            start_date (datetime): Start date filter
            end_date (datetime): End date filter
            symbol (str): Symbol filter
            
        Returns:
            list: Filtered trades
        """
        try:
            # Apply filters
            if self.use_real_mongo:
                # Use real MongoDB
                filter = {}
                if start_date:
                    filter["timestamp"] = {"$gte": start_date}
                if end_date:
                    if "timestamp" in filter:
                        filter["timestamp"]["$lte"] = end_date
                    else:
                        filter["timestamp"] = {"$lte": end_date}
                if symbol:
                    filter["symbol"] = symbol
                
                trades = list(self.mongodb.trades.find(
                    filter=filter,
                    sort=[("timestamp", -1)]
                ))
                
                return trades
            elif self.use_postgresql:
                # Use PostgreSQL
                query = self.session.query(Trade)
                
                # Apply filters if provided
                if start_date:
                    query = query.filter(Trade.timestamp >= start_date)
                if end_date:
                    query = query.filter(Trade.timestamp <= end_date)
                if symbol:
                    query = query.filter(Trade.symbol == symbol)
                
                # Order by timestamp descending
                query = query.order_by(Trade.timestamp.desc())
                
                # Execute query and convert to list of dictionaries
                trades = []
                for trade in query.all():
                    trade_dict = {
                        "_id": str(trade.id),
                        "timestamp": trade.timestamp,
                        "symbol": trade.symbol,
                        "option_type": trade.option_type,
                        "strike_price": trade.strike_price,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "trade_type": trade.trade_type,
                        "profit_loss": trade.profit_loss
                    }
                    trades.append(trade_dict)
                
                return trades
            else:
                # Use in-memory storage
                if not hasattr(self, 'collections'):
                    self.collections = {
                        "option_data": [],
                        "trades": [],
                        "model_metrics": [],
                        "model_states": [],
                        "backtest_results": []
                    }
                    logger.info("Initialized collections for in-memory storage")
                
                # Use simulated MongoDB with in-memory collections
                start_timestamp = start_date.isoformat() if start_date else ""
                end_timestamp = end_date.isoformat() if end_date else datetime.datetime.now().isoformat()
                
                # Find matching trades
                all_trades = self._find("trades")
                
                filtered_trades = []
                for trade in all_trades:
                    timestamp = trade.get("timestamp", "")
                    trade_symbol = trade.get("symbol", "")
                    
                    if timestamp >= start_timestamp and timestamp <= end_timestamp:
                        if not symbol or trade_symbol == symbol:
                            filtered_trades.append(trade)
                
                # Sort by timestamp
                filtered_trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                
                return filtered_trades
                
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    def save_model_metrics(self, metrics):
        """
        Save model training metrics.
        
        Args:
            metrics (dict): Model performance metrics
            
        Returns:
            bool: True if successful
        """
        try:
            # Ensure timestamp is present
            if "timestamp" not in metrics:
                metrics["timestamp"] = datetime.datetime.now()
            
            if self.use_real_mongo:
                # Use real MongoDB
                result = self.mongodb.model_metrics.insert_one(metrics)
                return result.acknowledged
            elif self.use_postgresql:
                # Use PostgreSQL
                # Extract main metrics fields
                model_name = metrics.get("model_name", "default_model")
                accuracy = metrics.get("accuracy")
                loss = metrics.get("loss")
                reward = metrics.get("reward")
                
                # Prepare additional metrics for JSON storage
                metrics_json = metrics.copy()
                for field in ["timestamp", "model_name", "accuracy", "loss", "reward"]:
                    if field in metrics_json:
                        del metrics_json[field]
                
                # Convert to JSON
                metrics_json_str = json.dumps(metrics_json)
                
                # Create new ModelMetric record
                model_metric = ModelMetric(
                    timestamp=metrics["timestamp"],
                    model_name=model_name,
                    accuracy=accuracy,
                    loss=loss,
                    reward=reward,
                    metrics_json=metrics_json_str
                )
                
                # Add to session and commit
                self.session.add(model_metric)
                self.session.commit()
                return True
            else:
                # Use in-memory storage
                if hasattr(self, 'collections') and "model_metrics" in self.collections:
                    self._insert_one("model_metrics", metrics)
                    return True
                else:
                    logger.warning("No collections attribute or model_metrics collection found")
                    # Initialize collections if not present
                    if not hasattr(self, 'collections'):
                        self.collections = {
                            "option_data": [],
                            "trades": [],
                            "model_metrics": [],
                            "model_states": [],
                            "backtest_results": []
                        }
                        logger.info("Initialized collections for in-memory storage")
                    
                    # Now insert the metrics
                    self._insert_one("model_metrics", metrics)
                    return True
                
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
            return False
    
    def get_model_metrics(self, limit=100):
        """
        Get recent model training metrics.
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            list: Model metrics
        """
        try:
            if self.use_real_mongo:
                # Use real MongoDB
                metrics = list(self.mongodb.model_metrics.find(
                    sort=[("timestamp", -1)],
                    limit=limit
                ))
                return metrics
            elif self.use_postgresql:
                # Use PostgreSQL
                metrics = []
                try:
                    # Query model metrics from PostgreSQL
                    model_metrics = self.session.query(ModelMetric).order_by(
                        ModelMetric.timestamp.desc()
                    ).limit(limit).all()
                    
                    # Convert SQLAlchemy objects to dictionaries
                    for metric in model_metrics:
                        metric_dict = {
                            "_id": str(metric.id),
                            "timestamp": metric.timestamp,
                            "model_name": metric.model_name,
                            "accuracy": metric.accuracy,
                            "loss": metric.loss,
                            "reward": metric.reward
                        }
                        
                        # Add additional metrics from JSON
                        if metric.metrics_json:
                            try:
                                additional_metrics = json.loads(metric.metrics_json)
                                metric_dict.update(additional_metrics)
                            except:
                                pass
                        
                        metrics.append(metric_dict)
                    
                    return metrics
                except Exception as e:
                    logger.error(f"Error querying PostgreSQL for model metrics: {e}")
                    return []
            else:
                # Use in-memory storage
                if hasattr(self, 'collections') and "model_metrics" in self.collections:
                    metrics = self._find("model_metrics", sort=("timestamp", -1), limit=limit)
                    return metrics
                else:
                    logger.warning("No collections attribute or model_metrics collection found")
                    return []
                
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return []
    
    def save_model_state(self, model_name, model_binary, metadata=None):
        """
        Save model state binary data.
        
        Args:
            model_name (str): Name of the model
            model_binary (binary): Serialized model state
            metadata (dict): Additional metadata
            
        Returns:
            bool: True if successful
        """
        try:
            # Create document with model state
            timestamp = datetime.datetime.now()
            metadata_dict = metadata or {}
            
            if self.use_real_mongo:
                # In a real implementation with MongoDB, we can store large binary data
                # in GridFS, which is designed for this purpose
                from pymongo import GridFS
                
                # Store the model binary in GridFS
                fs = GridFS(self.mongodb)
                file_id = fs.put(model_binary, filename=f"{model_name}_{int(time.time())}")
                
                # Store reference to the file in the document
                document = {
                    "model_name": model_name,
                    "metadata": metadata_dict,
                    "timestamp": timestamp,
                    "model_file_id": file_id
                }
                
                # Insert document into model_states collection
                result = self.mongodb.model_states.insert_one(document)
                return result.acknowledged
            elif self.use_postgresql:
                # Use PostgreSQL
                try:
                    # Convert metadata to JSON
                    metadata_json = json.dumps(metadata_dict)
                    
                    # Create new ModelState record
                    model_state = ModelState(
                        timestamp=timestamp,
                        model_name=model_name,
                        model_binary=model_binary,
                        metadata_json=metadata_json
                    )
                    
                    # Add to session and commit
                    self.session.add(model_state)
                    self.session.commit()
                    return True
                except Exception as e:
                    logger.error(f"Error saving model state to PostgreSQL: {e}")
                    return False
            else:
                # For in-memory version, just note that it was saved
                document = {
                    "model_name": model_name,
                    "metadata": metadata_dict,
                    "timestamp": timestamp,
                    "model_binary_saved": True
                }
                
                # Insert into model_states collection
                if hasattr(self, 'collections') and "model_states" in self.collections:
                    self._insert_one("model_states", document)
                    return True
                else:
                    logger.warning("No collections attribute or model_states collection found")
                    # Initialize collections if not present
                    if not hasattr(self, 'collections'):
                        self.collections = {
                            "option_data": [],
                            "trades": [],
                            "model_metrics": [],
                            "model_states": [],
                            "backtest_results": []
                        }
                        logger.info("Initialized collections for in-memory storage")
                    
                    # Now insert the document
                    self._insert_one("model_states", document)
                    return True
                
        except Exception as e:
            logger.error(f"Error saving model state: {e}")
            return False
    
    def load_model_state(self, model_name):
        """
        Load model state binary data.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            tuple: (model_binary, metadata) or (None, None) if not found
        """
        try:
            if self.use_real_mongo:
                # Find the most recent model state for this model
                model_state = self.mongodb.model_states.find_one(
                    {"model_name": model_name},
                    sort=[("timestamp", -1)]
                )
                
                if not model_state:
                    return None, None
                
                # Retrieve the binary data from GridFS
                from pymongo import GridFS
                fs = GridFS(self.mongodb)
                
                if "model_file_id" in model_state:
                    file_id = model_state["model_file_id"]
                    grid_out = fs.get(file_id)
                    model_binary = grid_out.read()
                    metadata = model_state.get("metadata", {})
                    return model_binary, metadata
                
                return None, model_state.get("metadata", {})
            elif self.use_postgresql:
                # Use PostgreSQL
                try:
                    # Query the most recent model state for this model
                    model_state = self.session.query(ModelState).filter(
                        ModelState.model_name == model_name
                    ).order_by(ModelState.timestamp.desc()).first()
                    
                    if not model_state:
                        return None, None
                    
                    # Extract binary data and metadata
                    model_binary = model_state.model_binary
                    
                    # Parse metadata from JSON
                    metadata = {}
                    if model_state.metadata_json:
                        try:
                            metadata = json.loads(model_state.metadata_json)
                        except:
                            logger.warning(f"Failed to parse metadata JSON for model {model_name}")
                    
                    return model_binary, metadata
                except Exception as e:
                    logger.error(f"Error loading model state from PostgreSQL: {e}")
                    return None, None
            else:
                # For in-memory implementation
                if hasattr(self, 'collections') and "model_states" in self.collections:
                    model_state = self._find_one("model_states", {"model_name": model_name})
                    
                    if not model_state:
                        return None, None
                    
                    # Return simulated binary data
                    model_binary = b"simulated_model_binary_data"
                    metadata = model_state.get("metadata", {})
                    
                    return model_binary, metadata
                else:
                    logger.warning("No collections attribute or model_states collection found")
                    return None, None
                
        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            return None, None