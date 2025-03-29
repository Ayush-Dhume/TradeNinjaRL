"""
Web dashboard for monitoring and controlling the RL options trading system.
"""
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

logger = logging.getLogger(__name__)

def setup_routes(app, components):
    """
    Set up web dashboard routes.
    
    Args:
        app: Flask application
        components: System components
        
    Returns:
        None
    """
    @app.route('/')
    def index():
        """Render the dashboard home page."""
        return render_template('index.html')
    
    @app.route('/dashboard')
    def dashboard():
        """Render the trading dashboard."""
        # Get market status
        market_open = components['upstox_api'].is_market_open()
        
        # Get current index value
        index_name = components['data_processor'].current_index or "NIFTY 50"
        try:
            index_quote = components['upstox_api'].get_index_quote(index_name)
            spot_price = index_quote.get("last_price", 0)
            index_change = index_quote.get("change", 0)
            index_change_pct = index_quote.get("change_percentage", 0)
        except Exception as e:
            logger.error(f"Error getting index quote: {e}")
            spot_price = 0
            index_change = 0
            index_change_pct = 0
        
        # Get available indices
        available_indices = components['upstox_api'].get_indices()
        
        # Get recent trades
        recent_trades = components['mongo_db'].get_trades(
            start_date=datetime.now() - timedelta(days=1)
        )
        
        return render_template(
            'dashboard.html',
            market_open=market_open,
            index_name=index_name,
            spot_price=spot_price,
            index_change=index_change,
            index_change_pct=index_change_pct,
            available_indices=available_indices,
            recent_trades=recent_trades
        )
    
    @app.route('/api/option_chain')
    def get_option_chain():
        """API endpoint to get current option chain data."""
        try:
            index_name = request.args.get('index', components['data_processor'].current_index or "NIFTY 50")
            expiry_date = request.args.get('expiry', None)
            
            # Get option chain
            option_chain = components['upstox_api'].get_option_chain(index_name, expiry_date)
            
            # Format response
            calls = [format_option(opt) for opt in option_chain.get("calls", [])]
            puts = [format_option(opt) for opt in option_chain.get("puts", [])]
            
            return jsonify({
                "success": True,
                "calls": calls,
                "puts": puts,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/set_index', methods=['POST'])
    def set_index():
        """API endpoint to set the current index."""
        try:
            data = request.json
            index_name = data.get('index')
            
            if not index_name:
                return jsonify({"success": False, "error": "Index name required"}), 400
            
            # Set index in data processor
            success = components['data_processor'].set_index(index_name)
            
            if success:
                return jsonify({"success": True})
            else:
                return jsonify({"success": False, "error": "Invalid index name"}), 400
        except Exception as e:
            logger.error(f"Error setting index: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/set_expiry', methods=['POST'])
    def set_expiry():
        """API endpoint to set the current expiry date."""
        try:
            data = request.json
            expiry_date = data.get('expiry')
            
            if not expiry_date:
                return jsonify({"success": False, "error": "Expiry date required"}), 400
            
            # Set expiry in data processor
            success = components['data_processor'].set_expiry(expiry_date)
            
            if success:
                return jsonify({"success": True})
            else:
                return jsonify({"success": False, "error": "Invalid expiry date format"}), 400
        except Exception as e:
            logger.error(f"Error setting expiry: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/training')
    def training():
        """Render the model training page."""
        # Get model metrics
        metrics = components['mongo_db'].get_model_metrics(limit=20)
        
        return render_template('training.html', metrics=metrics)
    
    @app.route('/api/start_training', methods=['POST'])
    def start_training():
        """API endpoint to start model training."""
        try:
            data = request.json
            timesteps = int(data.get('timesteps', 100000))
            
            # Validate input
            if timesteps <= 0 or timesteps > 10000000:
                return jsonify({"success": False, "error": "Invalid timesteps value"}), 400
            
            # Start training in a separate thread
            import threading
            
            def train_model():
                try:
                    logger.info(f"Starting model training with {timesteps} timesteps")
                    components['model'].train(total_timesteps=timesteps)
                    logger.info("Model training completed")
                except Exception as e:
                    logger.error(f"Error during model training: {e}")
            
            thread = threading.Thread(target=train_model)
            thread.daemon = True
            thread.start()
            
            return jsonify({"success": True, "message": "Training started"})
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/model_metrics')
    def get_model_metrics():
        """API endpoint to get model training metrics."""
        try:
            limit = int(request.args.get('limit', 100))
            metrics = components['mongo_db'].get_model_metrics(limit=limit)
            
            # Format metrics for chart display
            formatted_metrics = []
            for metric in metrics:
                formatted_metrics.append({
                    "steps": metric.get("steps", 0),
                    "mean_reward": metric.get("mean_reward", 0),
                    "timestamp": metric.get("timestamp", 0)
                })
            
            return jsonify({
                "success": True,
                "metrics": formatted_metrics
            })
        except Exception as e:
            logger.error(f"Error fetching model metrics: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/backtesting')
    def backtesting():
        """Render the backtesting page."""
        # Get recent backtest results
        backtest_results = components['mongo_db'].db.backtest_results.find().sort("timestamp", -1).limit(5)
        
        return render_template('backtesting.html', backtest_results=list(backtest_results))
    
    @app.route('/api/start_backtest', methods=['POST'])
    def start_backtest():
        """API endpoint to start backtesting."""
        try:
            data = request.json
            initial_balance = float(data.get('initial_balance', 100000))
            days = int(data.get('days', 30))
            
            # Validate input
            if initial_balance <= 0:
                return jsonify({"success": False, "error": "Invalid initial balance"}), 400
            
            if days <= 0 or days > 365:
                return jsonify({"success": False, "error": "Invalid days value"}), 400
            
            # Import backtester
            from backtesting import Backtester
            
            # Initialize backtester
            backtester = Backtester(
                components['data_processor'],
                components['model'],
                components['mongo_db']
            )
            
            # Start backtesting in a separate thread
            import threading
            
            def run_backtest():
                try:
                    logger.info(f"Starting backtest with {initial_balance} initial balance over {days} days")
                    metrics = backtester.run_backtest(
                        start_date=datetime.now() - timedelta(days=days),
                        end_date=datetime.now(),
                        initial_balance=initial_balance,
                        plot_results=True
                    )
                    logger.info("Backtest completed")
                except Exception as e:
                    logger.error(f"Error during backtesting: {e}")
            
            thread = threading.Thread(target=run_backtest)
            thread.daemon = True
            thread.start()
            
            return jsonify({"success": True, "message": "Backtesting started"})
        except Exception as e:
            logger.error(f"Error starting backtest: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/backtest_results')
    def get_backtest_results():
        """API endpoint to get backtest results."""
        try:
            backtest_id = request.args.get('id')
            
            if backtest_id:
                # Get specific backtest result
                import bson
                result = components['mongo_db'].db.backtest_results.find_one({"_id": bson.ObjectId(backtest_id)})
                
                if not result:
                    return jsonify({"success": False, "error": "Backtest not found"}), 404
                
                # Convert ObjectId to string for serialization
                result["_id"] = str(result["_id"])
                
                return jsonify({
                    "success": True,
                    "result": result
                })
            else:
                # Get recent backtest results
                results = components['mongo_db'].db.backtest_results.find().sort("timestamp", -1).limit(10)
                
                # Convert ObjectId to string for serialization
                formatted_results = []
                for result in results:
                    result["_id"] = str(result["_id"])
                    formatted_results.append(result)
                
                return jsonify({
                    "success": True,
                    "results": formatted_results
                })
        except Exception as e:
            logger.error(f"Error fetching backtest results: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/available_expiries')
    def get_available_expiries():
        """API endpoint to get available expiry dates."""
        try:
            index_name = request.args.get('index', components['data_processor'].current_index or "NIFTY 50")
            
            # Get available options
            options_df = components['upstox_api'].get_available_options(index_name)
            
            if options_df.empty:
                return jsonify({
                    "success": True,
                    "expiries": []
                })
            
            # Extract unique expiry dates
            expiries = options_df['expiry_date'].unique().tolist() if 'expiry_date' in options_df.columns else []
            
            # Sort expiries
            expiries.sort()
            
            return jsonify({
                "success": True,
                "expiries": expiries
            })
        except Exception as e:
            logger.error(f"Error fetching available expiries: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


def format_option(option):
    """
    Format option data for API response.
    
    Args:
        option (dict): Raw option data
        
    Returns:
        dict: Formatted option data
    """
    return {
        "symbol": option.get("symbol", ""),
        "strike_price": option.get("strike_price", 0),
        "last_price": option.get("last_price", 0),
        "change": option.get("change", 0),
        "change_percentage": option.get("change_percentage", 0),
        "volume": option.get("volume", 0),
        "open_interest": option.get("open_interest", 0),
        "implied_volatility": option.get("implied_volatility", 0),
        "delta": option.get("delta", 0),
        "gamma": option.get("gamma", 0),
        "theta": option.get("theta", 0),
        "vega": option.get("vega", 0)
    }
