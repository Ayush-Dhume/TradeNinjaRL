from flask import Flask, render_template, jsonify, request, redirect, url_for
import os
import logging
import datetime
import json
import threading

# Import our modules
from upstox_api import UpstoxAPI
from data_processor import DataProcessor
from database import DatabaseHandler
from model import RLModel
from trading_env import OptionsEnv
from backtesting import Backtester

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Add environment variables to Flask configuration for template access
app.config['UPSTOX_API_KEY'] = os.environ.get("UPSTOX_API_KEY", "")
app.config['UPSTOX_API_SECRET'] = os.environ.get("UPSTOX_API_SECRET", "")
app.config['UPSTOX_REDIRECT_URI'] = os.environ.get("UPSTOX_REDIRECT_URI", "")

# Initialize components
def initialize_components():
    logger.info("Initializing application components...")
    
    # Determine the application URL for the redirect URI
    replit_domains = os.environ.get("REPLIT_DOMAINS")
    replit_dev_domain = os.environ.get("REPLIT_DEV_DOMAIN")
    
    if replit_domains:
        # In production Replit environment
        app_url = f"https://{replit_domains.split(',')[0]}"
    elif replit_dev_domain:
        # In development Replit environment
        app_url = f"https://{replit_dev_domain}"
    else:
        # Local development
        app_url = "http://localhost:5000"
    
    # Default redirect URI based on the app URL
    default_redirect_uri = f"{app_url}/upstox/callback"
    logger.info(f"Using default redirect URI: {default_redirect_uri}")
    
    # Initialize Upstox API
    upstox_api = UpstoxAPI(
        api_key=os.environ.get("UPSTOX_API_KEY", ""),
        api_secret=os.environ.get("UPSTOX_API_SECRET", ""),
        redirect_uri=os.environ.get("UPSTOX_REDIRECT_URI", default_redirect_uri)
    )
    
    # Initialize DatabaseHandler
    db_handler = DatabaseHandler(
        connection_string=os.environ.get("MONGO_URI", "mongodb://localhost:27017/"),
        db_name="options_trading"
    )
    
    # Initialize data processor
    data_processor = DataProcessor(upstox_api, db_handler)
    
    # Initialize trading environment
    env = OptionsEnv(data_processor)
    
    # Initialize RL model
    model = RLModel(env, db_handler)
    
    # Attempt to get initial data
    try:
        # This will raise an exception if authentication is required
        data_processor.collect_live_data()
        logger.info("Successfully collected initial market data")
    except Exception as e:
        logger.warning(f"Could not collect initial live data: {e}")
        logger.warning("Authentication required for live market data")
    
    return {
        "upstox_api": upstox_api,
        "db_handler": db_handler,
        "data_processor": data_processor,
        "environment": env,
        "model": model
    }

# Create global components
components = initialize_components()

# Background data collection
def background_data_collection():
    logger.info("Starting background data collection...")
    while True:
        try:
            # Check if market is open
            if components["upstox_api"].is_market_open():
                logger.info("Market is open, collecting data...")
                components["data_processor"].collect_live_data()
            else:
                logger.info("Market is closed, skipping data collection")
            
            # Sleep for 5 minutes
            import time
            time.sleep(300)
        except Exception as e:
            logger.error(f"Error in background data collection: {e}")
            import time
            time.sleep(60)  # Sleep for 1 minute on error

# Start background data collection thread
data_thread = threading.Thread(target=background_data_collection, daemon=True)
data_thread.start()

# Routes
@app.route('/heartbeat')
def heartbeat():
    """Simple route to verify server is running."""
    return jsonify({"status": "ok", "timestamp": datetime.datetime.now().isoformat()})

@app.route('/')
def index():
    """Render the dashboard home page."""
    # Check if Upstox API is authenticated
    is_authenticated = False
    auth_url = None
    
    try:
        # Test authentication by making a simple API call
        components["upstox_api"].is_market_open()
        is_authenticated = True
    except Exception as e:
        logger.warning(f"Upstox API not authenticated: {e}")
        # Generate OAuth URL for authentication
        auth_url = components["upstox_api"].get_auth_url()
    
    return render_template(
        'index.html',
        is_authenticated=is_authenticated,
        auth_url=auth_url
    )

@app.route('/dashboard')
def dashboard():
    """Render the trading dashboard."""
    # Check authentication first
    is_authenticated = False
    auth_url = None
    
    try:
        # Test authentication by making a simple API call
        components["upstox_api"].is_market_open()
        is_authenticated = True
    except Exception as e:
        logger.warning(f"Upstox API not authenticated for dashboard: {e}")
        # Generate OAuth URL for authentication
        auth_url = components["upstox_api"].get_auth_url()
        
        # If not authenticated, render a dashboard with limited functionality
        return render_template(
            'dashboard.html',
            market_open=False,
            index_name=components["data_processor"].current_index,
            spot_price=0,
            index_change=0,
            index_change_pct=0,
            available_indices=[],
            recent_trades=[],
            is_authenticated=False,
            auth_url=auth_url,
            auth_required=True,
            error="API authentication required to access live market data."
        )
    
    # If we get here, we're authenticated
    try:
        # Get market status
        market_open = components["upstox_api"].is_market_open()
        
        # Get current index value
        index_name = components["data_processor"].current_index
        index_quote = components["upstox_api"].get_index_quote(index_name)
        spot_price = index_quote.get("last_price", 0)
        index_change = index_quote.get("change", 0)
        index_change_pct = index_quote.get("change_percentage", 0)
        
        # Get available indices
        available_indices = components["upstox_api"].get_indices()
        
        # Get recent trades
        recent_trades = components["db_handler"].get_trades(
            start_date=datetime.datetime.now() - datetime.timedelta(days=1)
        )
        
        return render_template(
            'dashboard.html',
            market_open=market_open,
            index_name=index_name,
            spot_price=spot_price,
            index_change=index_change,
            index_change_pct=index_change_pct,
            available_indices=available_indices,
            recent_trades=recent_trades,
            is_authenticated=is_authenticated
        )
    except Exception as e:
        logger.error(f"Error loading dashboard data: {e}")
        return render_template(
            'dashboard.html',
            market_open=False,
            index_name=components["data_processor"].current_index,
            spot_price=0,
            index_change=0,
            index_change_pct=0,
            available_indices=[],
            recent_trades=[],
            is_authenticated=False,
            auth_url=auth_url,
            error=f"Error loading market data: {str(e)}"
        )

@app.route('/training')
def training():
    """Render the model training page."""
    # Get model metrics
    metrics = components["db_handler"].get_model_metrics(limit=20)
    
    return render_template('training.html', metrics=metrics)

@app.route('/backtesting')
def backtesting():
    """Render the backtesting page."""
    # Get recent backtest results
    try:
        if hasattr(components["db_handler"], "db"):
            backtest_results = components["db_handler"].db.backtest_results.find().sort("timestamp", -1).limit(5)
            backtest_results = list(backtest_results)
        else:
            # Use the PostgreSQL query method
            backtest_results = components["db_handler"]._pg_find_backtest_results(sort=[("timestamp", -1)], limit=5)
    except Exception as e:
        logger.error(f"Error fetching backtest results: {e}")
        backtest_results = []
    
    return render_template('backtesting.html', backtest_results=backtest_results)

@app.route('/model_docs')
def model_docs():
    """Render the model documentation page."""
    return render_template('model_docs.html')

@app.route('/api/option_chain')
def get_option_chain():
    """API endpoint to get current option chain data."""
    try:
        index_name = request.args.get('index', components["data_processor"].current_index)
        expiry_date = request.args.get('expiry', None)
        
        # Get option chain
        option_chain = components["upstox_api"].get_option_chain(index_name, expiry_date)
        
        # Format response
        calls = option_chain.get("calls", [])
        puts = option_chain.get("puts", [])
        
        return jsonify({
            "success": True,
            "calls": calls,
            "puts": puts,
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/available_expiries')
def get_available_expiries():
    """API endpoint to get available expiry dates."""
    try:
        index_name = request.args.get('index', components["data_processor"].current_index)
        
        # Get available options
        options_df = components["upstox_api"].get_available_options(index_name)
        
        if hasattr(options_df, 'empty') and options_df.empty:
            return jsonify({
                "success": True,
                "expiries": []
            })
        
        # Extract unique expiry dates
        if hasattr(options_df, 'expiry_date'):
            expiries = options_df['expiry_date'].unique().tolist()
        else:
            # For our mock implementation which might not return a real DataFrame
            expiries = []
            if hasattr(options_df, 'to_dict'):
                data_dict = options_df.to_dict('records')
                expiries = list(set(item.get('expiry_date') for item in data_dict if 'expiry_date' in item))
        
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

@app.route('/api/set_index', methods=['POST'])
def set_index():
    """API endpoint to set the current index."""
    try:
        data = request.json
        index_name = data.get('index')
        
        if not index_name:
            return jsonify({"success": False, "error": "Index name required"}), 400
        
        # Set index in data processor
        success = components["data_processor"].set_index(index_name)
        
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
        success = components["data_processor"].set_expiry(expiry_date)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Invalid expiry date format"}), 400
    except Exception as e:
        logger.error(f"Error setting expiry: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

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
        def train_model():
            try:
                logger.info(f"Starting model training with {timesteps} timesteps")
                components["model"].train(total_timesteps=timesteps)
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
        metrics = components["db_handler"].get_model_metrics(limit=limit)
        
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
        
        # Initialize backtester
        backtester = Backtester(
            components["data_processor"],
            components["model"],
            components["db_handler"]
        )
        
        # Start backtesting in a separate thread
        def run_backtest():
            try:
                logger.info(f"Starting backtest with {initial_balance} initial balance over {days} days")
                metrics = backtester.run_backtest(
                    start_date=datetime.datetime.now() - datetime.timedelta(days=days),
                    end_date=datetime.datetime.now(),
                    initial_balance=float(initial_balance),
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
        
        # Determine which database method to use
        if backtest_id:
            # Get specific backtest result
            if hasattr(components["db_handler"], "db"):
                # MongoDB
                result = components["db_handler"].db.backtest_results.find_one({"_id": backtest_id})
            else:
                # PostgreSQL
                result = components["db_handler"]._pg_find_one_backtest_results({"id": backtest_id})
            
            if not result:
                return jsonify({"success": False, "error": "Backtest not found"}), 404
            
            # Convert _id to string for serialization
            if hasattr(result, 'get') and "_id" in result:
                result["_id"] = str(result["_id"])
            
            return jsonify({
                "success": True,
                "result": result
            })
        else:
            # Get recent backtest results
            if hasattr(components["db_handler"], "db"):
                # MongoDB
                results = components["db_handler"].db.backtest_results.find().sort("timestamp", -1).limit(10)
                formatted_results = []
                for result in results:
                    if hasattr(result, 'get') and "_id" in result:
                        result["_id"] = str(result["_id"])
                    formatted_results.append(result)
            else:
                # PostgreSQL
                formatted_results = components["db_handler"]._pg_find_backtest_results(
                    sort=[("timestamp", -1)], 
                    limit=10
                )
            
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

# Upstox API OAuth2 flow routes
@app.route('/upstox/callback')
def upstox_callback():
    """Handle callback from Upstox OAuth authentication."""
    try:
        # Get the authorization code from query parameters
        auth_code = request.args.get('code')
        error = request.args.get('error')
        error_description = request.args.get('error_description')
        
        # Handle errors from OAuth flow
        if error:
            error_message = f"OAuth Error: {error}"
            if error_description:
                error_message += f" - {error_description}"
            logger.error(f"Upstox OAuth error: {error_message}")
            return redirect(url_for('index', error=error_message))
        
        if not auth_code:
            logger.error("Authorization code not provided in callback")
            return redirect(url_for('index', error="Authorization code not provided"))
        
        # Exchange the authorization code for an access token
        success = components["upstox_api"].exchange_code_for_token(auth_code)
        
        if success:
            logger.info("Successfully authenticated with Upstox API")
            return redirect(url_for('dashboard'))
        else:
            logger.error("Failed to exchange code for token")
            return redirect(url_for('index', error="Authentication failed"))
            
    except Exception as e:
            logger.error(f"Error in callback: {e}")
            return redirect(url_for('index', error=str(e)))

@app.route('/upstox/auth')
def upstox_auth():
    """Generate and redirect to Upstox authentication URL."""
    try:
        # Update API credentials from environment variables first
        components["upstox_api"].api_key = os.environ.get("UPSTOX_API_KEY", components["upstox_api"].api_key)
        components["upstox_api"].api_secret = os.environ.get("UPSTOX_API_SECRET", components["upstox_api"].api_secret)
        
        # Get the redirect URI from environment or calculate it
        replit_domains = os.environ.get("REPLIT_DOMAINS")
        replit_dev_domain = os.environ.get("REPLIT_DEV_DOMAIN")
        
        # Generate a default redirect URI based on the app URL
        if replit_domains:
            # In production Replit environment
            app_url = f"https://{replit_domains.split(',')[0]}"
            default_redirect_uri = f"{app_url}/upstox/callback"
        elif replit_dev_domain:
            # In development Replit environment
            app_url = f"https://{replit_dev_domain}"
            default_redirect_uri = f"{app_url}/upstox/callback"
        else:
            # Local development
            default_redirect_uri = f"{request.host_url.rstrip('/')}/upstox/callback"
        
        # Set the redirect URI (prefer environment variable if available)
        components["upstox_api"].redirect_uri = os.environ.get("UPSTOX_REDIRECT_URI", default_redirect_uri)
        
        # Log the credentials being used (masking sensitive data)
        api_key_masked = f"{components['upstox_api'].api_key[:4]}{'*' * (len(components['upstox_api'].api_key) - 4)}" if components["upstox_api"].api_key and len(components["upstox_api"].api_key) > 4 else "Not set"
        logger.info(f"Using API Key: {api_key_masked}")
        logger.info(f"Using API Secret: {'Set' if components['upstox_api'].api_secret else 'Not set'}")
        logger.info(f"Using Redirect URI: {components['upstox_api'].redirect_uri}")
        
        # Check if we have all necessary credentials
        if not components["upstox_api"].api_key or not components["upstox_api"].api_secret:
            return render_template('index.html', error="API Key and Secret are required for authentication. Please set them first.")
        
        # Generate the authorization URL with the updated credentials
        auth_url = components["upstox_api"].get_auth_url()
        
        # Log the URL for debugging
        logger.info(f"Generated Upstox authorization URL: {auth_url}")
        
        # Directly redirect to the Upstox authorization URL
        return redirect(auth_url)
    except Exception as e:
        logger.error(f"Error generating Upstox authorization URL: {e}")
        return render_template('index.html', error=f"Failed to generate authorization URL: {str(e)}")

# API endpoint for Upstox API credentials
@app.route('/api/set_api_key', methods=['POST'])
def set_api_key():
    """API endpoint to set the Upstox API key and secret."""
    try:
        data = request.json
        api_key = data.get('api_key')
        api_secret = data.get('api_secret')
        redirect_uri = data.get('redirect_uri')
        
        if not api_key and not api_secret and not redirect_uri:
            return jsonify({
                "success": False,
                "message": "No API credentials provided. Please provide at least API key or API secret."
            }), 400
            
        success = False
        message = "API credentials not provided"
        
        # Set API key in both runtime and environment variables
        if api_key:
            components["upstox_api"].api_key = api_key
            os.environ["UPSTOX_API_KEY"] = api_key
            success = True
            message = "API key updated"
            logger.info("Upstox API key updated")
        
        # Set API secret in both runtime and environment variables
        if api_secret:
            components["upstox_api"].api_secret = api_secret
            os.environ["UPSTOX_API_SECRET"] = api_secret
            success = True
            message = "API credentials updated"
            logger.info("Upstox API secret updated")
            
        # Set redirect URI in both runtime and environment variables
        if redirect_uri:
            components["upstox_api"].redirect_uri = redirect_uri
            os.environ["UPSTOX_REDIRECT_URI"] = redirect_uri
            success = True
            message = "API credentials and redirect URI updated"
            logger.info("Upstox redirect URI updated")
        
        # If both API key and secret are provided, attempt authentication
        if api_key and api_secret:
            try:
                auth_success = components["upstox_api"]._authenticate()
                if auth_success:
                    message = "Authentication successful"
                    logger.info("Upstox authentication successful")
                else:
                    message = "API credentials updated but authentication requires OAuth flow"
                    logger.info("Upstox authentication requires OAuth flow")
                    # Generate auth URL for the frontend to redirect the user
                    auth_url = components["upstox_api"].get_auth_url()
            except Exception as e:
                logger.error(f"Error during Upstox authentication: {e}")
                message = f"API credentials updated but authentication failed: {str(e)}"
                auth_url = components["upstox_api"].get_auth_url()
                return jsonify({
                    "success": success, 
                    "message": message,
                    "requires_oauth": True,
                    "auth_url": auth_url
                })
        
        return jsonify({"success": success, "message": message})
    
    except Exception as e:
        logger.error(f"Error setting API credentials: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)