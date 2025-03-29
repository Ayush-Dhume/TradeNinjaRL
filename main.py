import os
import logging
from app import app

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for Upstox API credentials on startup
upstox_api_key = os.environ.get("UPSTOX_API_KEY")
upstox_api_secret = os.environ.get("UPSTOX_API_SECRET")
upstox_redirect_uri = os.environ.get("UPSTOX_REDIRECT_URI")

if not upstox_api_key or not upstox_api_secret:
    logger.warning("==================================================================")
    logger.warning("UPSTOX API CREDENTIALS NOT FOUND IN ENVIRONMENT VARIABLES")
    logger.warning("This application requires real-time market data from Upstox API")
    logger.warning("Please configure your API credentials through the web interface")
    logger.warning("==================================================================")
else:
    logger.info("Upstox API credentials found in environment variables")
    if not upstox_redirect_uri:
        logger.warning("No redirect URI configured, will use default")

# Start the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)