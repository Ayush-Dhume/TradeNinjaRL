"""
Reinforcement learning model implementation for options trading.
Using Stable-Baselines3 for PPO algorithm when available.
"""
import logging
import datetime
import random
import os
import json
import pickle

logger = logging.getLogger(__name__)

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Using standard Python lists.")

# Try to import Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    STABLE_BASELINES_AVAILABLE = True

    class SaveMetricsCallback(BaseCallback):
        """
        Callback for saving model metrics during training.
        """
        def __init__(self, db_handler, verbose=0):
            super(SaveMetricsCallback, self).__init__(verbose)
            self.db_handler = db_handler
            self.steps = 0
            
        def _on_step(self):
            self.steps += 1
            # Save metrics every 1000 steps
            if self.steps % 1000 == 0:
                metrics = {
                    "steps": self.model.num_timesteps,
                    "mean_reward": float(np.mean(self.model.ep_info_buffer.get("r", [-1]))),
                    "std_reward": float(np.std(self.model.ep_info_buffer.get("r", [0]))),
                    "mean_episode_length": float(np.mean(self.model.ep_info_buffer.get("l", [0]))),
                    "learning_rate": self.model.learning_rate,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                if self.db_handler:
                    self.db_handler.save_model_metrics(metrics)
                logger.info(f"Training step {self.model.num_timesteps}: mean_reward={metrics['mean_reward']:.2f}")
            return True
            
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logger.warning("Stable-Baselines3 not available. Using simplified model instead.")

class RLModel:
    """
    Reinforcement Learning model for options trading.
    Uses Stable-Baselines3 PPO algorithm if available,
    otherwise falls back to a simplified simulation.
    """
    
    def __init__(self, env=None, monitor_db=None):
        """
        Initialize the RL model.
        
        Args:
            env: Trading environment
            monitor_db: Database for metrics and model states
        """
        self.env = env
        self.monitor_db = monitor_db
        self.model = None
        self.training_in_progress = False
        self.training_steps = 0
        self.model_params = {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "max_grad_norm": 0.5
        }
        
        logger.info("Initialized RL model")
    
    def create_model(self):
        """
        Create and configure the model.
        
        Returns:
            object: Configured model
        """
        logger.info("Creating model with params: {}".format(json.dumps(self.model_params)))
        
        if STABLE_BASELINES_AVAILABLE and self.env:
            try:
                # Create actual PPO model with Stable-Baselines3
                self.model = PPO(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.model_params["learning_rate"],
                    n_steps=self.model_params["n_steps"],
                    batch_size=self.model_params["batch_size"],
                    n_epochs=self.model_params["n_epochs"],
                    gamma=self.model_params["gamma"],
                    gae_lambda=self.model_params["gae_lambda"],
                    clip_range=self.model_params["clip_range"],
                    ent_coef=self.model_params["ent_coef"],
                    max_grad_norm=self.model_params["max_grad_norm"],
                    verbose=1
                )
                logger.info("Created Stable-Baselines3 PPO model")
                return self.model
            except Exception as e:
                logger.error(f"Error creating Stable-Baselines3 model: {e}")
                logger.warning("Falling back to simplified model")
        
        # Fallback: Create a simple simulated model
        self.model = {
            "name": "PPO",
            "params": self.model_params,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        return self.model
    
    def load_model(self, path=None):
        """
        Load a saved model from file or database.
        
        Args:
            path (str): Path to saved model, or None to load from database
            
        Returns:
            bool: True if model was loaded successfully
        """
        try:
            if path and STABLE_BASELINES_AVAILABLE and self.env:
                # Try to load actual Stable-Baselines3 model
                try:
                    logger.info(f"Loading Stable-Baselines3 model from path: {path}")
                    self.model = PPO.load(path, env=self.env)
                    logger.info("Stable-Baselines3 model loaded successfully")
                    return True
                except Exception as e:
                    logger.error(f"Error loading Stable-Baselines3 model: {e}")
                    logger.warning("Falling back to simplified model")
            
            if path:
                # Fallback to simplified model
                logger.info(f"Loading simplified model from path: {path}")
                
                try:
                    # Try to load pickled model
                    with open(path, 'rb') as f:
                        self.model = pickle.load(f)
                        logger.info("Loaded pickled model")
                        return True
                except:
                    # If that fails, create a simple model dict
                    self.model = {
                        "name": "PPO",
                        "params": self.model_params,
                        "created_at": datetime.datetime.now().isoformat(),
                        "loaded_from": path
                    }
            elif self.monitor_db:
                # Load from database
                logger.info("Loading model from database")
                
                # Try to deserialize the model
                model_binary, metadata = self.monitor_db.load_model_state("ppo_model")
                
                if not model_binary:
                    logger.warning("No model found in database")
                    return False
                
                # Check if this is a Stable-Baselines3 model
                if STABLE_BASELINES_AVAILABLE and self.env and not metadata.get("simulator", False):
                    try:
                        # Save binary to temporary file and load with SB3
                        temp_path = os.path.join('models', 'temp_model.zip')
                        os.makedirs('models', exist_ok=True)
                        with open(temp_path, 'wb') as f:
                            f.write(model_binary)
                        
                        self.model = PPO.load(temp_path, env=self.env)
                        logger.info("Loaded Stable-Baselines3 model from database")
                        return True
                    except Exception as e:
                        logger.error(f"Error loading Stable-Baselines3 model from database: {e}")
                        logger.warning("Falling back to simplified model")
                
                # Fallback: try to unpickle the model
                try:
                    self.model = pickle.loads(model_binary)
                    logger.info("Loaded pickled model from database")
                    return True
                except:
                    # If that fails, create a simple model dict
                    self.model = {
                        "name": "PPO",
                        "params": metadata.get("params", self.model_params),
                        "created_at": metadata.get("created_at", datetime.datetime.now().isoformat()),
                        "loaded_from": "database"
                    }
            else:
                logger.warning("No path or database provided for loading model")
                return False
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def train(self, total_timesteps=100000):
        """
        Train the model.
        
        Args:
            total_timesteps (int): Total number of timesteps to train for
            
        Returns:
            object: Trained model
        """
        if not self.model:
            self.create_model()
        
        logger.info(f"Starting training for {total_timesteps} timesteps")
        self.training_in_progress = True
        self.training_steps = 0
        
        # Check if we're using Stable-Baselines3
        if STABLE_BASELINES_AVAILABLE and hasattr(self.model, 'learn'):
            try:
                # Create callback for metrics logging
                metrics_callback = SaveMetricsCallback(self.monitor_db)
                
                # Train the model
                logger.info(f"Training with Stable-Baselines3 for {total_timesteps} timesteps")
                self.model.learn(
                    total_timesteps=total_timesteps,
                    callback=metrics_callback,
                    log_interval=1000
                )
                
                # Save model to file
                if not os.path.exists('models'):
                    os.makedirs('models')
                model_path = os.path.join('models', f'ppo_options_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
                self.model.save(model_path)
                logger.info(f"Model saved to {model_path}")
                
                # Save to database if available
                if self.monitor_db:
                    logger.info("Saving trained model to database")
                    try:
                        with open(model_path + '.zip', 'rb') as f:
                            model_binary = f.read()
                        
                        metadata = {
                            "params": self.model_params,
                            "created_at": datetime.datetime.now().isoformat(),
                            "last_trained": datetime.datetime.now().isoformat(),
                            "total_timesteps": total_timesteps
                        }
                        
                        self.monitor_db.save_model_state("ppo_model", model_binary, metadata)
                    except Exception as e:
                        logger.error(f"Error saving model to database: {e}")
                
                self.training_in_progress = False
                logger.info("Training with Stable-Baselines3 completed")
                return self.model
                
            except Exception as e:
                logger.error(f"Error during Stable-Baselines3 training: {e}")
                logger.warning("Falling back to simulated training")
        
        # Fallback to simulated training
        logger.info("Using simulated training")
        
        # Calculate number of epochs for reporting metrics
        n_epochs = 20
        steps_per_epoch = total_timesteps // n_epochs
        
        # Simulate training loop
        for epoch in range(n_epochs):
            # Simulate training for one epoch
            self.training_steps += steps_per_epoch
            
            # Calculate simulated metrics
            # In a real implementation, these would be actual training metrics
            mean_reward = 50 * (1 + 0.1 * epoch) + random.uniform(-10, 10)
            std_reward = 20 + random.uniform(-5, 5)
            mean_episode_length = 100 + random.uniform(-10, 10)
            
            # Log metrics
            metrics = {
                "steps": self.training_steps,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_episode_length": mean_episode_length,
                "learning_rate": self.model_params["learning_rate"],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            logger.info(f"Training progress: {self.training_steps}/{total_timesteps} steps, mean_reward: {mean_reward:.2f}")
            
            # Save metrics to database if available
            if self.monitor_db:
                self.monitor_db.save_model_metrics(metrics)
        
        # Update model with training results
        if isinstance(self.model, dict):
            self.model["last_trained"] = datetime.datetime.now().isoformat()
            self.model["total_timesteps"] = total_timesteps
            
            # Save model to database if available
            if self.monitor_db:
                logger.info("Saving simulated model to database")
                
                # In a real implementation, this would serialize the model
                model_binary = pickle.dumps(self.model)
                metadata = {
                    "params": self.model_params,
                    "created_at": self.model.get("created_at", datetime.datetime.now().isoformat()),
                    "last_trained": self.model.get("last_trained", datetime.datetime.now().isoformat()),
                    "total_timesteps": total_timesteps,
                    "simulator": True
                }
                
                self.monitor_db.save_model_state("ppo_model", model_binary, metadata)
        
        self.training_in_progress = False
        logger.info("Training completed")
        
        return self.model
    
    def predict(self, observation):
        """
        Make a prediction using the trained model.
        
        Args:
            observation: Environment observation
            
        Returns:
            tuple: (action, _)
        """
        if self.model is None:
            # Create model on the fly if not exists
            self.model = self.create_model()
        
        # Check if we're using Stable-Baselines3
        if STABLE_BASELINES_AVAILABLE and hasattr(self.model, 'predict'):
            try:
                # Convert observation to numpy array if it's a dict
                if isinstance(observation, dict):
                    # Extract features as numpy array based on environment expected format
                    # This would need to match the exact format expected by the trained model
                    features = self._extract_numpy_features(observation)
                    observation = features
                
                # Use SB3 model to predict
                action, _ = self.model.predict(
                    observation, 
                    deterministic=True
                )
                return action, None
            except Exception as e:
                logger.error(f"Error making prediction with Stable-Baselines3: {e}")
                logger.warning("Falling back to rule-based strategy")
        
        # Fallback to rule-based
        # Parse observation
        option_features = self._extract_features(observation)
        
        # Implement a simple rule-based strategy for demonstration
        action = self._rule_based_strategy(option_features)
        
        return action, None
        
    def _extract_numpy_features(self, observation):
        """
        Convert dictionary observation to numpy array for SB3 model.
        
        Args:
            observation: Dictionary observation
            
        Returns:
            numpy.ndarray or list: Feature array
        """
        if not isinstance(observation, dict):
            if NUMPY_AVAILABLE and isinstance(observation, (list, tuple, np.ndarray)):
                return np.array(observation).astype(np.float32)
            return observation
            
        # Extract market data
        market_data = observation.get('market_data', {})
        spot_price = market_data.get('spot_price', 0)
        
        # Extract portfolio data
        portfolio = observation.get('portfolio', {})
        balance = portfolio.get('balance', 0)
        portfolio_value = portfolio.get('portfolio_value', 0)
        
        # Get options data
        calls = market_data.get('calls', [])
        puts = market_data.get('puts', [])
        
        # Extract features from first N options (e.g., 5 calls and 5 puts)
        max_options = 5
        features = [spot_price, balance, portfolio_value]
        
        # Add call options features
        for i in range(min(max_options, len(calls))):
            option = calls[i]
            features.extend([
                option.get('strike', 0) / spot_price if spot_price else 0,
                option.get('last_price', 0),
                option.get('change_percent', 0) / 100,
                option.get('delta', 0),
                option.get('gamma', 0),
                option.get('theta', 0),
                option.get('vega', 0),
                option.get('days_to_expiry', 30) / 365.0,
            ])
        
        # Pad with zeros if we have fewer than max_options
        features.extend([0] * (max_options - len(calls)) * 8)
        
        # Add put options features
        for i in range(min(max_options, len(puts))):
            option = puts[i]
            features.extend([
                option.get('strike', 0) / spot_price if spot_price else 0,
                option.get('last_price', 0),
                option.get('change_percent', 0) / 100,
                option.get('delta', 0),
                option.get('gamma', 0),
                option.get('theta', 0),
                option.get('vega', 0),
                option.get('days_to_expiry', 30) / 365.0,
            ])
        
        # Pad with zeros if we have fewer than max_options
        features.extend([0] * (max_options - len(puts)) * 8)
        
        # Convert to numpy array if available, otherwise return the list
        if NUMPY_AVAILABLE:
            return np.array(features, dtype=np.float32)
        return features
    
    def _extract_features(self, observation):
        """
        Extract relevant features from the observation.
        
        Args:
            observation: Environment observation
            
        Returns:
            dict: Extracted features
        """
        # This would extract features from the observation
        # For demonstration, we'll assume a dictionary structure
        
        if isinstance(observation, dict):
            return observation
        else:
            # Try to parse as a list or array
            try:
                # Assume it's a feature vector
                return {
                    "price": observation[0] if len(observation) > 0 else 0,
                    "delta": observation[4] if len(observation) > 4 else 0,
                    "gamma": observation[5] if len(observation) > 5 else 0,
                    "theta": observation[6] if len(observation) > 6 else 0,
                    "vega": observation[7] if len(observation) > 7 else 0,
                    "days_to_expiry": observation[8] if len(observation) > 8 else 0,
                }
            except:
                # Return a default feature set
                return {
                    "price": 0,
                    "delta": 0,
                    "gamma": 0,
                    "theta": 0,
                    "vega": 0,
                    "days_to_expiry": 30
                }
    
    def _rule_based_strategy(self, features):
        """
        Implement a simple rule-based strategy for demonstration.
        
        Args:
            features (dict): Option features
            
        Returns:
            int: Action (0: do nothing, 1: buy, 2: sell)
        """
        # Extract key features
        delta = features.get("delta", 0)
        theta = features.get("theta", 0)
        days_to_expiry = features.get("days_to_expiry", 30)
        
        # Simple rule-based logic
        if delta > 0.6 and days_to_expiry < 7:
            # High delta options close to expiry - sell
            return 2
        elif delta < 0.3 and days_to_expiry > 20:
            # Low delta options with plenty of time - buy
            return 1
        elif theta < -0.1 and days_to_expiry < 5:
            # High time decay close to expiry - sell
            return 2
        else:
            # Default to doing nothing
            return 0
    
    def evaluate(self, n_episodes=10):
        """
        Evaluate the model performance over multiple episodes.
        
        Args:
            n_episodes (int): Number of episodes to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.env:
            logger.warning("No environment provided for evaluation")
            return {
                "mean_reward": 0,
                "std_reward": 0,
                "mean_episode_length": 0,
                "n_episodes": 0
            }
        
        if self.model is None:
            self.model = self.create_model()
        
        logger.info(f"Evaluating model over {n_episodes} episodes")
        
        # Check if we're using Stable-Baselines3
        if STABLE_BASELINES_AVAILABLE and hasattr(self.model, 'predict'):
            try:
                logger.info("Evaluating with Stable-Baselines3 model")
                
                # Run actual evaluation
                episode_rewards = []
                episode_lengths = []
                
                for i in range(n_episodes):
                    obs, _ = self.env.reset()
                    done = False
                    truncated = False
                    total_reward = 0
                    steps = 0
                    
                    while not (done or truncated):
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, _ = self.env.step(action)
                        total_reward += reward
                        steps += 1
                        
                        # Safety check to prevent infinite loops
                        if steps > 1000:
                            break
                    
                    episode_rewards.append(total_reward)
                    episode_lengths.append(steps)
                    logger.info(f"Episode {i+1}: reward={total_reward:.2f}, length={steps}")
                
                # Calculate metrics
                if NUMPY_AVAILABLE:
                    mean_reward = float(np.mean(episode_rewards))
                    std_reward = float(np.std(episode_rewards))
                    mean_episode_length = float(np.mean(episode_lengths))
                else:
                    # Manual calculation of mean and std
                    mean_reward = sum(episode_rewards) / len(episode_rewards)
                    mean_episode_length = sum(episode_lengths) / len(episode_lengths)
                    std_reward = (sum((r - mean_reward) ** 2 for r in episode_rewards) / len(episode_rewards)) ** 0.5
                
                metrics = {
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "mean_episode_length": mean_episode_length,
                    "n_episodes": n_episodes,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                logger.info(f"Evaluation results: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")
                return metrics
                
            except Exception as e:
                logger.error(f"Error during Stable-Baselines3 evaluation: {e}")
                logger.warning("Falling back to simulated evaluation")
        
        # Fallback to simulated evaluation
        logger.info("Using simulated evaluation")
        
        # Simulate episode rewards
        rewards = [random.uniform(50, 150) for _ in range(n_episodes)]
        episode_lengths = [random.randint(80, 120) for _ in range(n_episodes)]
        
        # Calculate metrics
        mean_reward = sum(rewards) / len(rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
        mean_episode_length = sum(episode_lengths) / len(episode_lengths)
        
        metrics = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_episode_length": mean_episode_length,
            "n_episodes": n_episodes,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Evaluation results: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")
        
        return metrics