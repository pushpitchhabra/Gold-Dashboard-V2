"""
Updater Module for Gold Trading Dashboard
Handles daily model retraining and data updates
"""

import pandas as pd
import yaml
import logging
import schedule
import time
from datetime import datetime, timedelta
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldModelUpdater:
    def __init__(self, config_path="config.yaml"):
        """Initialize updater with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_loader = None
        self.feature_engineer = None
        self.model_trainer = None
        
        # Initialize components lazily
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize required components"""
        from data_loader import GoldDataLoader
        from feature_engineer import GoldFeatureEngineer
        from model_trainer import GoldModelTrainer
        
        self.data_loader = GoldDataLoader()
        self.feature_engineer = GoldFeatureEngineer()
        self.model_trainer = GoldModelTrainer()
    
    def update_historical_data(self):
        """
        Update historical data with latest market data
        """
        logger.info("Starting historical data update...")
        
        try:
            # Update historical data file with new data
            updated_df = self.data_loader.update_historical_data()
            
            if updated_df is not None and not updated_df.empty:
                logger.info(f"Historical data updated successfully. Total records: {len(updated_df)}")
                return True
            else:
                logger.warning("No new data to add to historical file")
                return False
                
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
            return False
    
    def retrain_model(self, force_retrain=False):
        """
        Retrain the model with updated data
        """
        logger.info("Starting model retraining...")
        
        try:
            # Check if model needs retraining
            if not force_retrain and not self._should_retrain_model():
                logger.info("Model retraining not needed at this time")
                return False
            
            # Load updated historical data
            historical_data = self.data_loader.load_historical_data()
            
            if historical_data.empty:
                logger.error("No historical data available for training")
                return False
            
            logger.info(f"Loaded {len(historical_data)} historical records for retraining")
            
            # Prepare features for training
            features, target = self.feature_engineer.prepare_features_for_training(historical_data)
            
            if features.empty or target.empty:
                logger.error("No feature data available for training")
                return False
            
            logger.info(f"Prepared {len(features)} samples with {len(features.columns)} features")
            
            # Train the model
            metrics = self.model_trainer.train_model(features, target)
            
            if not metrics:
                logger.error("Model training failed - no metrics returned")
                return False
            
            # Save the trained model
            self.model_trainer.save_model(features.columns.tolist())
            
            # Log training results
            logger.info("Model retraining completed successfully!")
            logger.info(f"New model accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Cross-validation mean: {metrics.get('cv_mean', 0):.4f}")
            
            # Save update log
            self._log_update_event("model_retrain", {
                'accuracy': metrics['accuracy'],
                'cv_mean': metrics.get('cv_mean', 0),
                'training_samples': len(features),
                'features_count': len(features.columns)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error log
            self._log_update_event("model_retrain_error", {
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            return False
    
    def _should_retrain_model(self):
        """
        Determine if model should be retrained based on various criteria
        """
        try:
            # Check if model file exists
            model_file = self.config['model']['model_file']
            if not os.path.exists(model_file):
                logger.info("Model file doesn't exist, retraining needed")
                return True
            
            # Check model age
            metadata_file = model_file.replace('.pkl', '_metadata.yaml')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                training_date = datetime.fromisoformat(metadata.get('training_date', '2000-01-01'))
                days_since_training = (datetime.now() - training_date).days
                
                # Retrain if model is older than 7 days
                if days_since_training >= 7:
                    logger.info(f"Model is {days_since_training} days old, retraining needed")
                    return True
            
            # Check data freshness
            historical_file = self.config['data']['historical_file']
            if os.path.exists(historical_file):
                df = pd.read_csv(historical_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                
                last_data_date = df['datetime'].max()
                hours_since_data = (datetime.now() - last_data_date.replace(tzinfo=None)).total_seconds() / 3600
                
                # Retrain if data is more than 24 hours old
                if hours_since_data >= 24:
                    logger.info(f"Data is {hours_since_data:.1f} hours old, retraining needed")
                    return True
            
            logger.info("Model retraining not needed based on current criteria")
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain criteria: {e}")
            return True  # Default to retraining if we can't determine
    
    def _log_update_event(self, event_type, details):
        """
        Log update events for monitoring
        """
        # Convert numpy objects to native Python types
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': convert_numpy_types(details)
        }
        
        log_file = 'data/update_log.yaml'
        
        # Load existing log with error handling
        log_data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    log_data = yaml.safe_load(f) or []
            except yaml.constructor.ConstructorError:
                # If YAML is corrupted, start fresh
                logger.warning("Corrupted log file detected, starting fresh log")
                log_data = []
            except Exception as e:
                logger.error(f"Error loading log file: {e}")
                log_data = []
        
        # Add new entry
        log_data.append(log_entry)
        
        # Keep only last 100 entries
        log_data = log_data[-100:]
        
        # Save log with error handling
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w') as f:
                yaml.dump(log_data, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving log file: {e}")
    
    def daily_update_routine(self):
        """
        Complete daily update routine
        """
        logger.info("Starting daily update routine...")
        
        update_results = {
            'timestamp': datetime.now().isoformat(),
            'data_updated': False,
            'model_retrained': False,
            'errors': []
        }
        
        try:
            # Update historical data
            data_updated = self.update_historical_data()
            update_results['data_updated'] = data_updated
            
            # Retrain model if needed
            model_retrained = self.retrain_model()
            update_results['model_retrained'] = model_retrained
            
            # Log the update routine
            self._log_update_event("daily_routine", update_results)
            
            logger.info("Daily update routine completed successfully!")
            logger.info(f"Data updated: {data_updated}, Model retrained: {model_retrained}")
            
            return update_results
            
        except Exception as e:
            error_msg = f"Error in daily update routine: {e}"
            logger.error(error_msg)
            update_results['errors'].append(error_msg)
            
            # Log the error
            self._log_update_event("daily_routine_error", {'error': error_msg})
            
            return update_results
    
    def schedule_daily_updates(self, time_str="02:00"):
        """
        Schedule daily updates to run automatically
        """
        logger.info(f"Scheduling daily updates at {time_str}")
        
        # Schedule the daily routine
        schedule.every().day.at(time_str).do(self.daily_update_routine)
        
        logger.info("Daily updates scheduled. Starting scheduler...")
        
        # Keep the scheduler running
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def run_update_now(self):
        """
        Run update routine immediately (for manual triggering)
        """
        logger.info("Running manual update...")
        return self.daily_update_routine()
    
    def force_retrain_now(self):
        """
        Force model retraining immediately
        """
        logger.info("Running forced model retraining...")
        
        # Update data first
        self.update_historical_data()
        
        # Force retrain
        return self.retrain_model(force_retrain=True)
    
    def get_update_status(self):
        """
        Get status of recent updates
        """
        log_file = 'data/update_log.yaml'
        
        if not os.path.exists(log_file):
            return {
                'last_update': None,
                'last_retrain': None,
                'recent_updates': []
            }
        
        try:
            with open(log_file, 'r') as f:
                log_data = yaml.safe_load(f) or []
            
            # Find last update and retrain events
            last_update = None
            last_retrain = None
            
            for entry in reversed(log_data):
                if entry['event_type'] == 'daily_routine' and last_update is None:
                    last_update = entry['timestamp']
                
                if entry['event_type'] == 'model_retrain' and last_retrain is None:
                    last_retrain = entry['timestamp']
                
                if last_update and last_retrain:
                    break
            
            return {
                'last_update': last_update,
                'last_retrain': last_retrain,
                'recent_updates': log_data[-10:]  # Last 10 entries
            }
            
        except Exception as e:
            logger.error(f"Error getting update status: {e}")
            return {
                'last_update': None,
                'last_retrain': None,
                'recent_updates': []
            }

def main():
    """Test the updater or run scheduled updates"""
    import sys
    
    updater = GoldModelUpdater()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "schedule":
            # Run scheduled updates
            time_str = sys.argv[2] if len(sys.argv) > 2 else "02:00"
            updater.schedule_daily_updates(time_str)
        
        elif command == "update":
            # Run update now
            result = updater.run_update_now()
            print("Update Results:")
            print(f"Data Updated: {result['data_updated']}")
            print(f"Model Retrained: {result['model_retrained']}")
            if result['errors']:
                print(f"Errors: {result['errors']}")
        
        elif command == "retrain":
            # Force retrain
            success = updater.force_retrain_now()
            print(f"Forced retrain {'successful' if success else 'failed'}")
        
        elif command == "status":
            # Show status
            status = updater.get_update_status()
            print("Update Status:")
            print(f"Last Update: {status['last_update']}")
            print(f"Last Retrain: {status['last_retrain']}")
            print(f"Recent Updates: {len(status['recent_updates'])}")
    
    else:
        print("Usage:")
        print("  python updater.py schedule [time]  - Schedule daily updates (default: 02:00)")
        print("  python updater.py update          - Run update now")
        print("  python updater.py retrain         - Force retrain model")
        print("  python updater.py status          - Show update status")

if __name__ == "__main__":
    main()
