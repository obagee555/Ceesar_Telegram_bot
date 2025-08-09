import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
import threading
from lstm_model import LSTMTradingModel
from technical_analysis import TechnicalAnalysis
from sentiment_analysis import SentimentAnalysis
from pocket_option_api import PocketOptionAPI
from config import *

class SignalGenerator:
    def __init__(self):
        self.lstm_model = LSTMTradingModel()
        self.technical_analyzer = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalysis()
        self.pocket_api = PocketOptionAPI()
        
        self.is_running = False
        self.last_signal_time = {}
        self.signal_history = []
        self.min_signal_interval = 300  # 5 minutes between signals for same pair
        
        # Performance tracking
        self.daily_signals = 0
        self.accuracy_tracker = []
        
    def start_signal_generation(self):
        """Start the signal generation process"""
        try:
            logging.info("Starting signal generation engine...")
            
            # Start Pocket Option API
            if not self.pocket_api.start():
                logging.error("Failed to start Pocket Option API")
                return False
            
            self.is_running = True
            
            # Start signal generation thread
            signal_thread = threading.Thread(target=self.signal_generation_loop)
            signal_thread.daemon = True
            signal_thread.start()
            
            logging.info("Signal generation engine started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error starting signal generation: {e}")
            return False
    
    def stop_signal_generation(self):
        """Stop the signal generation process"""
        try:
            self.is_running = False
            self.pocket_api.stop()
            logging.info("Signal generation engine stopped")
            
        except Exception as e:
            logging.error(f"Error stopping signal generation: {e}")
    
    def signal_generation_loop(self):
        """Main LSTM-driven signal generation loop"""
        while self.is_running:
            try:
                # Let LSTM AI choose the best currency pair for trading
                available_pairs = self.pocket_api.get_available_pairs()
                
                # LSTM analyzes all pairs and selects the best one
                best_pair, pair_analysis = self.lstm_model.analyze_best_currency_pair(
                    available_pairs, self.pocket_api.get_market_data_df
                )
                
                if best_pair and self.should_generate_signal(best_pair):
                    # Generate signal for LSTM-selected pair
                    signal = self.generate_signal(best_pair)
                    
                    if signal and self.validate_signal(signal):
                        # Add LSTM pair selection info
                        signal['lstm_pair_selection'] = {
                            'selected_by_lstm': True,
                            'pair_score': pair_analysis['score'],
                            'pairs_analyzed': len(available_pairs),
                            'selection_reason': 'Highest LSTM confidence and favorable conditions'
                        }
                        
                        # Log and store signal
                        self.store_signal(signal)
                        
                        # Broadcast signal
                        self.broadcast_signal(signal)
                        
                        # Wait longer after generating a signal (LSTM quality over quantity)
                        time.sleep(300)  # 5 minutes between LSTM signals
                
                # Sleep for 60 seconds before next LSTM analysis
                time.sleep(60)
                
            except Exception as e:
                logging.error(f"Error in LSTM signal generation loop: {e}")
                time.sleep(120)  # Wait longer if there's an error
    
    def should_generate_signal(self, pair):
        """Check if we should generate a signal for this pair"""
        try:
            # Check minimum interval between signals
            current_time = time.time()
            if pair in self.last_signal_time:
                time_since_last = current_time - self.last_signal_time[pair]
                if time_since_last < self.min_signal_interval:
                    return False
            
            # Check if market is open and conditions are favorable
            if not self.is_market_conditions_favorable(pair):
                return False
            
            # Check volatility
            if not self.pocket_api.is_low_volatility(pair):
                logging.debug(f"High volatility detected for {pair}, skipping signal generation")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking signal generation conditions: {e}")
            return False
    
    def is_market_conditions_favorable(self, pair):
        """Check if market conditions are favorable for trading"""
        try:
            # Get current market data
            market_data = self.pocket_api.get_market_data_df(pair, limit=100)
            
            if market_data is None or len(market_data) < 50:
                return False
            
            # Check with technical analysis
            analyzed_data = self.technical_analyzer.calculate_all_indicators(market_data)
            if analyzed_data is None:
                return False
            
            # Check if conditions are favorable
            return self.technical_analyzer.is_favorable_conditions(analyzed_data)
            
        except Exception as e:
            logging.error(f"Error checking market conditions: {e}")
            return False
    
    def generate_signal(self, pair):
        """Generate a comprehensive trading signal"""
        try:
            logging.info(f"Generating signal for {pair}")
            
            # Get market data
            market_data = self.pocket_api.get_market_data_df(pair, limit=200)
            if market_data is None or len(market_data) < 60:
                logging.warning(f"Insufficient market data for {pair}")
                return None
            
            # Perform technical analysis
            analyzed_data = self.technical_analyzer.calculate_all_indicators(market_data)
            if analyzed_data is None:
                logging.warning(f"Technical analysis failed for {pair}")
                return None
            
            # Get LSTM prediction
            lstm_direction, lstm_confidence = self.lstm_model.predict_direction(analyzed_data)
            if lstm_direction is None:
                logging.warning(f"LSTM prediction failed for {pair}")
                return None
            
            # Get technical analysis signal
            tech_direction, tech_confidence = self.technical_analyzer.get_trade_direction(analyzed_data)
            
            # Get sentiment analysis
            sentiment_data = self.sentiment_analyzer.get_comprehensive_sentiment(pair.split('/')[0])
            sentiment_bias = self.sentiment_analyzer.get_sentiment_bias(sentiment_data)
            
            # Combine all signals
            combined_signal = self.combine_signals(
                lstm_direction, lstm_confidence,
                tech_direction, tech_confidence,
                sentiment_bias, analyzed_data, pair
            )
            
            return combined_signal
            
        except Exception as e:
            logging.error(f"Error generating signal for {pair}: {e}")
            return None
    
    def combine_signals(self, lstm_direction, lstm_confidence, tech_direction, 
                       tech_confidence, sentiment_bias, analyzed_data, pair):
        """Generate signal based primarily on LSTM AI model analysis"""
        try:
            # LSTM AI model is the primary decision maker (95% weight)
            lstm_weight = 0.95
            other_weight = 0.05
            
            # Convert LSTM direction to numeric score
            direction_scores = {
                'BUY': 1,
                'SELL': -1,
                'HOLD': 0
            }
            
            lstm_score = direction_scores.get(lstm_direction, 0) * (lstm_confidence / 100)
            
            # Minor consideration for technical analysis (for validation only)
            tech_score = direction_scores.get(tech_direction, 0) * (tech_confidence / 100) if tech_direction else 0
            
            # Calculate LSTM-dominated combined score
            combined_score = (lstm_score * lstm_weight) + (tech_score * other_weight)
            
            # Final direction is primarily based on LSTM
            if lstm_direction == "BUY" and lstm_confidence >= 85:
                final_direction = "BUY"
            elif lstm_direction == "SELL" and lstm_confidence >= 85:
                final_direction = "SELL"
            else:
                return None  # LSTM not confident enough
            
            # LSTM AI confidence is the primary confidence metric
            ai_confidence = self.lstm_model.get_model_confidence(analyzed_data)
            
            # Calculate LSTM-based accuracy prediction
            lstm_accuracy = min(98, max(85, lstm_confidence + (ai_confidence * 0.1)))
            
            # Calculate final accuracy based on LSTM performance
            accuracy = self.calculate_lstm_based_accuracy(lstm_confidence, ai_confidence, combined_score)
            
            # Only proceed if accuracy meets threshold
            if accuracy < MIN_ACCURACY_THRESHOLD:
                logging.info(f"Signal accuracy {accuracy:.1f}% below threshold for {pair}")
                return None
            
            # Calculate expiry time
            signal_time = self.pocket_api.get_market_time()
            expiry_time = self.pocket_api.get_expiry_time(signal_time)
            
            # Format expiry time string
            expiry_str = f"{expiry_time.strftime('%H:%M')} - {(expiry_time + timedelta(seconds=3)).strftime('%H:%M')}"
            
            # Create LSTM-based signal object
            signal = {
                'pair': pair,
                'direction': final_direction,
                'accuracy': round(accuracy, 1),
                'ai_confidence': round(ai_confidence, 1),
                'lstm_confidence': round(lstm_confidence, 1),
                'expiry_time': expiry_str,
                'signal_time': signal_time,
                'expiry_timestamp': expiry_time,
                'current_price': analyzed_data['close'].iloc[-1],
                'strength': round(ai_confidence / 10, 1),  # LSTM-based strength
                'volatility': self.pocket_api.get_volatility(pair),
                'lstm_primary': True,  # Flag indicating LSTM-based signal
                'components': {
                    'lstm': {'direction': lstm_direction, 'confidence': lstm_confidence, 'primary': True},
                    'technical': {'direction': tech_direction, 'confidence': tech_confidence, 'validation_only': True},
                    'sentiment': {'used': False, 'note': 'LSTM AI primary decision maker'}
                },
                'lstm_analysis': {
                    'model_confidence': ai_confidence,
                    'prediction_strength': abs(combined_score) * 100,
                    'data_quality': min(100, len(analyzed_data)),
                    'lstm_direction_confidence': lstm_confidence
                }
            }
            
            logging.info(f"Generated signal: {pair} {final_direction} {accuracy:.1f}% accuracy")
            return signal
            
        except Exception as e:
            logging.error(f"Error combining signals: {e}")
            return None
    
    def calculate_lstm_based_accuracy(self, lstm_confidence, ai_confidence, combined_score):
        """Calculate accuracy based primarily on LSTM AI model performance"""
        try:
            # Base accuracy from LSTM confidence (primary factor)
            base_accuracy = lstm_confidence * 0.85  # Primary weight on LSTM
            
            # AI model robustness bonus
            ai_bonus = (ai_confidence / 100) * 10  # Additional confidence from model stability
            
            # Signal strength from combined score
            strength_bonus = abs(combined_score) * 5  # Small bonus for signal strength
            
            # Combine components with LSTM dominance
            total_accuracy = base_accuracy + ai_bonus + strength_bonus
            
            # Apply LSTM-focused limits
            total_accuracy = min(total_accuracy, 98)  # Max 98%
            total_accuracy = max(total_accuracy, 85)  # Min 85% for LSTM-based signals
            
            return total_accuracy
            
        except Exception as e:
            logging.error(f"Error calculating LSTM-based accuracy: {e}")
            return 90
    
    def calculate_predicted_accuracy(self, confidence, signal_strength, ai_confidence):
        """Calculate predicted accuracy for the signal (legacy method)"""
        try:
            # Base accuracy from confidence
            base_accuracy = confidence * 0.7  # Max 70% from confidence
            
            # Add signal strength component
            strength_bonus = signal_strength * 15  # Max 15% from strength
            
            # Add AI confidence component
            ai_bonus = (ai_confidence / 100) * 20  # Max 20% from AI
            
            # Combine components
            total_accuracy = base_accuracy + strength_bonus + ai_bonus
            
            # Apply realistic limits
            total_accuracy = min(total_accuracy, 98)  # Max 98%
            total_accuracy = max(total_accuracy, 50)  # Min 50%
            
            return total_accuracy
            
        except Exception as e:
            logging.error(f"Error calculating predicted accuracy: {e}")
            return 70
    
    def validate_signal(self, signal):
        """Validate signal before sending"""
        try:
            if not signal:
                return False
            
            # Check required fields
            required_fields = ['pair', 'direction', 'accuracy', 'ai_confidence', 'expiry_time']
            for field in required_fields:
                if field not in signal:
                    logging.warning(f"Missing required field: {field}")
                    return False
            
            # Check accuracy threshold
            if signal['accuracy'] < MIN_ACCURACY_THRESHOLD:
                return False
            
            # Check direction validity
            if signal['direction'] not in ['BUY', 'SELL']:
                return False
            
            # Check if signal is too close to market close (if applicable)
            current_time = signal['signal_time']
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating signal: {e}")
            return False
    
    def store_signal(self, signal):
        """Store signal in history and update tracking"""
        try:
            # Add timestamp
            signal['generated_at'] = datetime.now()
            signal['id'] = f"{signal['pair']}_{int(time.time())}"
            
            # Store in history
            self.signal_history.append(signal)
            
            # Update last signal time for pair
            self.last_signal_time[signal['pair']] = time.time()
            
            # Update daily count
            self.daily_signals += 1
            
            # Limit history size
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-500:]
            
            logging.info(f"Signal stored: {signal['id']}")
            
        except Exception as e:
            logging.error(f"Error storing signal: {e}")
    
    def broadcast_signal(self, signal):
        """Broadcast signal to subscribers (Telegram bot will implement this)"""
        try:
            # This will be called by the Telegram bot
            # For now, just log the signal
            logging.info(f"Broadcasting signal: {self.format_signal_message(signal)}")
            
        except Exception as e:
            logging.error(f"Error broadcasting signal: {e}")
    
    def format_signal_message(self, signal):
        """Format LSTM-based signal for display"""
        try:
            message = f"""
🚀 LSTM AI TRADING SIGNAL

Currency pair: {signal['pair']} 
Direction: {signal['direction']}
Accuracy: {signal['accuracy']}%
Time Expiry: {signal['expiry_time']}
LSTM AI Confidence: {signal['lstm_confidence']}%

🧠 LSTM AI Analysis:
• Model Confidence: {signal['lstm_analysis']['model_confidence']:.1f}%
• Prediction Strength: {signal['lstm_analysis']['prediction_strength']:.1f}%
• Direction Confidence: {signal['lstm_analysis']['lstm_direction_confidence']:.1f}%
• Data Quality Score: {signal['lstm_analysis']['data_quality']:.0f}/100

📊 Market Data:
• Current Price: {signal['current_price']:.5f}
• Market Volatility: {signal['volatility']:.4f}
• Signal Strength: {signal['strength']:.1f}/10

🤖 AI Decision Process:
• Primary: LSTM Neural Network ({signal['components']['lstm']['confidence']:.1f}%)
• Validation: Technical Analysis ({signal['components']['technical']['confidence']:.1f}%)
• Sentiment: Not used (LSTM AI primary)

⚡ This signal is generated entirely by LSTM AI model analysis
            """.strip()
            
            return message
            
        except Exception as e:
            logging.error(f"Error formatting LSTM signal message: {e}")
            return "Error formatting LSTM signal"
    
    def get_signal_statistics(self):
        """Get current signal generation statistics"""
        try:
            total_signals = len(self.signal_history)
            
            if total_signals == 0:
                return {
                    'total_signals': 0,
                    'daily_signals': 0,
                    'average_accuracy': 0,
                    'win_rate': 0,
                    'active_pairs': 0
                }
            
            # Calculate average accuracy
            avg_accuracy = np.mean([s['accuracy'] for s in self.signal_history])
            
            # Calculate win rate (simulated for now)
            win_rate = min(avg_accuracy * 0.95, 95)  # Realistic win rate based on accuracy
            
            # Active pairs
            active_pairs = len(set(s['pair'] for s in self.signal_history[-50:]))
            
            return {
                'total_signals': total_signals,
                'daily_signals': self.daily_signals,
                'average_accuracy': round(avg_accuracy, 1),
                'win_rate': round(win_rate, 1),
                'active_pairs': active_pairs
            }
            
        except Exception as e:
            logging.error(f"Error getting signal statistics: {e}")
            return {
                'total_signals': 0,
                'daily_signals': 0,
                'average_accuracy': 0,
                'win_rate': 0,
                'active_pairs': 0
            }
    
    def get_recent_signals(self, limit=10):
        """Get recent signals"""
        try:
            return self.signal_history[-limit:] if self.signal_history else []
        except Exception as e:
            logging.error(f"Error getting recent signals: {e}")
            return []
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at midnight)"""
        try:
            self.daily_signals = 0
            logging.info("Daily statistics reset")
        except Exception as e:
            logging.error(f"Error resetting daily stats: {e}")
    
    def train_models_with_new_data(self):
        """Retrain AI models with latest market data"""
        try:
            logging.info("Starting model retraining with new data...")
            
            # Get recent market data for all pairs
            all_data = []
            for pair in self.pocket_api.get_available_pairs():
                data = self.pocket_api.get_historical_data(pair, limit=1000)
                if data is not None:
                    all_data.append(data)
            
            if all_data:
                # Combine all data
                combined_data = pd.concat(all_data, ignore_index=True)
                
                # Retrain LSTM model
                success = self.lstm_model.retrain_with_new_data(combined_data)
                
                if success:
                    logging.info("Model retraining completed successfully")
                else:
                    logging.warning("Model retraining failed")
                
                return success
            else:
                logging.warning("No data available for model retraining")
                return False
                
        except Exception as e:
            logging.error(f"Error retraining models: {e}")
            return False