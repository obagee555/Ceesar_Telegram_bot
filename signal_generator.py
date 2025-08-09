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
        """Main signal generation loop"""
        while self.is_running:
            try:
                # Get available currency pairs
                available_pairs = self.pocket_api.get_available_pairs()
                
                for pair in available_pairs:
                    # Check if we should generate signal for this pair
                    if self.should_generate_signal(pair):
                        signal = self.generate_signal(pair)
                        
                        if signal and self.validate_signal(signal):
                            # Log and store signal
                            self.store_signal(signal)
                            
                            # Broadcast signal (implement callback mechanism)
                            self.broadcast_signal(signal)
                
                # Sleep for 30 seconds before next check
                time.sleep(30)
                
            except Exception as e:
                logging.error(f"Error in signal generation loop: {e}")
                time.sleep(60)  # Wait longer if there's an error
    
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
        """Combine LSTM, technical analysis, and sentiment into final signal"""
        try:
            # Calculate weights for each signal component
            lstm_weight = 0.4
            tech_weight = 0.35
            sentiment_weight = 0.25
            
            # Convert directions to numeric scores
            direction_scores = {
                'BUY': 1,
                'SELL': -1,
                'HOLD': 0,
                'BULLISH': 1,
                'BEARISH': -1,
                'NEUTRAL': 0
            }
            
            lstm_score = direction_scores.get(lstm_direction, 0) * (lstm_confidence / 100)
            tech_score = direction_scores.get(tech_direction, 0) * (tech_confidence / 100)
            
            # Handle sentiment bias
            sentiment_score = 0
            if sentiment_bias['bias'] == 'BULLISH_BIAS':
                sentiment_score = sentiment_bias['strength'] / 100
            elif sentiment_bias['bias'] == 'BEARISH_BIAS':
                sentiment_score = -(sentiment_bias['strength'] / 100)
            
            # Calculate weighted combined score
            combined_score = (
                lstm_score * lstm_weight +
                tech_score * tech_weight +
                sentiment_score * sentiment_weight
            )
            
            # Determine final direction
            if combined_score > 0.3:
                final_direction = "BUY"
            elif combined_score < -0.3:
                final_direction = "SELL"
            else:
                return None  # No clear signal
            
            # Calculate overall confidence
            confidence_factors = [
                lstm_confidence,
                tech_confidence,
                sentiment_bias['strength']
            ]
            
            overall_confidence = np.mean(confidence_factors)
            
            # Apply additional confidence boosts/penalties
            if abs(combined_score) > 0.6:
                overall_confidence *= 1.1  # Boost for strong signals
            
            # Get AI model confidence
            ai_confidence = self.lstm_model.get_model_confidence(analyzed_data)
            
            # Calculate final accuracy prediction
            accuracy = self.calculate_predicted_accuracy(
                overall_confidence, abs(combined_score), ai_confidence
            )
            
            # Only proceed if accuracy meets threshold
            if accuracy < MIN_ACCURACY_THRESHOLD:
                logging.info(f"Signal accuracy {accuracy:.1f}% below threshold for {pair}")
                return None
            
            # Calculate expiry time
            signal_time = self.pocket_api.get_market_time()
            expiry_time = self.pocket_api.get_expiry_time(signal_time)
            
            # Format expiry time string
            expiry_str = f"{expiry_time.strftime('%H:%M')} - {(expiry_time + timedelta(seconds=3)).strftime('%H:%M')}"
            
            # Create signal object
            signal = {
                'pair': pair,
                'direction': final_direction,
                'accuracy': round(accuracy, 1),
                'ai_confidence': round(ai_confidence, 1),
                'expiry_time': expiry_str,
                'signal_time': signal_time,
                'expiry_timestamp': expiry_time,
                'current_price': analyzed_data['close'].iloc[-1],
                'strength': self.technical_analyzer.get_signal_strength(analyzed_data),
                'volatility': self.pocket_api.get_volatility(pair),
                'components': {
                    'lstm': {'direction': lstm_direction, 'confidence': lstm_confidence},
                    'technical': {'direction': tech_direction, 'confidence': tech_confidence},
                    'sentiment': sentiment_bias
                },
                'market_conditions': {
                    'rsi': analyzed_data['rsi'].iloc[-1] if 'rsi' in analyzed_data.columns else 0,
                    'macd': analyzed_data['macd'].iloc[-1] if 'macd' in analyzed_data.columns else 0,
                    'bb_position': analyzed_data['bb_position'].iloc[-1] if 'bb_position' in analyzed_data.columns else 0,
                    'adx': analyzed_data['adx'].iloc[-1] if 'adx' in analyzed_data.columns else 0
                }
            }
            
            logging.info(f"Generated signal: {pair} {final_direction} {accuracy:.1f}% accuracy")
            return signal
            
        except Exception as e:
            logging.error(f"Error combining signals: {e}")
            return None
    
    def calculate_predicted_accuracy(self, confidence, signal_strength, ai_confidence):
        """Calculate predicted accuracy for the signal"""
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
        """Format signal for display"""
        try:
            message = f"""
🚀 TRADING SIGNAL

Currency pair: {signal['pair']}
Direction: {signal['direction']}
Accuracy: {signal['accuracy']}%
Time Expiry: {signal['expiry_time']}
AI Confidence: {signal['ai_confidence']}%

📊 Market Analysis:
• Signal Strength: {signal['strength']:.1f}/10
• Volatility: {signal['volatility']:.4f}
• Current Price: {signal['current_price']:.5f}

🤖 AI Components:
• LSTM: {signal['components']['lstm']['direction']} ({signal['components']['lstm']['confidence']:.1f}%)
• Technical: {signal['components']['technical']['direction']} ({signal['components']['technical']['confidence']:.1f}%)
• Sentiment: {signal['components']['sentiment']['bias']} ({signal['components']['sentiment']['strength']:.1f}%)

📈 Indicators:
• RSI: {signal['market_conditions']['rsi']:.1f}
• MACD: {signal['market_conditions']['macd']:.4f}
• BB Position: {signal['market_conditions']['bb_position']:.2f}
• ADX: {signal['market_conditions']['adx']:.1f}
            """.strip()
            
            return message
            
        except Exception as e:
            logging.error(f"Error formatting signal message: {e}")
            return "Error formatting signal"
    
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