import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
import threading
from typing import Dict, List, Optional, Tuple
import json

# Import simplified components
from simplified_lstm_model import SimplifiedLSTMModel
from technical_analysis import TechnicalAnalysis
from sentiment_analysis import SentimentAnalysis
from risk_management import RiskManager
from performance_tracker import PerformanceTracker
from demo_pocket_option_api import DemoPocketOptionAPI
from config import *

class SimplifiedSignalGenerator:
    def __init__(self):
        # Initialize simplified components
        self.lstm_model = SimplifiedLSTMModel()
        self.technical_analyzer = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalysis()
        self.risk_manager = RiskManager()
        self.performance_tracker = PerformanceTracker()
        self.pocket_api = DemoPocketOptionAPI()
        
        # Signal generation state
        self.is_running = False
        self.last_signal_time = {}
        self.signal_history = []
        self.min_signal_interval = 300  # 5 minutes between signals
        
        # Performance tracking
        self.daily_signals = 0
        self.accuracy_tracker = []
        
        # Signal quality thresholds
        self.min_confidence_threshold = 80.0
        self.min_confluence_score = 0.6
        self.max_risk_score = 0.4
        
    def start_signal_generation(self):
        """Start the simplified signal generation process"""
        try:
            logging.info("🚀 Starting Simplified Signal Generation Engine...")
            
            # Start Pocket Option API
            if not self.pocket_api.start():
                logging.error("Failed to start Pocket Option API")
                return False
            
            # Start performance tracking
            self.performance_tracker.start_tracking()
            
            # Start risk management
            self.risk_manager.set_risk_level("MODERATE")
            
            self.is_running = True
            
            # Start signal generation thread
            signal_thread = threading.Thread(target=self.signal_generation_loop)
            signal_thread.daemon = True
            signal_thread.start()
            
            logging.info("✅ Simplified Signal Generation Engine started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error starting simplified signal generation: {e}")
            return False
    
    def stop_signal_generation(self):
        """Stop the simplified signal generation process"""
        try:
            self.is_running = False
            self.pocket_api.stop()
            self.performance_tracker.stop_tracking()
            logging.info("🛑 Simplified Signal Generation Engine stopped")
            
        except Exception as e:
            logging.error(f"Error stopping simplified signal generation: {e}")
    
    def signal_generation_loop(self):
        """Main simplified signal generation loop"""
        while self.is_running:
            try:
                # Get available currency pairs
                available_pairs = self.pocket_api.get_available_pairs()
                
                if not available_pairs:
                    logging.warning("No available pairs found")
                    time.sleep(60)
                    continue
                
                # Simplified LSTM AI selects the best currency pair
                best_pair, pair_analysis = self.lstm_model.analyze_best_currency_pair(
                    available_pairs, self.pocket_api.get_market_data_df
                )
                
                if best_pair and self.should_generate_signal(best_pair):
                    # Generate simplified signal
                    signal = self.generate_simplified_signal(best_pair, pair_analysis)
                    
                    if signal and self.validate_simplified_signal(signal):
                        # Store and broadcast signal
                        self.store_signal(signal)
                        self.broadcast_signal(signal)
                        
                        # Update performance tracking
                        self.performance_tracker.record_signal(signal)
                        
                        # Wait between signals
                        time.sleep(self.min_signal_interval)
                
                # Sleep before next analysis
                time.sleep(60)
                
            except Exception as e:
                logging.error(f"Error in simplified signal generation loop: {e}")
                time.sleep(60)
    
    def generate_simplified_signal(self, pair: str, pair_analysis: Dict) -> Optional[Dict]:
        """Generate simplified signal using core components"""
        try:
            logging.info(f"🔍 Generating simplified signal for {pair}")
            
            # Get market data
            market_data = self.pocket_api.get_market_data_df(pair)
            if market_data is None or len(market_data) < 100:
                return None
            
            # 1. Technical analysis
            technical_analysis = self.technical_analyzer.calculate_all_indicators(market_data)
            
            # 2. Sentiment analysis
            sentiment_analysis = self.sentiment_analyzer.get_comprehensive_sentiment(pair)
            
            # 3. Simplified LSTM prediction
            lstm_prediction = self.lstm_model.predict_direction(market_data)
            
            # 4. Combine analyses
            combined_analysis = self.combine_simplified_analyses(
                technical_analysis, sentiment_analysis, lstm_prediction, pair_analysis
            )
            
            # 5. Calculate final signal
            signal = self.calculate_final_signal(combined_analysis, pair, market_data)
            
            # 6. Risk assessment
            risk_assessment = self.risk_manager.evaluate_signal_risk(signal)
            
            # Add risk assessment to signal
            signal['risk_assessment'] = risk_assessment
            
            return signal
            
        except Exception as e:
            logging.error(f"Error generating simplified signal: {e}")
            return None
    
    def combine_simplified_analyses(self, technical_analysis: pd.DataFrame, 
                                  sentiment_analysis: Dict, lstm_prediction: Dict,
                                  pair_analysis: Dict) -> Dict:
        """Combine simplified analysis components"""
        try:
            # Calculate weighted scores for each component
            scores = {}
            
            # 1. Technical analysis (40% weight)
            if technical_analysis is not None:
                tech_strength = self.technical_analyzer.get_signal_strength(technical_analysis)
                tech_direction = self.technical_analyzer.get_trade_direction(technical_analysis)
                scores['technical'] = {
                    'score': tech_strength / 100,
                    'weight': 0.40,
                    'direction': tech_direction,
                    'confidence': tech_strength
                }
            else:
                scores['technical'] = {
                    'score': 0.5,
                    'weight': 0.40,
                    'direction': 'neutral',
                    'confidence': 50
                }
            
            # 2. Sentiment analysis (20% weight)
            sentiment_score = sentiment_analysis.get('confidence', 50) / 100
            scores['sentiment'] = {
                'score': sentiment_score,
                'weight': 0.20,
                'direction': sentiment_analysis.get('overall_bias', 'neutral'),
                'confidence': sentiment_analysis.get('confidence', 50)
            }
            
            # 3. Simplified LSTM (40% weight)
            lstm_score = lstm_prediction.get('confidence', 0) / 100
            scores['lstm'] = {
                'score': lstm_score,
                'weight': 0.40,
                'direction': lstm_prediction.get('direction', 'neutral'),
                'confidence': lstm_prediction.get('confidence', 0)
            }
            
            # Calculate weighted average
            total_score = 0
            total_weight = 0
            direction_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            
            for component, data in scores.items():
                total_score += data['score'] * data['weight']
                total_weight += data['weight']
                direction_votes[data['direction']] += data['weight']
            
            weighted_score = total_score / total_weight if total_weight > 0 else 0
            
            # Determine overall direction
            overall_direction = max(direction_votes, key=direction_votes.get)
            
            # Calculate confidence
            confidence = self.calculate_overall_confidence(scores)
            
            return {
                'component_scores': scores,
                'weighted_score': weighted_score,
                'overall_direction': overall_direction,
                'confidence': confidence,
                'direction_votes': direction_votes,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error combining simplified analyses: {e}")
            return self.get_default_combined_analysis()
    
    def calculate_overall_confidence(self, scores: Dict) -> float:
        """Calculate overall confidence based on component scores"""
        try:
            confidence_scores = []
            weights = []
            
            for component, data in scores.items():
                confidence_scores.append(data['confidence'])
                weights.append(data['weight'])
            
            # Calculate weighted average confidence
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_confidence = sum(c * w for c, w in zip(confidence_scores, weights)) / total_weight
                return min(weighted_confidence, 100)
            else:
                return 50.0
                
        except Exception as e:
            logging.error(f"Error calculating overall confidence: {e}")
            return 50.0
    
    def calculate_final_signal(self, combined_analysis: Dict, pair: str, market_data: pd.DataFrame) -> Dict:
        """Calculate final trading signal"""
        try:
            current_price = market_data['close'].iloc[-1]
            weighted_score = combined_analysis['weighted_score']
            overall_direction = combined_analysis['overall_direction']
            confidence = combined_analysis['confidence']
            
            # Calculate signal strength
            signal_strength = weighted_score * 100
            
            # Determine entry and target prices
            if overall_direction == 'bullish':
                entry_price = current_price * 1.001  # Slightly above current price
                target_price = current_price * 1.005  # 0.5% target
                stop_loss = current_price * 0.998  # 0.2% stop loss
            elif overall_direction == 'bearish':
                entry_price = current_price * 0.999  # Slightly below current price
                target_price = current_price * 0.995  # 0.5% target
                stop_loss = current_price * 1.002  # 0.2% stop loss
            else:
                return None  # No signal for neutral direction
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Generate signal ID
            signal_id = f"SIM_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            signal = {
                'id': signal_id,
                'pair': pair,
                'direction': overall_direction,
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'risk_reward_ratio': risk_reward_ratio,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'timestamp': datetime.now(),
                'expiry_time': EXPIRY_TIME,
                'analysis_summary': {
                    'weighted_score': weighted_score,
                    'component_scores': combined_analysis['component_scores'],
                    'direction_votes': combined_analysis['direction_votes']
                }
            }
            
            return signal
            
        except Exception as e:
            logging.error(f"Error calculating final signal: {e}")
            return None
    
    def validate_simplified_signal(self, signal: Dict) -> bool:
        """Validate simplified signal quality"""
        try:
            if not signal:
                return False
            
            # Check confidence threshold
            if signal['confidence'] < self.min_confidence_threshold:
                logging.info(f"Signal confidence {signal['confidence']:.1f}% below threshold {self.min_confidence_threshold}%")
                return False
            
            # Check signal strength
            if signal['signal_strength'] < 60:
                logging.info(f"Signal strength {signal['signal_strength']:.1f} below minimum 60")
                return False
            
            # Check risk-reward ratio
            if signal['risk_reward_ratio'] < 1.5:
                logging.info(f"Risk-reward ratio {signal['risk_reward_ratio']:.2f} below minimum 1.5")
                return False
            
            # Check risk assessment
            if 'risk_assessment' in signal:
                risk_score = signal['risk_assessment'].get('risk_score', 0)
                if risk_score > self.max_risk_score:
                    logging.info(f"Risk score {risk_score:.2f} above maximum {self.max_risk_score}")
                    return False
            
            # Check if trading is halted
            if self.risk_manager.should_halt_trading():
                logging.info("Trading halted by risk manager")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating simplified signal: {e}")
            return False
    
    def should_generate_signal(self, pair: str) -> bool:
        """Check if we should generate a signal for this pair"""
        try:
            # Check time interval
            current_time = datetime.now()
            last_signal_time = self.last_signal_time.get(pair)
            
            if last_signal_time:
                time_diff = (current_time - last_signal_time).total_seconds()
                if time_diff < self.min_signal_interval:
                    return False
            
            # Check daily signal limit
            if self.daily_signals >= 15:  # Maximum 15 signals per day
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking signal generation: {e}")
            return False
    
    def store_signal(self, signal: Dict):
        """Store signal in history"""
        try:
            self.signal_history.append(signal)
            self.last_signal_time[signal['pair']] = datetime.now()
            self.daily_signals += 1
            
            # Keep only last 100 signals
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
            
            logging.info(f"📊 Signal stored: {signal['pair']} {signal['direction']} "
                        f"(Confidence: {signal['confidence']:.1f}%, Strength: {signal['signal_strength']:.1f})")
            
        except Exception as e:
            logging.error(f"Error storing signal: {e}")
    
    def broadcast_signal(self, signal: Dict):
        """Broadcast signal to Telegram and other channels"""
        try:
            # Format signal message
            message = self.format_simplified_signal_message(signal)
            
            # Send to Telegram (integrate with your telegram bot)
            # self.telegram_bot.send_signal(message)
            
            # Log signal
            logging.info(f"📡 Signal broadcasted: {signal['id']}")
            
        except Exception as e:
            logging.error(f"Error broadcasting signal: {e}")
    
    def format_simplified_signal_message(self, signal: Dict) -> str:
        """Format simplified signal message"""
        try:
            analysis = signal['analysis_summary']
            
            message = f"""
🚀 **SIMPLIFIED SIGNAL ALERT** 🚀

📊 **Pair:** {signal['pair']}
🎯 **Direction:** {signal['direction'].upper()}
💰 **Entry:** {signal['entry_price']:.5f}
🎯 **Target:** {signal['target_price']:.5f}
🛑 **Stop Loss:** {signal['stop_loss']:.5f}
⚖️ **Risk/Reward:** {signal['risk_reward_ratio']:.2f}

📈 **Confidence:** {signal['confidence']:.1f}%
💪 **Signal Strength:** {signal['signal_strength']:.1f}%

🔍 **Analysis Breakdown:**
• Technical Analysis: {analysis['component_scores']['technical']['confidence']:.1f}%
• Sentiment Analysis: {analysis['component_scores']['sentiment']['confidence']:.1f}%
• Simplified LSTM: {analysis['component_scores']['lstm']['confidence']:.1f}%

⏰ **Expiry:** {signal['expiry_time']} seconds
🆔 **Signal ID:** {signal['id']}

🎯 **Enhanced AI-Powered Signal**
            """
            
            return message.strip()
            
        except Exception as e:
            logging.error(f"Error formatting signal message: {e}")
            return f"Signal: {signal['pair']} {signal['direction']}"
    
    def get_default_combined_analysis(self) -> Dict:
        """Return default combined analysis"""
        return {
            'component_scores': {},
            'weighted_score': 0.5,
            'overall_direction': 'neutral',
            'confidence': 50.0,
            'direction_votes': {'bullish': 0, 'bearish': 0, 'neutral': 1},
            'timestamp': datetime.now()
        }