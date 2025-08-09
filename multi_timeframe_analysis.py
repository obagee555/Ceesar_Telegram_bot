import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from technical_analysis import TechnicalAnalysis
from market_microstructure import MarketMicrostructure

class MultiTimeframeAnalysis:
    def __init__(self):
        self.technical_analyzer = TechnicalAnalysis()
        self.microstructure_analyzer = MarketMicrostructure()
        
        # Define timeframes and their weights
        self.timeframes = {
            '1m': {'period': '1min', 'weight': 0.1, 'description': 'Scalping'},
            '5m': {'period': '5min', 'weight': 0.2, 'description': 'Short-term'},
            '15m': {'period': '15min', 'weight': 0.25, 'description': 'Medium-term'},
            '1h': {'period': '1hour', 'weight': 0.25, 'description': 'Trend confirmation'},
            '4h': {'period': '4hour', 'weight': 0.15, 'description': 'Major trend'},
            '1d': {'period': '1day', 'weight': 0.05, 'description': 'Long-term bias'}
        }
        
        self.analysis_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
    def analyze_all_timeframes(self, pair: str, market_data_func) -> Dict:
        """Analyze all timeframes for a given currency pair"""
        try:
            logging.info(f"Starting multi-timeframe analysis for {pair}")
            
            timeframe_analysis = {}
            weighted_signals = []
            
            for tf, config in self.timeframes.items():
                try:
                    # Get market data for this timeframe
                    tf_data = self.get_timeframe_data(pair, config['period'], market_data_func)
                    
                    if tf_data is not None and len(tf_data) > 50:
                        # Analyze this timeframe
                        analysis = self.analyze_timeframe(tf_data, tf, config)
                        timeframe_analysis[tf] = analysis
                        
                        # Add weighted signal
                        if analysis['signal_strength'] > 0:
                            weighted_signals.append({
                                'timeframe': tf,
                                'direction': analysis['direction'],
                                'strength': analysis['signal_strength'] * config['weight'],
                                'confidence': analysis['confidence'],
                                'weight': config['weight']
                            })
                    
                except Exception as e:
                    logging.warning(f"Error analyzing {tf} timeframe for {pair}: {e}")
                    continue
            
            # Calculate combined analysis
            combined_analysis = self.calculate_combined_analysis(timeframe_analysis, weighted_signals)
            
            return {
                'timeframe_analysis': timeframe_analysis,
                'combined_analysis': combined_analysis,
                'weighted_signals': weighted_signals,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error in multi-timeframe analysis: {e}")
            return self.get_default_mtf_analysis()
    
    def get_timeframe_data(self, pair: str, period: str, market_data_func) -> Optional[pd.DataFrame]:
        """Get market data for specific timeframe"""
        try:
            # This would integrate with your Pocket Option API
            # For now, we'll simulate different timeframes from 1-minute data
            base_data = market_data_func(pair)
            
            if base_data is None or len(base_data) < 100:
                return None
            
            # Resample data to different timeframes
            if period == '1min':
                return base_data
            elif period == '5min':
                return self.resample_data(base_data, '5T')
            elif period == '15min':
                return self.resample_data(base_data, '15T')
            elif period == '1hour':
                return self.resample_data(base_data, '1H')
            elif period == '4hour':
                return self.resample_data(base_data, '4H')
            elif period == '1day':
                return self.resample_data(base_data, '1D')
            else:
                return base_data
                
        except Exception as e:
            logging.error(f"Error getting timeframe data: {e}")
            return None
    
    def resample_data(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """Resample 1-minute data to different timeframes"""
        try:
            # Ensure datetime index
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            
            # Resample OHLCV data
            resampled = data.resample(period).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled.reset_index()
            
        except Exception as e:
            logging.error(f"Error resampling data: {e}")
            return data
    
    def analyze_timeframe(self, data: pd.DataFrame, timeframe: str, config: Dict) -> Dict:
        """Analyze a single timeframe"""
        try:
            # Technical analysis
            tech_analysis = self.technical_analyzer.calculate_all_indicators(data)
            
            if tech_analysis is None:
                return self.get_default_timeframe_analysis(timeframe)
            
            # Market microstructure analysis
            volume_data = pd.DataFrame({'volume': data['volume'] if 'volume' in data.columns else [1000] * len(data)})
            microstructure = self.microstructure_analyzer.analyze_order_flow(data, volume_data)
            
            # Get signal direction and strength
            signal_strength = self.technical_analyzer.get_signal_strength(tech_analysis)
            direction = self.technical_analyzer.get_trade_direction(tech_analysis)
            
            # Calculate confidence based on multiple factors
            confidence = self.calculate_timeframe_confidence(tech_analysis, microstructure, signal_strength)
            
            # Trend analysis
            trend_analysis = self.analyze_trend_strength(tech_analysis, timeframe)
            
            return {
                'timeframe': timeframe,
                'direction': direction,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'trend_strength': trend_analysis['strength'],
                'trend_direction': trend_analysis['direction'],
                'support_levels': self.find_support_levels(tech_analysis),
                'resistance_levels': self.find_resistance_levels(tech_analysis),
                'volatility': self.calculate_volatility(tech_analysis),
                'vwap_analysis': microstructure['vwap'],
                'smart_money': microstructure['smart_money_flow'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error analyzing timeframe {timeframe}: {e}")
            return self.get_default_timeframe_analysis(timeframe)
    
    def calculate_timeframe_confidence(self, tech_analysis: pd.DataFrame, microstructure: Dict, signal_strength: float) -> float:
        """Calculate confidence score for a timeframe"""
        try:
            confidence_factors = []
            
            # Technical indicators confidence
            if 'rsi' in tech_analysis.columns:
                rsi = tech_analysis['rsi'].iloc[-1]
                if rsi < 30 or rsi > 70:
                    confidence_factors.append(0.8)
                elif rsi < 40 or rsi > 60:
                    confidence_factors.append(0.6)
                else:
                    confidence_factors.append(0.3)
            
            if 'macd' in tech_analysis.columns and 'macd_signal' in tech_analysis.columns:
                macd = tech_analysis['macd'].iloc[-1]
                macd_signal = tech_analysis['macd_signal'].iloc[-1]
                macd_diff = abs(macd - macd_signal)
                if macd_diff > 0.001:
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.4)
            
            # Smart money confidence
            smart_money_confidence = microstructure['smart_money_flow']['confidence'] / 100
            confidence_factors.append(smart_money_confidence)
            
            # Signal strength confidence
            confidence_factors.append(min(signal_strength / 100, 1.0))
            
            # VWAP confidence
            vwap_analysis = microstructure['vwap']
            if vwap_analysis['price_vs_vwap'] == 'above':
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            # Calculate average confidence
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors) * 100
            else:
                return 50.0
                
        except Exception as e:
            logging.error(f"Error calculating timeframe confidence: {e}")
            return 50.0
    
    def analyze_trend_strength(self, tech_analysis: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze trend strength for a timeframe"""
        try:
            # Calculate trend strength using multiple indicators
            trend_score = 0
            trend_direction = 'neutral'
            
            # Moving averages trend
            if 'sma_5' in tech_analysis.columns and 'sma_20' in tech_analysis.columns:
                sma_5 = tech_analysis['sma_5'].iloc[-1]
                sma_20 = tech_analysis['sma_20'].iloc[-1]
                
                if sma_5 > sma_20:
                    trend_score += 1
                    trend_direction = 'bullish'
                else:
                    trend_score -= 1
                    trend_direction = 'bearish'
            
            # ADX trend strength
            if 'adx' in tech_analysis.columns:
                adx = tech_analysis['adx'].iloc[-1]
                if adx > 25:
                    trend_score += 1
                elif adx < 20:
                    trend_score -= 1
            
            # Price action trend
            if len(tech_analysis) >= 20:
                recent_highs = tech_analysis['high'].tail(10).max()
                recent_lows = tech_analysis['low'].tail(10).min()
                current_price = tech_analysis['close'].iloc[-1]
                
                if current_price > (recent_highs + recent_lows) / 2:
                    trend_score += 1
                else:
                    trend_score -= 1
            
            # Normalize trend strength
            trend_strength = min(abs(trend_score) / 3 * 100, 100)
            
            return {
                'strength': trend_strength,
                'direction': trend_direction,
                'score': trend_score
            }
            
        except Exception as e:
            logging.error(f"Error analyzing trend strength: {e}")
            return {'strength': 50, 'direction': 'neutral', 'score': 0}
    
    def find_support_levels(self, tech_analysis: pd.DataFrame) -> List[float]:
        """Find support levels using technical analysis"""
        try:
            support_levels = []
            
            # Use Bollinger Bands lower band
            if 'bb_lower' in tech_analysis.columns:
                support_levels.append(tech_analysis['bb_lower'].iloc[-1])
            
            # Use recent lows
            if len(tech_analysis) >= 20:
                recent_low = tech_analysis['low'].tail(20).min()
                support_levels.append(recent_low)
            
            # Use moving averages as support
            if 'sma_20' in tech_analysis.columns:
                support_levels.append(tech_analysis['sma_20'].iloc[-1])
            
            return sorted(list(set(support_levels)), reverse=True)
            
        except Exception as e:
            logging.error(f"Error finding support levels: {e}")
            return []
    
    def find_resistance_levels(self, tech_analysis: pd.DataFrame) -> List[float]:
        """Find resistance levels using technical analysis"""
        try:
            resistance_levels = []
            
            # Use Bollinger Bands upper band
            if 'bb_upper' in tech_analysis.columns:
                resistance_levels.append(tech_analysis['bb_upper'].iloc[-1])
            
            # Use recent highs
            if len(tech_analysis) >= 20:
                recent_high = tech_analysis['high'].tail(20).max()
                resistance_levels.append(recent_high)
            
            # Use moving averages as resistance
            if 'sma_20' in tech_analysis.columns:
                resistance_levels.append(tech_analysis['sma_20'].iloc[-1])
            
            return sorted(list(set(resistance_levels)))
            
        except Exception as e:
            logging.error(f"Error finding resistance levels: {e}")
            return []
    
    def calculate_volatility(self, tech_analysis: pd.DataFrame) -> float:
        """Calculate volatility for the timeframe"""
        try:
            if 'atr' in tech_analysis.columns:
                return tech_analysis['atr'].iloc[-1]
            elif len(tech_analysis) >= 20:
                returns = tech_analysis['close'].pct_change().dropna()
                return returns.std() * 100
            else:
                return 0.0
                
        except Exception as e:
            logging.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def calculate_combined_analysis(self, timeframe_analysis: Dict, weighted_signals: List[Dict]) -> Dict:
        """Calculate combined analysis from all timeframes"""
        try:
            if not weighted_signals:
                return self.get_default_combined_analysis()
            
            # Calculate weighted direction
            bullish_weight = sum([s['strength'] for s in weighted_signals if s['direction'] == 'bullish'])
            bearish_weight = sum([s['strength'] for s in weighted_signals if s['direction'] == 'bearish'])
            
            if bullish_weight > bearish_weight:
                combined_direction = 'bullish'
                direction_strength = bullish_weight / (bullish_weight + bearish_weight) * 100
            else:
                combined_direction = 'bearish'
                direction_strength = bearish_weight / (bullish_weight + bearish_weight) * 100
            
            # Calculate average confidence
            avg_confidence = sum([s['confidence'] * s['weight'] for s in weighted_signals]) / sum([s['weight'] for s in weighted_signals])
            
            # Calculate timeframe alignment
            bullish_timeframes = [s['timeframe'] for s in weighted_signals if s['direction'] == 'bullish']
            bearish_timeframes = [s['timeframe'] for s in weighted_signals if s['direction'] == 'bearish']
            
            alignment_score = len(bullish_timeframes) if combined_direction == 'bullish' else len(bearish_timeframes)
            max_alignment = len(self.timeframes)
            alignment_percentage = (alignment_score / max_alignment) * 100
            
            return {
                'direction': combined_direction,
                'direction_strength': direction_strength,
                'confidence': avg_confidence,
                'alignment_score': alignment_score,
                'alignment_percentage': alignment_percentage,
                'bullish_timeframes': bullish_timeframes,
                'bearish_timeframes': bearish_timeframes,
                'signal_quality': 'high' if avg_confidence > 70 and alignment_percentage > 60 else 'medium' if avg_confidence > 50 else 'low'
            }
            
        except Exception as e:
            logging.error(f"Error calculating combined analysis: {e}")
            return self.get_default_combined_analysis()
    
    def get_default_mtf_analysis(self) -> Dict:
        """Return default multi-timeframe analysis"""
        return {
            'timeframe_analysis': {},
            'combined_analysis': self.get_default_combined_analysis(),
            'weighted_signals': [],
            'timestamp': datetime.now()
        }
    
    def get_default_timeframe_analysis(self, timeframe: str) -> Dict:
        """Return default timeframe analysis"""
        return {
            'timeframe': timeframe,
            'direction': 'neutral',
            'signal_strength': 0,
            'confidence': 50,
            'trend_strength': 50,
            'trend_direction': 'neutral',
            'support_levels': [],
            'resistance_levels': [],
            'volatility': 0,
            'vwap_analysis': {'vwap': 0, 'vwap_upper': 0, 'vwap_lower': 0},
            'smart_money': {'smart_money_bias': 'neutral', 'confidence': 0},
            'timestamp': datetime.now()
        }
    
    def get_default_combined_analysis(self) -> Dict:
        """Return default combined analysis"""
        return {
            'direction': 'neutral',
            'direction_strength': 50,
            'confidence': 50,
            'alignment_score': 0,
            'alignment_percentage': 0,
            'bullish_timeframes': [],
            'bearish_timeframes': [],
            'signal_quality': 'low'
        }