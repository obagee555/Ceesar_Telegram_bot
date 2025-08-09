import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, ADXIndicator, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeSMAIndicator
import logging
from config import TECHNICAL_INDICATORS

class TechnicalAnalysis:
    def __init__(self):
        self.indicators = TECHNICAL_INDICATORS
        
    def calculate_all_indicators(self, df):
        """Calculate all technical indicators for given data"""
        try:
            if df is None or len(df) < 50:
                logging.warning("Insufficient data for technical analysis")
                return None
            
            # Ensure we have required columns
            if 'close' not in df.columns:
                logging.error("Close price column not found")
                return None
            
            # Create a copy to avoid modifying original data
            result_df = df.copy()
            
            # Add missing columns if not present
            if 'high' not in result_df.columns:
                result_df['high'] = result_df['close'] * 1.001
            if 'low' not in result_df.columns:
                result_df['low'] = result_df['close'] * 0.999
            if 'volume' not in result_df.columns:
                result_df['volume'] = 1000  # Default volume
            
            # Calculate RSI
            result_df = self.calculate_rsi(result_df)
            
            # Calculate MACD
            result_df = self.calculate_macd(result_df)
            
            # Calculate Bollinger Bands
            result_df = self.calculate_bollinger_bands(result_df)
            
            # Calculate Moving Averages
            result_df = self.calculate_moving_averages(result_df)
            
            # Calculate Stochastic Oscillator
            result_df = self.calculate_stochastic(result_df)
            
            # Calculate ATR
            result_df = self.calculate_atr(result_df)
            
            # Calculate ADX
            result_df = self.calculate_adx(result_df)
            
            # Calculate Support and Resistance
            result_df = self.calculate_support_resistance(result_df)
            
            # Calculate Trend Strength
            result_df = self.calculate_trend_strength(result_df)
            
            return result_df
            
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            return None
    
    def calculate_rsi(self, df):
        """Calculate Relative Strength Index"""
        try:
            rsi_period = self.indicators['RSI']['period']
            rsi = RSIIndicator(close=df['close'], window=rsi_period)
            df['rsi'] = rsi.rsi()
            return df
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            return df
    
    def calculate_macd(self, df):
        """Calculate MACD indicator"""
        try:
            fast = self.indicators['MACD']['fast']
            slow = self.indicators['MACD']['slow']
            signal = self.indicators['MACD']['signal']
            
            macd = MACD(close=df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            return df
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            return df
    
    def calculate_bollinger_bands(self, df):
        """Calculate Bollinger Bands"""
        try:
            period = self.indicators['BB']['period']
            std = self.indicators['BB']['std']
            
            bb = BollingerBands(close=df['close'], window=period, window_dev=std)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")
            return df
    
    def calculate_moving_averages(self, df):
        """Calculate Simple and Exponential Moving Averages"""
        try:
            # Simple Moving Averages
            for period in self.indicators['SMA']['periods']:
                sma = SMAIndicator(close=df['close'], window=period)
                df[f'sma_{period}'] = sma.sma_indicator()
            
            # Exponential Moving Averages
            for period in self.indicators['EMA']['periods']:
                ema = EMAIndicator(close=df['close'], window=period)
                df[f'ema_{period}'] = ema.ema_indicator()
            
            return df
        except Exception as e:
            logging.error(f"Error calculating moving averages: {e}")
            return df
    
    def calculate_stochastic(self, df):
        """Calculate Stochastic Oscillator"""
        try:
            k_period = self.indicators['STOCH']['k_period']
            d_period = self.indicators['STOCH']['d_period']
            
            stoch = StochasticOscillator(
                high=df['high'], 
                low=df['low'], 
                close=df['close'],
                window=k_period,
                smooth_window=d_period
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            return df
        except Exception as e:
            logging.error(f"Error calculating Stochastic: {e}")
            return df
    
    def calculate_atr(self, df):
        """Calculate Average True Range"""
        try:
            period = self.indicators['ATR']['period']
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period)
            df['atr'] = atr.average_true_range()
            return df
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return df
    
    def calculate_adx(self, df):
        """Calculate Average Directional Index"""
        try:
            period = self.indicators['ADX']['period']
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=period)
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            return df
        except Exception as e:
            logging.error(f"Error calculating ADX: {e}")
            return df
    
    def calculate_support_resistance(self, df, window=20):
        """Calculate dynamic support and resistance levels"""
        try:
            # Rolling window high and low
            df['resistance'] = df['high'].rolling(window=window).max()
            df['support'] = df['low'].rolling(window=window).min()
            
            # Pivot points
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = 2 * df['pivot'] - df['low']
            df['s1'] = 2 * df['pivot'] - df['high']
            
            return df
        except Exception as e:
            logging.error(f"Error calculating support/resistance: {e}")
            return df
    
    def calculate_trend_strength(self, df):
        """Calculate trend strength indicator"""
        try:
            # Calculate trend strength based on multiple indicators
            conditions = []
            
            # RSI conditions
            if 'rsi' in df.columns:
                rsi_bullish = df['rsi'] > 50
                rsi_bearish = df['rsi'] < 50
                conditions.append(rsi_bullish.astype(int) - rsi_bearish.astype(int))
            
            # MACD conditions
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd_bullish = df['macd'] > df['macd_signal']
                macd_bearish = df['macd'] < df['macd_signal']
                conditions.append(macd_bullish.astype(int) - macd_bearish.astype(int))
            
            # Moving average conditions
            if 'sma_5' in df.columns and 'sma_20' in df.columns:
                ma_bullish = df['sma_5'] > df['sma_20']
                ma_bearish = df['sma_5'] < df['sma_20']
                conditions.append(ma_bullish.astype(int) - ma_bearish.astype(int))
            
            # Price vs Bollinger Bands
            if 'bb_position' in df.columns:
                bb_bullish = df['bb_position'] > 0.7
                bb_bearish = df['bb_position'] < 0.3
                conditions.append(bb_bullish.astype(int) - bb_bearish.astype(int))
            
            # Combine all conditions
            if conditions:
                df['trend_strength'] = sum(conditions) / len(conditions)
            else:
                df['trend_strength'] = 0
            
            return df
        except Exception as e:
            logging.error(f"Error calculating trend strength: {e}")
            return df
    
    def get_signal_strength(self, df):
        """Calculate overall signal strength (0-10 scale)"""
        try:
            if df is None or len(df) == 0:
                return 0
            
            latest = df.iloc[-1]
            strength_factors = []
            
            # RSI factor
            if 'rsi' in latest:
                rsi = latest['rsi']
                if rsi > 70 or rsi < 30:  # Overbought/Oversold
                    strength_factors.append(8)
                elif rsi > 60 or rsi < 40:
                    strength_factors.append(6)
                else:
                    strength_factors.append(4)
            
            # MACD factor
            if 'macd' in latest and 'macd_signal' in latest:
                macd_diff = abs(latest['macd'] - latest['macd_signal'])
                if macd_diff > 0.001:
                    strength_factors.append(7)
                else:
                    strength_factors.append(4)
            
            # Bollinger Bands factor
            if 'bb_position' in latest:
                bb_pos = latest['bb_position']
                if bb_pos > 0.8 or bb_pos < 0.2:
                    strength_factors.append(8)
                elif bb_pos > 0.6 or bb_pos < 0.4:
                    strength_factors.append(6)
                else:
                    strength_factors.append(3)
            
            # ADX factor (trend strength)
            if 'adx' in latest:
                adx = latest['adx']
                if adx > 25:
                    strength_factors.append(7)
                elif adx > 20:
                    strength_factors.append(5)
                else:
                    strength_factors.append(3)
            
            # Average all factors
            if strength_factors:
                return min(10, sum(strength_factors) / len(strength_factors))
            else:
                return 5
                
        except Exception as e:
            logging.error(f"Error calculating signal strength: {e}")
            return 0
    
    def get_trade_direction(self, df):
        """Determine trade direction based on technical analysis"""
        try:
            if df is None or len(df) == 0:
                return "HOLD", 0
            
            latest = df.iloc[-1]
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            
            # RSI analysis
            if 'rsi' in latest:
                rsi = latest['rsi']
                total_signals += 1
                if rsi > 50:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            
            # MACD analysis
            if 'macd' in latest and 'macd_signal' in latest:
                total_signals += 1
                if latest['macd'] > latest['macd_signal']:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            
            # Moving Average analysis
            if 'ema_12' in latest and 'ema_26' in latest:
                total_signals += 1
                if latest['ema_12'] > latest['ema_26']:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            
            # Bollinger Bands analysis
            if 'bb_position' in latest:
                total_signals += 1
                if latest['bb_position'] > 0.5:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            
            # Stochastic analysis
            if 'stoch_k' in latest and 'stoch_d' in latest:
                total_signals += 1
                if latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] > 20:
                    bullish_signals += 1
                elif latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'] < 80:
                    bearish_signals += 1
            
            # Calculate confidence
            if total_signals > 0:
                bullish_ratio = bullish_signals / total_signals
                bearish_ratio = bearish_signals / total_signals
                
                if bullish_ratio > 0.6:
                    direction = "BUY"
                    confidence = min(95, bullish_ratio * 100)
                elif bearish_ratio > 0.6:
                    direction = "SELL"
                    confidence = min(95, bearish_ratio * 100)
                else:
                    direction = "HOLD"
                    confidence = 50
            else:
                direction = "HOLD"
                confidence = 0
            
            return direction, confidence
            
        except Exception as e:
            logging.error(f"Error determining trade direction: {e}")
            return "HOLD", 0
    
    def calculate_volatility_score(self, df):
        """Calculate volatility score for the asset"""
        try:
            if df is None or len(df) < 20:
                return 1.0
            
            # Calculate price volatility
            price_volatility = df['close'].pct_change().std()
            
            # Calculate ATR-based volatility
            if 'atr' in df.columns:
                atr_volatility = df['atr'].iloc[-1] / df['close'].iloc[-1]
            else:
                atr_volatility = price_volatility
            
            # Calculate Bollinger Band width
            if 'bb_width' in df.columns:
                bb_volatility = df['bb_width'].iloc[-1]
            else:
                bb_volatility = price_volatility
            
            # Combine volatility measures
            volatility_score = (price_volatility + atr_volatility + bb_volatility) / 3
            
            return volatility_score
            
        except Exception as e:
            logging.error(f"Error calculating volatility score: {e}")
            return 1.0
    
    def is_favorable_conditions(self, df):
        """Check if current market conditions are favorable for trading"""
        try:
            if df is None or len(df) < 20:
                return False
            
            # Check volatility
            volatility = self.calculate_volatility_score(df)
            if volatility > 0.02:  # High volatility threshold
                return False
            
            # Check if there are clear signals
            strength = self.get_signal_strength(df)
            if strength < 6:  # Minimum strength threshold
                return False
            
            # Check ADX for trend strength
            if 'adx' in df.columns:
                adx = df['adx'].iloc[-1]
                if adx < 20:  # Weak trend
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking market conditions: {e}")
            return False
    
    def generate_signal_summary(self, df):
        """Generate a comprehensive signal summary"""
        try:
            if df is None:
                return None
            
            direction, confidence = self.get_trade_direction(df)
            strength = self.get_signal_strength(df)
            volatility = self.calculate_volatility_score(df)
            is_favorable = self.is_favorable_conditions(df)
            
            latest = df.iloc[-1]
            
            summary = {
                'direction': direction,
                'confidence': confidence,
                'strength': strength,
                'volatility': volatility,
                'is_favorable': is_favorable,
                'current_price': latest['close'],
                'technical_details': {
                    'rsi': latest.get('rsi', 0),
                    'macd': latest.get('macd', 0),
                    'macd_signal': latest.get('macd_signal', 0),
                    'bb_position': latest.get('bb_position', 0),
                    'adx': latest.get('adx', 0),
                    'support': latest.get('support', 0),
                    'resistance': latest.get('resistance', 0)
                }
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating signal summary: {e}")
            return None