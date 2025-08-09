import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math
from dataclasses import dataclass

@dataclass
class Pattern:
    pattern_type: str
    confidence: float
    direction: str  # 'bullish', 'bearish', 'neutral'
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    timeframe: str
    timestamp: datetime
    description: str

@dataclass
class HarmonicPattern:
    pattern_type: str  # 'Gartley', 'Bat', 'Butterfly', 'Crab', 'Cypher'
    x_ab_ratio: float
    bc_ab_ratio: float
    cd_bc_ratio: float
    confidence: float
    direction: str
    entry_zone: Tuple[float, float]
    target_zone: Tuple[float, float]
    stop_loss: float

class PatternRecognition:
    def __init__(self):
        self.patterns = []
        self.harmonic_patterns = []
        self.elliott_waves = []
        self.wyckoff_patterns = []
        self.order_blocks = []
        
        # Harmonic pattern ratios
        self.harmonic_ratios = {
            'Gartley': {
                'x_ab': 0.618,
                'bc_ab': 0.382,
                'cd_bc': 1.272,
                'cd_ab': 0.786
            },
            'Bat': {
                'x_ab': 0.382,
                'bc_ab': 0.382,
                'cd_bc': 2.618,
                'cd_ab': 1.000
            },
            'Butterfly': {
                'x_ab': 0.786,
                'bc_ab': 0.382,
                'cd_bc': 1.618,
                'cd_ab': 1.272
            },
            'Crab': {
                'x_ab': 0.382,
                'bc_ab': 0.382,
                'cd_bc': 3.618,
                'cd_ab': 1.618
            },
            'Cypher': {
                'x_ab': 0.382,
                'bc_ab': 0.113,
                'cd_bc': 1.414,
                'cd_ab': 0.786
            }
        }
        
        # Elliott Wave characteristics
        self.elliott_wave_rules = {
            'wave_2_retracement': (0.382, 0.618),
            'wave_3_extension': (1.618, 2.618),
            'wave_4_retracement': (0.236, 0.382),
            'wave_5_extension': (0.618, 1.000)
        }
        
    def analyze_all_patterns(self, data: pd.DataFrame, timeframe: str = '1h') -> Dict:
        """Analyze all pattern types in the data"""
        try:
            if data is None or len(data) < 100:
                return self.get_default_pattern_analysis()
            
            # Reset patterns
            self.patterns = []
            self.harmonic_patterns = []
            self.elliott_waves = []
            self.wyckoff_patterns = []
            self.order_blocks = []
            
            # Analyze different pattern types
            self.detect_harmonic_patterns(data, timeframe)
            self.detect_elliott_waves(data, timeframe)
            self.detect_wyckoff_patterns(data, timeframe)
            self.detect_order_blocks(data, timeframe)
            self.detect_chart_patterns(data, timeframe)
            
            # Combine all patterns
            combined_analysis = self.combine_pattern_analysis()
            
            return {
                'harmonic_patterns': self.harmonic_patterns,
                'elliott_waves': self.elliott_waves,
                'wyckoff_patterns': self.wyckoff_patterns,
                'order_blocks': self.order_blocks,
                'chart_patterns': self.patterns,
                'combined_analysis': combined_analysis,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error analyzing patterns: {e}")
            return self.get_default_pattern_analysis()
    
    def detect_harmonic_patterns(self, data: pd.DataFrame, timeframe: str):
        """Detect harmonic patterns (Gartley, Bat, Butterfly, etc.)"""
        try:
            if len(data) < 50:
                return
            
            # Find swing highs and lows
            swing_points = self.find_swing_points(data)
            
            if len(swing_points) < 4:
                return
            
            # Check for harmonic patterns
            for i in range(len(swing_points) - 3):
                x, a, b, c, d = swing_points[i:i+5]
                
                # Calculate ratios
                x_ab_ratio = abs(b - a) / abs(x - a) if abs(x - a) > 0 else 0
                bc_ab_ratio = abs(c - b) / abs(b - a) if abs(b - a) > 0 else 0
                cd_bc_ratio = abs(d - c) / abs(c - b) if abs(c - b) > 0 else 0
                cd_ab_ratio = abs(d - c) / abs(b - a) if abs(b - a) > 0 else 0
                
                # Check each harmonic pattern
                for pattern_name, ratios in self.harmonic_ratios.items():
                    confidence = self.check_harmonic_pattern(
                        x_ab_ratio, bc_ab_ratio, cd_bc_ratio, cd_ab_ratio, ratios
                    )
                    
                    if confidence > 0.7:  # 70% confidence threshold
                        pattern = self.create_harmonic_pattern(
                            pattern_name, x, a, b, c, d, confidence, timeframe
                        )
                        self.harmonic_patterns.append(pattern)
            
        except Exception as e:
            logging.error(f"Error detecting harmonic patterns: {e}")
    
    def find_swing_points(self, data: pd.DataFrame) -> List[float]:
        """Find swing highs and lows in the data"""
        try:
            swing_points = []
            window = 5  # Look for swings in 5-period window
            
            for i in range(window, len(data) - window):
                high = data['high'].iloc[i]
                low = data['low'].iloc[i]
                
                # Check if this is a swing high
                if all(high >= data['high'].iloc[i-window:i]) and all(high >= data['high'].iloc[i+1:i+window+1]):
                    swing_points.append(high)
                
                # Check if this is a swing low
                elif all(low <= data['low'].iloc[i-window:i]) and all(low <= data['low'].iloc[i+1:i+window+1]):
                    swing_points.append(low)
            
            return swing_points
            
        except Exception as e:
            logging.error(f"Error finding swing points: {e}")
            return []
    
    def check_harmonic_pattern(self, x_ab: float, bc_ab: float, cd_bc: float, cd_ab: float, target_ratios: Dict) -> float:
        """Check if ratios match a harmonic pattern"""
        try:
            tolerance = 0.1  # 10% tolerance
            
            # Check each ratio
            x_ab_match = abs(x_ab - target_ratios['x_ab']) <= tolerance
            bc_ab_match = abs(bc_ab - target_ratios['bc_ab']) <= tolerance
            cd_bc_match = abs(cd_bc - target_ratios['cd_bc']) <= tolerance
            cd_ab_match = abs(cd_ab - target_ratios['cd_ab']) <= tolerance
            
            # Calculate confidence based on matches
            matches = sum([x_ab_match, bc_ab_match, cd_bc_match, cd_ab_match])
            confidence = matches / 4
            
            return confidence
            
        except Exception as e:
            logging.error(f"Error checking harmonic pattern: {e}")
            return 0.0
    
    def create_harmonic_pattern(self, pattern_type: str, x: float, a: float, b: float, c: float, d: float, confidence: float, timeframe: str) -> HarmonicPattern:
        """Create a harmonic pattern object"""
        try:
            # Determine direction based on pattern completion
            direction = 'bullish' if d < c else 'bearish'
            
            # Calculate entry and target zones
            if direction == 'bullish':
                entry_zone = (d * 0.995, d * 1.005)
                target_zone = (c * 0.995, c * 1.005)
                stop_loss = d * 0.99
            else:
                entry_zone = (d * 0.995, d * 1.005)
                target_zone = (c * 0.995, c * 1.005)
                stop_loss = d * 1.01
            
            return HarmonicPattern(
                pattern_type=pattern_type,
                x_ab_ratio=abs(b - a) / abs(x - a) if abs(x - a) > 0 else 0,
                bc_ab_ratio=abs(c - b) / abs(b - a) if abs(b - a) > 0 else 0,
                cd_bc_ratio=abs(d - c) / abs(c - b) if abs(c - b) > 0 else 0,
                confidence=confidence,
                direction=direction,
                entry_zone=entry_zone,
                target_zone=target_zone,
                stop_loss=stop_loss
            )
            
        except Exception as e:
            logging.error(f"Error creating harmonic pattern: {e}")
            return None
    
    def detect_elliott_waves(self, data: pd.DataFrame, timeframe: str):
        """Detect Elliott Wave patterns"""
        try:
            if len(data) < 100:
                return
            
            # Find potential wave structure
            waves = self.identify_elliott_waves(data)
            
            if len(waves) >= 5:
                # Validate Elliott Wave rules
                if self.validate_elliott_wave_rules(waves):
                    wave_analysis = {
                        'waves': waves,
                        'pattern_type': self.classify_elliott_pattern(waves),
                        'confidence': self.calculate_elliott_confidence(waves),
                        'direction': self.determine_elliott_direction(waves),
                        'target': self.calculate_elliott_target(waves),
                        'timeframe': timeframe,
                        'timestamp': datetime.now()
                    }
                    self.elliott_waves.append(wave_analysis)
            
        except Exception as e:
            logging.error(f"Error detecting Elliott waves: {e}")
    
    def identify_elliott_waves(self, data: pd.DataFrame) -> List[Dict]:
        """Identify potential Elliott Wave structure"""
        try:
            waves = []
            
            # Find significant swing points
            swing_points = self.find_swing_points(data)
            
            if len(swing_points) < 5:
                return waves
            
            # Identify waves based on price movement
            for i in range(len(swing_points) - 1):
                wave = {
                    'start': swing_points[i],
                    'end': swing_points[i + 1],
                    'length': abs(swing_points[i + 1] - swing_points[i]),
                    'direction': 'up' if swing_points[i + 1] > swing_points[i] else 'down'
                }
                waves.append(wave)
            
            return waves[:5]  # Return first 5 waves
            
        except Exception as e:
            logging.error(f"Error identifying Elliott waves: {e}")
            return []
    
    def validate_elliott_wave_rules(self, waves: List[Dict]) -> bool:
        """Validate Elliott Wave rules"""
        try:
            if len(waves) < 5:
                return False
            
            # Rule 1: Wave 2 cannot retrace more than 100% of Wave 1
            if len(waves) >= 2:
                wave1_length = waves[0]['length']
                wave2_length = waves[1]['length']
                if wave2_length > wave1_length:
                    return False
            
            # Rule 2: Wave 3 cannot be the shortest wave
            if len(waves) >= 3:
                wave3_length = waves[2]['length']
                other_waves = [w['length'] for i, w in enumerate(waves) if i != 2]
                if wave3_length < min(other_waves):
                    return False
            
            # Rule 3: Wave 4 cannot overlap with Wave 1
            if len(waves) >= 4:
                wave1_end = waves[0]['end']
                wave4_end = waves[3]['end']
                if abs(wave4_end - wave1_end) < wave1_end * 0.01:
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating Elliott wave rules: {e}")
            return False
    
    def classify_elliott_pattern(self, waves: List[Dict]) -> str:
        """Classify Elliott Wave pattern type"""
        try:
            if len(waves) < 5:
                return 'incomplete'
            
            # Simple classification based on wave structure
            wave_directions = [w['direction'] for w in waves[:5]]
            
            if wave_directions == ['up', 'down', 'up', 'down', 'up']:
                return 'impulse'
            elif wave_directions == ['down', 'up', 'down', 'up', 'down']:
                return 'corrective'
            else:
                return 'complex'
                
        except Exception as e:
            logging.error(f"Error classifying Elliott pattern: {e}")
            return 'unknown'
    
    def calculate_elliott_confidence(self, waves: List[Dict]) -> float:
        """Calculate confidence in Elliott Wave pattern"""
        try:
            if len(waves) < 5:
                return 0.0
            
            # Calculate confidence based on wave relationships
            confidence_factors = []
            
            # Wave 2 retracement
            if len(waves) >= 2:
                wave1_length = waves[0]['length']
                wave2_length = waves[1]['length']
                retracement = wave2_length / wave1_length
                if 0.382 <= retracement <= 0.618:
                    confidence_factors.append(0.8)
                elif 0.236 <= retracement <= 0.786:
                    confidence_factors.append(0.6)
                else:
                    confidence_factors.append(0.3)
            
            # Wave 3 extension
            if len(waves) >= 3:
                wave3_length = waves[2]['length']
                wave1_length = waves[0]['length']
                extension = wave3_length / wave1_length
                if 1.618 <= extension <= 2.618:
                    confidence_factors.append(0.9)
                elif 1.000 <= extension <= 3.000:
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.4)
            
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating Elliott confidence: {e}")
            return 0.0
    
    def detect_wyckoff_patterns(self, data: pd.DataFrame, timeframe: str):
        """Detect Wyckoff accumulation/distribution patterns"""
        try:
            if len(data) < 200:
                return
            
            # Look for Wyckoff phases
            wyckoff_analysis = self.analyze_wyckoff_phases(data)
            
            if wyckoff_analysis['pattern_detected']:
                self.wyckoff_patterns.append({
                    'pattern_type': wyckoff_analysis['pattern_type'],
                    'phase': wyckoff_analysis['current_phase'],
                    'confidence': wyckoff_analysis['confidence'],
                    'direction': wyckoff_analysis['direction'],
                    'support_level': wyckoff_analysis['support_level'],
                    'resistance_level': wyckoff_analysis['resistance_level'],
                    'timeframe': timeframe,
                    'timestamp': datetime.now()
                })
            
        except Exception as e:
            logging.error(f"Error detecting Wyckoff patterns: {e}")
    
    def analyze_wyckoff_phases(self, data: pd.DataFrame) -> Dict:
        """Analyze Wyckoff accumulation/distribution phases"""
        try:
            # Calculate volume and price relationships
            recent_data = data.tail(100)
            
            # Calculate volume trend
            volume_trend = recent_data['volume'].pct_change().rolling(20).mean().iloc[-1]
            
            # Calculate price trend
            price_trend = recent_data['close'].pct_change().rolling(20).mean().iloc[-1]
            
            # Identify Wyckoff phase
            if volume_trend > 0.1 and price_trend < -0.05:
                # High volume, declining price - possible accumulation
                pattern_type = 'accumulation'
                current_phase = 'phase_a'
                direction = 'bullish'
                confidence = 0.7
            elif volume_trend < -0.1 and price_trend > 0.05:
                # Low volume, rising price - possible distribution
                pattern_type = 'distribution'
                current_phase = 'phase_a'
                direction = 'bearish'
                confidence = 0.7
            else:
                pattern_type = 'none'
                current_phase = 'unknown'
                direction = 'neutral'
                confidence = 0.3
            
            # Calculate support and resistance levels
            support_level = recent_data['low'].min()
            resistance_level = recent_data['high'].max()
            
            return {
                'pattern_detected': pattern_type != 'none',
                'pattern_type': pattern_type,
                'current_phase': current_phase,
                'confidence': confidence,
                'direction': direction,
                'support_level': support_level,
                'resistance_level': resistance_level
            }
            
        except Exception as e:
            logging.error(f"Error analyzing Wyckoff phases: {e}")
            return {
                'pattern_detected': False,
                'pattern_type': 'none',
                'current_phase': 'unknown',
                'confidence': 0.0,
                'direction': 'neutral',
                'support_level': 0.0,
                'resistance_level': 0.0
            }
    
    def detect_order_blocks(self, data: pd.DataFrame, timeframe: str):
        """Detect institutional order blocks"""
        try:
            if len(data) < 50:
                return
            
            # Find order blocks based on volume and price action
            order_blocks = self.find_order_blocks(data)
            
            for block in order_blocks:
                self.order_blocks.append({
                    'type': block['type'],
                    'start_price': block['start_price'],
                    'end_price': block['end_price'],
                    'volume': block['volume'],
                    'strength': block['strength'],
                    'direction': block['direction'],
                    'timeframe': timeframe,
                    'timestamp': datetime.now()
                })
            
        except Exception as e:
            logging.error(f"Error detecting order blocks: {e}")
    
    def find_order_blocks(self, data: pd.DataFrame) -> List[Dict]:
        """Find institutional order blocks"""
        try:
            order_blocks = []
            
            # Look for high volume candles with strong price movement
            for i in range(10, len(data) - 10):
                current_candle = data.iloc[i]
                
                # Calculate volume ratio
                avg_volume = data['volume'].iloc[i-10:i+10].mean()
                volume_ratio = current_candle['volume'] / avg_volume
                
                # Calculate price movement
                price_change = abs(current_candle['close'] - current_candle['open'])
                avg_price_change = data['close'].pct_change().abs().iloc[i-10:i+10].mean()
                price_ratio = price_change / avg_price_change if avg_price_change > 0 else 0
                
                # Identify order block
                if volume_ratio > 2.0 and price_ratio > 1.5:
                    block_type = 'bullish' if current_candle['close'] > current_candle['open'] else 'bearish'
                    
                    order_blocks.append({
                        'type': block_type,
                        'start_price': current_candle['open'],
                        'end_price': current_candle['close'],
                        'volume': current_candle['volume'],
                        'strength': min(volume_ratio * price_ratio / 3, 1.0),
                        'direction': block_type
                    })
            
            return order_blocks
            
        except Exception as e:
            logging.error(f"Error finding order blocks: {e}")
            return []
    
    def detect_chart_patterns(self, data: pd.DataFrame, timeframe: str):
        """Detect traditional chart patterns"""
        try:
            if len(data) < 50:
                return
            
            # Detect common chart patterns
            patterns_to_check = [
                'double_top', 'double_bottom', 'head_and_shoulders',
                'inverse_head_and_shoulders', 'triangle', 'flag', 'wedge'
            ]
            
            for pattern_type in patterns_to_check:
                pattern = self.detect_specific_pattern(data, pattern_type)
                if pattern:
                    self.patterns.append(pattern)
            
        except Exception as e:
            logging.error(f"Error detecting chart patterns: {e}")
    
    def detect_specific_pattern(self, data: pd.DataFrame, pattern_type: str) -> Optional[Pattern]:
        """Detect a specific chart pattern"""
        try:
            if pattern_type == 'double_top':
                return self.detect_double_top(data)
            elif pattern_type == 'double_bottom':
                return self.detect_double_bottom(data)
            elif pattern_type == 'head_and_shoulders':
                return self.detect_head_and_shoulders(data)
            elif pattern_type == 'triangle':
                return self.detect_triangle(data)
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error detecting {pattern_type}: {e}")
            return None
    
    def detect_double_top(self, data: pd.DataFrame) -> Optional[Pattern]:
        """Detect double top pattern"""
        try:
            if len(data) < 30:
                return None
            
            # Look for two peaks at similar levels
            highs = data['high'].rolling(5).max()
            
            for i in range(20, len(data) - 10):
                # Check if current point is a peak
                if highs.iloc[i] == data['high'].iloc[i]:
                    # Look for another peak in the future
                    for j in range(i + 10, min(i + 30, len(data))):
                        if highs.iloc[j] == data['high'].iloc[j]:
                            # Check if peaks are at similar levels
                            peak1 = data['high'].iloc[i]
                            peak2 = data['high'].iloc[j]
                            
                            if abs(peak1 - peak2) / peak1 < 0.02:  # 2% tolerance
                                # Calculate pattern metrics
                                entry_price = data['close'].iloc[-1]
                                target_price = entry_price * 0.95  # 5% downside target
                                stop_loss = max(peak1, peak2) * 1.01
                                risk_reward = (entry_price - target_price) / (stop_loss - entry_price)
                                
                                return Pattern(
                                    pattern_type='double_top',
                                    confidence=0.75,
                                    direction='bearish',
                                    entry_price=entry_price,
                                    target_price=target_price,
                                    stop_loss=stop_loss,
                                    risk_reward_ratio=risk_reward,
                                    timeframe='1h',
                                    timestamp=datetime.now(),
                                    description=f'Double top at {peak1:.4f} and {peak2:.4f}'
                                )
            
            return None
            
        except Exception as e:
            logging.error(f"Error detecting double top: {e}")
            return None
    
    def detect_double_bottom(self, data: pd.DataFrame) -> Optional[Pattern]:
        """Detect double bottom pattern"""
        try:
            if len(data) < 30:
                return None
            
            # Look for two troughs at similar levels
            lows = data['low'].rolling(5).min()
            
            for i in range(20, len(data) - 10):
                # Check if current point is a trough
                if lows.iloc[i] == data['low'].iloc[i]:
                    # Look for another trough in the future
                    for j in range(i + 10, min(i + 30, len(data))):
                        if lows.iloc[j] == data['low'].iloc[j]:
                            # Check if troughs are at similar levels
                            trough1 = data['low'].iloc[i]
                            trough2 = data['low'].iloc[j]
                            
                            if abs(trough1 - trough2) / trough1 < 0.02:  # 2% tolerance
                                # Calculate pattern metrics
                                entry_price = data['close'].iloc[-1]
                                target_price = entry_price * 1.05  # 5% upside target
                                stop_loss = min(trough1, trough2) * 0.99
                                risk_reward = (target_price - entry_price) / (entry_price - stop_loss)
                                
                                return Pattern(
                                    pattern_type='double_bottom',
                                    confidence=0.75,
                                    direction='bullish',
                                    entry_price=entry_price,
                                    target_price=target_price,
                                    stop_loss=stop_loss,
                                    risk_reward_ratio=risk_reward,
                                    timeframe='1h',
                                    timestamp=datetime.now(),
                                    description=f'Double bottom at {trough1:.4f} and {trough2:.4f}'
                                )
            
            return None
            
        except Exception as e:
            logging.error(f"Error detecting double bottom: {e}")
            return None
    
    def detect_head_and_shoulders(self, data: pd.DataFrame) -> Optional[Pattern]:
        """Detect head and shoulders pattern"""
        try:
            if len(data) < 50:
                return None
            
            # This is a simplified implementation
            # In practice, you'd need more sophisticated logic
            
            # Look for three peaks with middle peak higher
            highs = data['high'].rolling(5).max()
            peaks = []
            
            for i in range(20, len(data) - 20):
                if highs.iloc[i] == data['high'].iloc[i]:
                    peaks.append((i, data['high'].iloc[i]))
            
            if len(peaks) >= 3:
                # Check if middle peak is higher
                left_shoulder = peaks[-3][1]
                head = peaks[-2][1]
                right_shoulder = peaks[-1][1]
                
                if head > left_shoulder and head > right_shoulder:
                    # Calculate pattern metrics
                    entry_price = data['close'].iloc[-1]
                    neckline = (left_shoulder + right_shoulder) / 2
                    target_price = entry_price - (head - neckline)
                    stop_loss = head * 1.01
                    risk_reward = (entry_price - target_price) / (stop_loss - entry_price)
                    
                    return Pattern(
                        pattern_type='head_and_shoulders',
                        confidence=0.8,
                        direction='bearish',
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        risk_reward_ratio=risk_reward,
                        timeframe='1h',
                        timestamp=datetime.now(),
                        description=f'Head and shoulders with head at {head:.4f}'
                    )
            
            return None
            
        except Exception as e:
            logging.error(f"Error detecting head and shoulders: {e}")
            return None
    
    def detect_triangle(self, data: pd.DataFrame) -> Optional[Pattern]:
        """Detect triangle pattern"""
        try:
            if len(data) < 30:
                return None
            
            # Calculate trend lines
            highs = data['high'].rolling(5).max()
            lows = data['low'].rolling(5).min()
            
            # Fit trend lines (simplified)
            recent_highs = highs.tail(20)
            recent_lows = lows.tail(20)
            
            # Check if highs are declining and lows are rising (ascending triangle)
            high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            if high_slope < -0.001 and low_slope > 0.001:
                # Ascending triangle (bullish)
                direction = 'bullish'
                confidence = 0.7
                target_price = data['close'].iloc[-1] * 1.03
                stop_loss = data['low'].iloc[-1] * 0.99
            elif high_slope > 0.001 and low_slope < -0.001:
                # Descending triangle (bearish)
                direction = 'bearish'
                confidence = 0.7
                target_price = data['close'].iloc[-1] * 0.97
                stop_loss = data['high'].iloc[-1] * 1.01
            else:
                return None
            
            entry_price = data['close'].iloc[-1]
            risk_reward = abs(target_price - entry_price) / abs(stop_loss - entry_price)
            
            return Pattern(
                pattern_type='triangle',
                confidence=confidence,
                direction=direction,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward,
                timeframe='1h',
                timestamp=datetime.now(),
                description=f'{direction.capitalize()} triangle pattern'
            )
            
        except Exception as e:
            logging.error(f"Error detecting triangle: {e}")
            return None
    
    def combine_pattern_analysis(self) -> Dict:
        """Combine all pattern analysis into a single recommendation"""
        try:
            # Calculate overall pattern sentiment
            bullish_patterns = 0
            bearish_patterns = 0
            total_confidence = 0
            pattern_count = 0
            
            # Count patterns by direction
            for pattern in self.patterns:
                if pattern.direction == 'bullish':
                    bullish_patterns += 1
                elif pattern.direction == 'bearish':
                    bearish_patterns += 1
                total_confidence += pattern.confidence
                pattern_count += 1
            
            for pattern in self.harmonic_patterns:
                if pattern.direction == 'bullish':
                    bullish_patterns += 1
                elif pattern.direction == 'bearish':
                    bearish_patterns += 1
                total_confidence += pattern.confidence
                pattern_count += 1
            
            for pattern in self.wyckoff_patterns:
                if pattern['direction'] == 'bullish':
                    bullish_patterns += 1
                elif pattern['direction'] == 'bearish':
                    bearish_patterns += 1
                total_confidence += pattern['confidence']
                pattern_count += 1
            
            # Determine overall direction
            if bullish_patterns > bearish_patterns:
                overall_direction = 'bullish'
                direction_strength = bullish_patterns / (bullish_patterns + bearish_patterns) * 100
            elif bearish_patterns > bullish_patterns:
                overall_direction = 'bearish'
                direction_strength = bearish_patterns / (bullish_patterns + bearish_patterns) * 100
            else:
                overall_direction = 'neutral'
                direction_strength = 50
            
            # Calculate average confidence
            avg_confidence = total_confidence / pattern_count if pattern_count > 0 else 0
            
            return {
                'overall_direction': overall_direction,
                'direction_strength': direction_strength,
                'average_confidence': avg_confidence,
                'total_patterns': pattern_count,
                'bullish_patterns': bullish_patterns,
                'bearish_patterns': bearish_patterns,
                'pattern_quality': 'high' if avg_confidence > 0.7 else 'medium' if avg_confidence > 0.5 else 'low'
            }
            
        except Exception as e:
            logging.error(f"Error combining pattern analysis: {e}")
            return self.get_default_combined_analysis()
    
    def get_default_pattern_analysis(self) -> Dict:
        """Return default pattern analysis"""
        return {
            'harmonic_patterns': [],
            'elliott_waves': [],
            'wyckoff_patterns': [],
            'order_blocks': [],
            'chart_patterns': [],
            'combined_analysis': self.get_default_combined_analysis(),
            'timestamp': datetime.now()
        }
    
    def get_default_combined_analysis(self) -> Dict:
        """Return default combined analysis"""
        return {
            'overall_direction': 'neutral',
            'direction_strength': 50,
            'average_confidence': 0,
            'total_patterns': 0,
            'bullish_patterns': 0,
            'bearish_patterns': 0,
            'pattern_quality': 'low'
        }