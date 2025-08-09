import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import talib

@dataclass
class OrderFlowData:
    timestamp: datetime
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    size_category: str  # 'retail', 'institutional'

@dataclass
class VolumeProfile:
    price_level: float
    volume: float
    poc_price: float  # Point of Control
    value_area_high: float
    value_area_low: float
    volume_nodes: List[float]

class MarketMicrostructure:
    def __init__(self):
        self.order_flow_data = []
        self.volume_profile_cache = {}
        self.vwap_cache = {}
        self.liquidity_zones = []
        self.smart_money_indicators = {}
        
    def analyze_order_flow(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """Analyze order flow patterns and market microstructure"""
        try:
            if price_data is None or len(price_data) < 100:
                return self.get_default_order_flow()
            
            # Calculate VWAP
            vwap = self.calculate_vwap(price_data, volume_data)
            
            # Analyze volume profile
            volume_profile = self.calculate_volume_profile(price_data, volume_data)
            
            # Detect liquidity zones
            liquidity_zones = self.detect_liquidity_zones(price_data, volume_data)
            
            # Analyze smart money flow
            smart_money_flow = self.analyze_smart_money_flow(price_data, volume_data)
            
            # Calculate order flow imbalance
            flow_imbalance = self.calculate_flow_imbalance(price_data, volume_data)
            
            return {
                'vwap': vwap,
                'volume_profile': volume_profile,
                'liquidity_zones': liquidity_zones,
                'smart_money_flow': smart_money_flow,
                'flow_imbalance': flow_imbalance,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error analyzing order flow: {e}")
            return self.get_default_order_flow()
    
    def calculate_vwap(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """Calculate Volume Weighted Average Price"""
        try:
            if 'close' not in price_data.columns or 'volume' not in volume_data.columns:
                return {'vwap': 0, 'vwap_upper': 0, 'vwap_lower': 0}
            
            # Calculate VWAP
            typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
            vwap = (typical_price * volume_data['volume']).cumsum() / volume_data['volume'].cumsum()
            
            # Calculate VWAP bands
            vwap_std = np.std(vwap)
            vwap_upper = vwap + (2 * vwap_std)
            vwap_lower = vwap - (2 * vwap_std)
            
            current_vwap = vwap.iloc[-1]
            
            return {
                'vwap': current_vwap,
                'vwap_upper': vwap_upper.iloc[-1],
                'vwap_lower': vwap_lower.iloc[-1],
                'vwap_trend': 'bullish' if current_vwap > vwap.iloc[-20] else 'bearish',
                'price_vs_vwap': 'above' if price_data['close'].iloc[-1] > current_vwap else 'below'
            }
            
        except Exception as e:
            logging.error(f"Error calculating VWAP: {e}")
            return {'vwap': 0, 'vwap_upper': 0, 'vwap_lower': 0}
    
    def calculate_volume_profile(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """Calculate volume profile and Point of Control"""
        try:
            if len(price_data) < 50:
                return self.get_default_volume_profile()
            
            # Create price bins
            price_range = price_data['high'].max() - price_data['low'].min()
            bin_size = price_range / 50
            price_bins = np.arange(price_data['low'].min(), price_data['high'].max(), bin_size)
            
            # Calculate volume at each price level
            volume_profile = {}
            for i in range(len(price_bins) - 1):
                mask = (price_data['close'] >= price_bins[i]) & (price_data['close'] < price_bins[i + 1])
                volume_profile[price_bins[i]] = volume_data['volume'][mask].sum()
            
            # Find Point of Control (highest volume price level)
            poc_price = max(volume_profile, key=volume_profile.get)
            max_volume = volume_profile[poc_price]
            
            # Calculate Value Area (70% of total volume)
            total_volume = sum(volume_profile.values())
            value_area_target = total_volume * 0.7
            
            # Find value area boundaries
            sorted_prices = sorted(volume_profile.keys())
            poc_index = sorted_prices.index(poc_price)
            
            # Expand value area from POC
            value_area_volume = volume_profile[poc_price]
            value_area_high = poc_price
            value_area_low = poc_price
            
            high_index = poc_index
            low_index = poc_index
            
            while value_area_volume < value_area_target and (high_index < len(sorted_prices) - 1 or low_index > 0):
                if high_index < len(sorted_prices) - 1:
                    high_volume = volume_profile[sorted_prices[high_index + 1]]
                else:
                    high_volume = 0
                
                if low_index > 0:
                    low_volume = volume_profile[sorted_prices[low_index - 1]]
                else:
                    low_volume = 0
                
                if high_volume > low_volume and high_index < len(sorted_prices) - 1:
                    high_index += 1
                    value_area_volume += high_volume
                    value_area_high = sorted_prices[high_index]
                elif low_index > 0:
                    low_index -= 1
                    value_area_volume += low_volume
                    value_area_low = sorted_prices[low_index]
                else:
                    break
            
            return {
                'poc_price': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'max_volume': max_volume,
                'total_volume': total_volume,
                'volume_profile': volume_profile
            }
            
        except Exception as e:
            logging.error(f"Error calculating volume profile: {e}")
            return self.get_default_volume_profile()
    
    def detect_liquidity_zones(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> List[Dict]:
        """Detect liquidity zones (support/resistance with high volume)"""
        try:
            liquidity_zones = []
            
            # Find high volume nodes
            volume_threshold = volume_data['volume'].quantile(0.8)
            high_volume_periods = volume_data[volume_data['volume'] > volume_threshold]
            
            for idx in high_volume_periods.index:
                price_level = price_data.loc[idx, 'close']
                volume_level = volume_data.loc[idx, 'volume']
                
                # Check if this is a significant level
                nearby_volume = volume_data[
                    (price_data['close'] >= price_level * 0.995) & 
                    (price_data['close'] <= price_level * 1.005)
                ]['volume'].sum()
                
                if nearby_volume > volume_threshold * 2:
                    liquidity_zones.append({
                        'price_level': price_level,
                        'volume': nearby_volume,
                        'strength': nearby_volume / volume_threshold,
                        'type': 'support' if price_data['close'].iloc[-1] > price_level else 'resistance'
                    })
            
            # Sort by strength and remove duplicates
            liquidity_zones = sorted(liquidity_zones, key=lambda x: x['strength'], reverse=True)
            
            # Remove zones too close to each other
            filtered_zones = []
            for zone in liquidity_zones:
                too_close = False
                for existing_zone in filtered_zones:
                    if abs(zone['price_level'] - existing_zone['price_level']) / existing_zone['price_level'] < 0.01:
                        too_close = True
                        break
                
                if not too_close:
                    filtered_zones.append(zone)
            
            return filtered_zones[:5]  # Return top 5 zones
            
        except Exception as e:
            logging.error(f"Error detecting liquidity zones: {e}")
            return []
    
    def analyze_smart_money_flow(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """Analyze smart money flow patterns"""
        try:
            if len(price_data) < 20:
                return self.get_default_smart_money_flow()
            
            # Calculate Money Flow Index
            typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
            money_flow = typical_price * volume_data['volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            
            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            
            # Calculate Accumulation/Distribution Line
            clv = ((price_data['close'] - price_data['low']) - (price_data['high'] - price_data['close'])) / (price_data['high'] - price_data['low'])
            adl = (clv * volume_data['volume']).cumsum()
            
            # Calculate On-Balance Volume
            obv = (np.sign(price_data['close'].diff()) * volume_data['volume']).cumsum()
            
            # Smart money indicators
            current_mfi = mfi.iloc[-1]
            mfi_trend = 'bullish' if current_mfi > 50 else 'bearish'
            
            adl_trend = 'bullish' if adl.iloc[-1] > adl.iloc[-20] else 'bearish'
            obv_trend = 'bullish' if obv.iloc[-1] > obv.iloc[-20] else 'bearish'
            
            # Smart money confidence
            smart_money_score = 0
            if mfi_trend == 'bullish': smart_money_score += 1
            if adl_trend == 'bullish': smart_money_score += 1
            if obv_trend == 'bullish': smart_money_score += 1
            
            return {
                'mfi': current_mfi,
                'mfi_trend': mfi_trend,
                'adl_trend': adl_trend,
                'obv_trend': obv_trend,
                'smart_money_score': smart_money_score,
                'smart_money_bias': 'bullish' if smart_money_score >= 2 else 'bearish',
                'confidence': min(smart_money_score / 3 * 100, 100)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing smart money flow: {e}")
            return self.get_default_smart_money_flow()
    
    def calculate_flow_imbalance(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """Calculate order flow imbalance"""
        try:
            if len(price_data) < 10:
                return {'imbalance': 0, 'bias': 'neutral'}
            
            # Calculate price momentum
            price_change = price_data['close'].pct_change()
            
            # Calculate volume-weighted price change
            volume_weighted_change = (price_change * volume_data['volume']).rolling(10).sum()
            
            # Calculate imbalance
            recent_imbalance = volume_weighted_change.iloc[-1]
            
            # Determine bias
            if recent_imbalance > 0.01:
                bias = 'bullish'
            elif recent_imbalance < -0.01:
                bias = 'bearish'
            else:
                bias = 'neutral'
            
            return {
                'imbalance': recent_imbalance,
                'bias': bias,
                'strength': abs(recent_imbalance) * 100
            }
            
        except Exception as e:
            logging.error(f"Error calculating flow imbalance: {e}")
            return {'imbalance': 0, 'bias': 'neutral'}
    
    def get_default_order_flow(self) -> Dict:
        """Return default order flow data when analysis fails"""
        return {
            'vwap': {'vwap': 0, 'vwap_upper': 0, 'vwap_lower': 0},
            'volume_profile': self.get_default_volume_profile(),
            'liquidity_zones': [],
            'smart_money_flow': self.get_default_smart_money_flow(),
            'flow_imbalance': {'imbalance': 0, 'bias': 'neutral'},
            'timestamp': datetime.now()
        }
    
    def get_default_volume_profile(self) -> Dict:
        """Return default volume profile data"""
        return {
            'poc_price': 0,
            'value_area_high': 0,
            'value_area_low': 0,
            'max_volume': 0,
            'total_volume': 0,
            'volume_profile': {}
        }
    
    def get_default_smart_money_flow(self) -> Dict:
        """Return default smart money flow data"""
        return {
            'mfi': 50,
            'mfi_trend': 'neutral',
            'adl_trend': 'neutral',
            'obv_trend': 'neutral',
            'smart_money_score': 0,
            'smart_money_bias': 'neutral',
            'confidence': 0
        }