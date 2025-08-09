import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
import threading
from typing import Dict, List, Optional
import json

class DemoPocketOptionAPI:
    def __init__(self):
        self.is_connected = False
        self.is_running = False
        self.available_pairs = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 
            'USD/CAD', 'USD/CHF', 'NZD/USD', 'EUR/GBP'
        ]
        self.current_prices = {}
        self.price_history = {}
        self.connection_thread = None
        
        # Initialize demo prices
        self.initialize_demo_prices()
        
    def initialize_demo_prices(self):
        """Initialize demo prices for all pairs"""
        base_prices = {
            'EUR/USD': 1.0850,
            'GBP/USD': 1.2650,
            'USD/JPY': 150.50,
            'AUD/USD': 0.6550,
            'USD/CAD': 1.3650,
            'USD/CHF': 0.8850,
            'NZD/USD': 0.6050,
            'EUR/GBP': 0.8580
        }
        
        for pair in self.available_pairs:
            self.current_prices[pair] = base_prices.get(pair, 1.0000)
            self.price_history[pair] = self.generate_demo_history(pair, base_prices.get(pair, 1.0000))
    
    def generate_demo_history(self, pair: str, base_price: float) -> pd.DataFrame:
        """Generate demo price history for a pair"""
        try:
            # Generate 200 data points
            n_points = 200
            timestamps = []
            prices = []
            
            current_time = datetime.now() - timedelta(minutes=n_points)
            
            for i in range(n_points):
                timestamps.append(current_time + timedelta(minutes=i))
                
                # Add some realistic price movement
                if i == 0:
                    price = base_price
                else:
                    # Random walk with some trend
                    change = np.random.normal(0, 0.0001)  # Small random change
                    trend = np.sin(i / 20) * 0.0002  # Cyclical trend
                    price = prices[-1] * (1 + change + trend)
                
                prices.append(price)
            
            # Create OHLCV data
            data = []
            for i in range(0, len(prices), 5):  # 5-minute candles
                if i + 4 < len(prices):
                    candle_prices = prices[i:i+5]
                    high = max(candle_prices)
                    low = min(candle_prices)
                    open_price = candle_prices[0]
                    close_price = candle_prices[-1]
                    volume = np.random.randint(1000, 5000)
                    
                    data.append({
                        'timestamp': timestamps[i],
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close_price,
                        'volume': volume
                    })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logging.error(f"Error generating demo history for {pair}: {e}")
            return pd.DataFrame()
    
    def start(self) -> bool:
        """Start the demo API"""
        try:
            logging.info("Starting Demo Pocket Option API...")
            
            self.is_connected = True
            self.is_running = True
            
            # Start price update thread
            self.connection_thread = threading.Thread(target=self.price_update_loop)
            self.connection_thread.daemon = True
            self.connection_thread.start()
            
            logging.info("Demo Pocket Option API started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error starting demo API: {e}")
            return False
    
    def stop(self):
        """Stop the demo API"""
        try:
            self.is_running = False
            self.is_connected = False
            logging.info("Demo Pocket Option API stopped")
            
        except Exception as e:
            logging.error(f"Error stopping demo API: {e}")
    
    def price_update_loop(self):
        """Update prices in real-time"""
        while self.is_running:
            try:
                for pair in self.available_pairs:
                    # Update current price with small random movement
                    current_price = self.current_prices[pair]
                    change = np.random.normal(0, 0.0001)
                    new_price = current_price * (1 + change)
                    self.current_prices[pair] = new_price
                    
                    # Add to history
                    new_data = {
                        'timestamp': datetime.now(),
                        'open': current_price,
                        'high': max(current_price, new_price),
                        'low': min(current_price, new_price),
                        'close': new_price,
                        'volume': np.random.randint(1000, 5000)
                    }
                    
                    # Add to history and keep only last 200 points
                    self.price_history[pair] = pd.concat([
                        self.price_history[pair], 
                        pd.DataFrame([new_data])
                    ]).tail(200).reset_index(drop=True)
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logging.error(f"Error in price update loop: {e}")
                time.sleep(5)
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available currency pairs"""
        return self.available_pairs.copy()
    
    def get_market_data_df(self, pair: str) -> Optional[pd.DataFrame]:
        """Get market data for a specific pair"""
        try:
            if pair not in self.price_history:
                return None
            
            return self.price_history[pair].copy()
            
        except Exception as e:
            logging.error(f"Error getting market data for {pair}: {e}")
            return None
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for a pair"""
        try:
            return self.current_prices.get(pair)
        except Exception as e:
            logging.error(f"Error getting current price for {pair}: {e}")
            return None
    
    def place_demo_trade(self, pair: str, direction: str, amount: float) -> Dict:
        """Place a demo trade"""
        try:
            current_price = self.get_current_price(pair)
            if current_price is None:
                return {'success': False, 'error': 'Invalid pair'}
            
            # Simulate trade result (50% win rate for demo)
            is_win = np.random.random() > 0.5
            
            trade_result = {
                'success': True,
                'pair': pair,
                'direction': direction,
                'amount': amount,
                'entry_price': current_price,
                'result': 'win' if is_win else 'loss',
                'profit': amount * 0.8 if is_win else -amount,
                'timestamp': datetime.now()
            }
            
            return trade_result
            
        except Exception as e:
            logging.error(f"Error placing demo trade: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_account_info(self) -> Dict:
        """Get demo account information"""
        return {
            'balance': 10000.0,
            'currency': 'USD',
            'demo': True,
            'account_type': 'demo'
        }