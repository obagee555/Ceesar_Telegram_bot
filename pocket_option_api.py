import requests
import websocket
import json
import threading
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
from config import *
import numpy as np

class PocketOptionAPI:
    def __init__(self):
        self.ssid = POCKET_OPTION_SSID
        self.base_url = "https://po.trade"
        self.ws_url = "wss://ws.po.trade/socket.io/?EIO=4&transport=websocket"
        self.session = requests.Session()
        self.ws = None
        self.is_connected = False
        self.market_data = {}
        self.subscribers = []
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        self.setup_authentication()
        
    def setup_authentication(self):
        """Setup authentication using SSID"""
        try:
            # Parse SSID data
            ssid_data = json.loads(self.ssid[2:])  # Remove '42' prefix
            
            self.auth_data = ssid_data[1]
            self.session_id = self.auth_data['session']
            self.uid = self.auth_data['uid']
            self.platform = self.auth_data['platform']
            
            logging.info(f"Authentication setup completed for UID: {self.uid}")
            
        except Exception as e:
            logging.error(f"Error setting up authentication: {e}")
    
    def connect_websocket(self):
        """Connect to Pocket Option WebSocket for real-time data"""
        try:
            def on_message(ws, message):
                self.handle_websocket_message(message)
            
            def on_error(ws, error):
                logging.error(f"WebSocket error: {error}")
                self.is_connected = False
            
            def on_close(ws, close_status_code, close_msg):
                logging.info("WebSocket connection closed")
                self.is_connected = False
            
            def on_open(ws):
                logging.info("WebSocket connection established")
                self.is_connected = True
                self.authenticate_websocket()
                self.subscribe_to_market_data()
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Error connecting to WebSocket: {e}")
            return False
    
    def authenticate_websocket(self):
        """Authenticate WebSocket connection"""
        try:
            auth_message = f'42["auth",{json.dumps(self.auth_data)}]'
            self.ws.send(auth_message)
            logging.info("WebSocket authentication sent")
            
        except Exception as e:
            logging.error(f"Error authenticating WebSocket: {e}")
    
    def subscribe_to_market_data(self):
        """Subscribe to real-time market data for all currency pairs"""
        try:
            # Get current day type (weekday/weekend)
            current_day = datetime.now().weekday()
            is_weekend = current_day >= 5  # Saturday = 5, Sunday = 6
            
            if is_weekend:
                pairs = CURRENCY_PAIRS['weekends']
            else:
                pairs = CURRENCY_PAIRS['weekdays']
            
            for pair in pairs:
                # Subscribe to price updates
                subscribe_msg = f'42["subscribe","{pair}"]'
                self.ws.send(subscribe_msg)
                
                # Initialize market data structure
                self.market_data[pair] = {
                    'prices': [],
                    'volumes': [],
                    'timestamps': [],
                    'current_price': 0,
                    'volatility': 0,
                    'last_update': None
                }
            
            logging.info(f"Subscribed to {len(pairs)} currency pairs")
            
        except Exception as e:
            logging.error(f"Error subscribing to market data: {e}")
    
    def handle_websocket_message(self, message):
        """Handle incoming WebSocket messages"""
        try:
            if message.startswith('42'):
                # Parse Socket.IO message
                data = json.loads(message[2:])
                event_type = data[0]
                event_data = data[1] if len(data) > 1 else None
                
                if event_type == "price_update":
                    self.process_price_update(event_data)
                elif event_type == "market_data":
                    self.process_market_data(event_data)
                    
        except Exception as e:
            logging.error(f"Error handling WebSocket message: {e}")
    
    def process_price_update(self, data):
        """Process real-time price updates"""
        try:
            if not data:
                return
                
            symbol = data.get('symbol')
            price = float(data.get('price', 0))
            volume = float(data.get('volume', 0))
            timestamp = datetime.now()
            
            if symbol in self.market_data:
                market_info = self.market_data[symbol]
                
                # Update current price
                market_info['current_price'] = price
                market_info['last_update'] = timestamp
                
                # Store historical data (keep last 1000 points)
                market_info['prices'].append(price)
                market_info['volumes'].append(volume)
                market_info['timestamps'].append(timestamp)
                
                # Limit history size
                if len(market_info['prices']) > 1000:
                    market_info['prices'] = market_info['prices'][-1000:]
                    market_info['volumes'] = market_info['volumes'][-1000:]
                    market_info['timestamps'] = market_info['timestamps'][-1000:]
                
                # Calculate volatility
                if len(market_info['prices']) >= 20:
                    recent_prices = market_info['prices'][-20:]
                    volatility = np.std(recent_prices) / np.mean(recent_prices)
                    market_info['volatility'] = volatility
                
                # Notify subscribers
                self.notify_subscribers(symbol, market_info)
                
        except Exception as e:
            logging.error(f"Error processing price update: {e}")
    
    def process_market_data(self, data):
        """Process general market data"""
        try:
            if not data:
                return
                
            # Process any additional market data
            logging.debug(f"Received market data: {data}")
            
        except Exception as e:
            logging.error(f"Error processing market data: {e}")
    
    def get_historical_data(self, symbol, timeframe='1m', limit=500):
        """Get historical price data"""
        try:
            # Construct API endpoint
            endpoint = f"{self.base_url}/api/v1/history/{symbol}"
            
            params = {
                'timeframe': timeframe,
                'limit': limit,
                'session': self.session_id
            }
            
            response = self.session.get(endpoint, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    # Convert to DataFrame
                    df = pd.DataFrame(data['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                    
                    return df
                else:
                    logging.warning(f"No historical data found for {symbol}")
                    return None
            else:
                logging.error(f"Error fetching historical data: {response.status_code}")
                return None
                
        except Exception as e:
            logging.error(f"Error getting historical data: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            if symbol in self.market_data:
                return self.market_data[symbol]['current_price']
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error getting current price: {e}")
            return None
    
    def get_market_data_df(self, symbol, limit=100):
        """Get market data as pandas DataFrame"""
        try:
            if symbol not in self.market_data:
                return None
                
            market_info = self.market_data[symbol]
            
            if len(market_info['prices']) < limit:
                limit = len(market_info['prices'])
            
            if limit == 0:
                return None
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': market_info['timestamps'][-limit:],
                'close': market_info['prices'][-limit:],
                'volume': market_info['volumes'][-limit:]
            })
            
            df = df.set_index('timestamp')
            return df
            
        except Exception as e:
            logging.error(f"Error getting market data DataFrame: {e}")
            return None
    
    def get_volatility(self, symbol):
        """Get current volatility for a symbol"""
        try:
            if symbol in self.market_data:
                return self.market_data[symbol]['volatility']
            else:
                return 0
                
        except Exception as e:
            logging.error(f"Error getting volatility: {e}")
            return 0
    
    def is_low_volatility(self, symbol):
        """Check if current volatility is low"""
        try:
            volatility = self.get_volatility(symbol)
            return volatility < VOLATILITY_THRESHOLD
            
        except Exception as e:
            logging.error(f"Error checking volatility: {e}")
            return False
    
    def get_market_time(self):
        """Get current market time with timezone offset"""
        try:
            # OTC markets are UTC-4
            utc_time = datetime.utcnow()
            market_time = utc_time - timedelta(hours=4)
            return market_time
            
        except Exception as e:
            logging.error(f"Error getting market time: {e}")
            return datetime.now()
    
    def get_expiry_time(self, signal_time=None):
        """Calculate expiry time based on signal time"""
        try:
            if signal_time is None:
                signal_time = self.get_market_time()
            
            # Add 2 minutes for expiry
            expiry_time = signal_time + timedelta(seconds=EXPIRY_TIME)
            return expiry_time
            
        except Exception as e:
            logging.error(f"Error calculating expiry time: {e}")
            return None
    
    def subscribe_to_updates(self, callback):
        """Subscribe to market data updates"""
        self.subscribers.append(callback)
    
    def notify_subscribers(self, symbol, market_data):
        """Notify all subscribers of market updates"""
        for callback in self.subscribers:
            try:
                callback(symbol, market_data)
            except Exception as e:
                logging.error(f"Error notifying subscriber: {e}")
    
    def start(self):
        """Start the API connection"""
        try:
            logging.info("Starting Pocket Option API...")
            success = self.connect_websocket()
            
            if success:
                # Wait for connection to establish
                time.sleep(2)
                return self.is_connected
            else:
                return False
                
        except Exception as e:
            logging.error(f"Error starting Pocket Option API: {e}")
            return False
    
    def stop(self):
        """Stop the API connection"""
        try:
            if self.ws:
                self.ws.close()
            self.is_connected = False
            logging.info("Pocket Option API stopped")
            
        except Exception as e:
            logging.error(f"Error stopping Pocket Option API: {e}")
    
    def get_available_pairs(self):
        """Get list of available currency pairs"""
        try:
            current_day = datetime.now().weekday()
            is_weekend = current_day >= 5
            
            if is_weekend:
                return CURRENCY_PAIRS['weekends']
            else:
                return CURRENCY_PAIRS['weekdays']
                
        except Exception as e:
            logging.error(f"Error getting available pairs: {e}")
            return []