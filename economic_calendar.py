import requests
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass

@dataclass
class EconomicEvent:
    event_id: str
    title: str
    currency: str
    impact: str  # 'high', 'medium', 'low'
    actual: Optional[float]
    forecast: Optional[float]
    previous: Optional[float]
    timestamp: datetime
    description: str
    source: str

class EconomicCalendar:
    def __init__(self):
        self.api_key = None  # Will be set from config
        self.events_cache = {}
        self.cache_expiry = 300  # 5 minutes
        self.currency_impact_map = {
            'USD': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD'],
            'EUR': ['EUR/USD', 'EUR/GBP', 'EUR/JPY', 'EUR/CHF'],
            'GBP': ['GBP/USD', 'EUR/GBP', 'GBP/JPY'],
            'JPY': ['USD/JPY', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY'],
            'AUD': ['AUD/USD', 'AUD/JPY'],
            'CAD': ['USD/CAD'],
            'CHF': ['USD/CHF', 'EUR/CHF'],
            'NZD': ['NZD/USD']
        }
        
        # Impact scoring
        self.impact_scores = {
            'high': 3,
            'medium': 2,
            'low': 1
        }
        
        # Economic indicators and their typical impact
        self.indicator_impact = {
            'Non-Farm Payrolls': {'impact': 'high', 'volatility': 0.8},
            'CPI': {'impact': 'high', 'volatility': 0.7},
            'GDP': {'impact': 'high', 'volatility': 0.6},
            'Interest Rate Decision': {'impact': 'high', 'volatility': 0.9},
            'Unemployment Rate': {'impact': 'medium', 'volatility': 0.5},
            'Retail Sales': {'impact': 'medium', 'volatility': 0.4},
            'PMI': {'impact': 'medium', 'volatility': 0.3},
            'Trade Balance': {'impact': 'medium', 'volatility': 0.4}
        }
        
    def set_api_key(self, api_key: str):
        """Set API key for economic calendar services"""
        self.api_key = api_key
    
    def get_economic_events(self, hours_ahead: int = 24) -> List[EconomicEvent]:
        """Fetch economic events for the next N hours"""
        try:
            cache_key = f"events_{hours_ahead}"
            
            # Check cache first
            if self.is_cache_valid(cache_key):
                return self.events_cache[cache_key]['data']
            
            # Fetch from multiple sources
            events = []
            
            # Try primary API (MarketAux)
            try:
                events.extend(self.fetch_marketaux_events(hours_ahead))
            except Exception as e:
                logging.warning(f"Failed to fetch from MarketAux: {e}")
            
            # Try secondary API (NewsAPI)
            try:
                events.extend(self.fetch_newsapi_events(hours_ahead))
            except Exception as e:
                logging.warning(f"Failed to fetch from NewsAPI: {e}")
            
            # Fallback to simulated events
            if not events:
                events = self.generate_simulated_events(hours_ahead)
            
            # Cache the results
            self.cache_events(cache_key, events)
            
            return events
            
        except Exception as e:
            logging.error(f"Error fetching economic events: {e}")
            return self.generate_simulated_events(hours_ahead)
    
    def fetch_marketaux_events(self, hours_ahead: int) -> List[EconomicEvent]:
        """Fetch events from MarketAux API"""
        try:
            if not self.api_key:
                return []
            
            # Calculate date range
            now = datetime.now()
            end_date = now + timedelta(hours=hours_ahead)
            
            url = "https://api.marketaux.com/v1/events/economic"
            params = {
                'api_token': self.api_key,
                'filter_currencies': 'USD,EUR,GBP,JPY,AUD,CAD,CHF,NZD',
                'date_from': now.strftime('%Y-%m-%d'),
                'date_to': end_date.strftime('%Y-%m-%d'),
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            events = []
            
            for event_data in data.get('data', []):
                event = self.parse_marketaux_event(event_data)
                if event:
                    events.append(event)
            
            return events
            
        except Exception as e:
            logging.error(f"Error fetching from MarketAux: {e}")
            return []
    
    def fetch_newsapi_events(self, hours_ahead: int) -> List[EconomicEvent]:
        """Fetch events from NewsAPI"""
        try:
            if not self.api_key:
                return []
            
            # NewsAPI doesn't have economic calendar, so we'll search for economic news
            url = "https://newsapi.org/v2/everything"
            params = {
                'apiKey': self.api_key,
                'q': 'economic calendar OR "interest rate" OR "GDP" OR "CPI" OR "employment"',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            events = []
            
            for article in data.get('articles', []):
                event = self.parse_newsapi_article(article)
                if event:
                    events.append(event)
            
            return events
            
        except Exception as e:
            logging.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    def parse_marketaux_event(self, event_data: Dict) -> Optional[EconomicEvent]:
        """Parse MarketAux event data"""
        try:
            event_id = event_data.get('id', '')
            title = event_data.get('title', '')
            currency = event_data.get('currency', '')
            impact = event_data.get('impact', 'low')
            
            # Parse values
            actual = self.parse_value(event_data.get('actual'))
            forecast = self.parse_value(event_data.get('forecast'))
            previous = self.parse_value(event_data.get('previous'))
            
            # Parse timestamp
            timestamp_str = event_data.get('date', '')
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            description = event_data.get('description', '')
            source = 'MarketAux'
            
            return EconomicEvent(
                event_id=event_id,
                title=title,
                currency=currency,
                impact=impact,
                actual=actual,
                forecast=forecast,
                previous=previous,
                timestamp=timestamp,
                description=description,
                source=source
            )
            
        except Exception as e:
            logging.error(f"Error parsing MarketAux event: {e}")
            return None
    
    def parse_newsapi_article(self, article: Dict) -> Optional[EconomicEvent]:
        """Parse NewsAPI article as economic event"""
        try:
            title = article.get('title', '')
            
            # Check if this is an economic event
            if not self.is_economic_event(title):
                return None
            
            # Extract currency and impact
            currency = self.extract_currency_from_title(title)
            impact = self.estimate_impact_from_title(title)
            
            # Generate event ID
            event_id = f"news_{hash(title)}"
            
            # Parse timestamp
            timestamp_str = article.get('publishedAt', '')
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            return EconomicEvent(
                event_id=event_id,
                title=title,
                currency=currency,
                impact=impact,
                actual=None,
                forecast=None,
                previous=None,
                timestamp=timestamp,
                description=article.get('description', ''),
                source='NewsAPI'
            )
            
        except Exception as e:
            logging.error(f"Error parsing NewsAPI article: {e}")
            return None
    
    def generate_simulated_events(self, hours_ahead: int) -> List[EconomicEvent]:
        """Generate simulated economic events for testing"""
        try:
            events = []
            now = datetime.now()
            
            # Common economic indicators
            indicators = [
                ('Non-Farm Payrolls', 'USD', 'high'),
                ('CPI', 'USD', 'high'),
                ('GDP', 'EUR', 'high'),
                ('Interest Rate Decision', 'GBP', 'high'),
                ('Unemployment Rate', 'USD', 'medium'),
                ('Retail Sales', 'EUR', 'medium'),
                ('PMI Manufacturing', 'JPY', 'medium'),
                ('Trade Balance', 'AUD', 'medium')
            ]
            
            for i, (indicator, currency, impact) in enumerate(indicators):
                # Schedule events over the next hours
                event_time = now + timedelta(hours=i * 2)
                
                if event_time <= now + timedelta(hours=hours_ahead):
                    # Generate realistic values
                    if 'Payrolls' in indicator:
                        actual = np.random.randint(150, 250)
                        forecast = actual + np.random.randint(-20, 20)
                        previous = actual + np.random.randint(-30, 30)
                    elif 'CPI' in indicator:
                        actual = round(np.random.uniform(2.0, 4.0), 1)
                        forecast = actual + np.random.uniform(-0.3, 0.3)
                        previous = actual + np.random.uniform(-0.5, 0.5)
                    elif 'GDP' in indicator:
                        actual = round(np.random.uniform(1.0, 3.0), 1)
                        forecast = actual + np.random.uniform(-0.5, 0.5)
                        previous = actual + np.random.uniform(-1.0, 1.0)
                    else:
                        actual = round(np.random.uniform(50, 60), 1)
                        forecast = actual + np.random.uniform(-2, 2)
                        previous = actual + np.random.uniform(-3, 3)
                    
                    event = EconomicEvent(
                        event_id=f"sim_{i}",
                        title=indicator,
                        currency=currency,
                        impact=impact,
                        actual=actual,
                        forecast=forecast,
                        previous=previous,
                        timestamp=event_time,
                        description=f"Simulated {indicator} release for {currency}",
                        source='Simulated'
                    )
                    
                    events.append(event)
            
            return events
            
        except Exception as e:
            logging.error(f"Error generating simulated events: {e}")
            return []
    
    def is_economic_event(self, title: str) -> bool:
        """Check if a news title represents an economic event"""
        economic_keywords = [
            'GDP', 'CPI', 'employment', 'payrolls', 'unemployment',
            'interest rate', 'inflation', 'retail sales', 'trade balance',
            'PMI', 'manufacturing', 'services', 'housing', 'consumer'
        ]
        
        title_lower = title.lower()
        return any(keyword.lower() in title_lower for keyword in economic_keywords)
    
    def extract_currency_from_title(self, title: str) -> str:
        """Extract currency from news title"""
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
        
        for currency in currencies:
            if currency in title:
                return currency
        
        return 'USD'  # Default
    
    def estimate_impact_from_title(self, title: str) -> str:
        """Estimate impact level from news title"""
        title_lower = title.lower()
        
        high_impact_keywords = ['GDP', 'payrolls', 'CPI', 'interest rate', 'inflation']
        medium_impact_keywords = ['employment', 'retail', 'trade', 'PMI', 'manufacturing']
        
        if any(keyword in title_lower for keyword in high_impact_keywords):
            return 'high'
        elif any(keyword in title_lower for keyword in medium_impact_keywords):
            return 'medium'
        else:
            return 'low'
    
    def parse_value(self, value_str: str) -> Optional[float]:
        """Parse string value to float"""
        try:
            if not value_str or value_str == 'N/A':
                return None
            return float(value_str.replace('%', '').replace(',', ''))
        except:
            return None
    
    def get_events_for_currency_pair(self, pair: str, hours_ahead: int = 24) -> List[EconomicEvent]:
        """Get economic events that could affect a specific currency pair"""
        try:
            all_events = self.get_economic_events(hours_ahead)
            relevant_events = []
            
            # Get currencies for this pair
            base_currency = pair.split('/')[0]
            quote_currency = pair.split('/')[1]
            
            for event in all_events:
                # Check if event affects either currency in the pair
                if event.currency in [base_currency, quote_currency]:
                    relevant_events.append(event)
            
            # Sort by timestamp and impact
            relevant_events.sort(key=lambda x: (x.timestamp, self.impact_scores.get(x.impact, 0)), reverse=True)
            
            return relevant_events
            
        except Exception as e:
            logging.error(f"Error getting events for pair {pair}: {e}")
            return []
    
    def calculate_market_impact(self, pair: str, events: List[EconomicEvent]) -> Dict:
        """Calculate potential market impact of economic events"""
        try:
            if not events:
                return self.get_default_market_impact()
            
            total_impact_score = 0
            high_impact_events = 0
            medium_impact_events = 0
            low_impact_events = 0
            
            upcoming_events = []
            recent_events = []
            
            now = datetime.now()
            
            for event in events:
                # Calculate impact score
                impact_score = self.impact_scores.get(event.impact, 0)
                total_impact_score += impact_score
                
                # Count events by impact
                if event.impact == 'high':
                    high_impact_events += 1
                elif event.impact == 'medium':
                    medium_impact_events += 1
                else:
                    low_impact_events += 1
                
                # Categorize by timing
                time_diff = (event.timestamp - now).total_seconds() / 3600  # hours
                
                if time_diff > 0 and time_diff <= 24:
                    upcoming_events.append(event)
                elif time_diff <= 0 and time_diff > -24:
                    recent_events.append(event)
            
            # Calculate volatility expectation
            volatility_score = min(total_impact_score / 10 * 100, 100)
            
            # Determine market sentiment
            sentiment = self.calculate_event_sentiment(recent_events)
            
            return {
                'total_impact_score': total_impact_score,
                'volatility_expectation': volatility_score,
                'high_impact_events': high_impact_events,
                'medium_impact_events': medium_impact_events,
                'low_impact_events': low_impact_events,
                'upcoming_events': len(upcoming_events),
                'recent_events': len(recent_events),
                'market_sentiment': sentiment,
                'trading_recommendation': self.get_trading_recommendation(volatility_score, sentiment)
            }
            
        except Exception as e:
            logging.error(f"Error calculating market impact: {e}")
            return self.get_default_market_impact()
    
    def calculate_event_sentiment(self, events: List[EconomicEvent]) -> str:
        """Calculate market sentiment from recent economic events"""
        try:
            if not events:
                return 'neutral'
            
            sentiment_score = 0
            
            for event in events:
                if event.actual is not None and event.forecast is not None:
                    # Compare actual vs forecast
                    if event.actual > event.forecast:
                        sentiment_score += 1
                    elif event.actual < event.forecast:
                        sentiment_score -= 1
            
            # Normalize sentiment
            if sentiment_score > 0:
                return 'bullish'
            elif sentiment_score < 0:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logging.error(f"Error calculating event sentiment: {e}")
            return 'neutral'
    
    def get_trading_recommendation(self, volatility_score: float, sentiment: str) -> str:
        """Get trading recommendation based on economic events"""
        try:
            if volatility_score > 70:
                return 'avoid_trading'
            elif volatility_score > 50:
                if sentiment == 'bullish':
                    return 'cautious_buy'
                elif sentiment == 'bearish':
                    return 'cautious_sell'
                else:
                    return 'wait_for_clearer_signals'
            else:
                return 'normal_trading'
                
        except Exception as e:
            logging.error(f"Error getting trading recommendation: {e}")
            return 'normal_trading'
    
    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        try:
            if cache_key not in self.events_cache:
                return False
            
            cache_time = self.events_cache[cache_key]['timestamp']
            return (datetime.now() - cache_time).total_seconds() < self.cache_expiry
            
        except Exception as e:
            logging.error(f"Error checking cache validity: {e}")
            return False
    
    def cache_events(self, cache_key: str, events: List[EconomicEvent]):
        """Cache economic events"""
        try:
            self.events_cache[cache_key] = {
                'data': events,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logging.error(f"Error caching events: {e}")
    
    def get_default_market_impact(self) -> Dict:
        """Return default market impact data"""
        return {
            'total_impact_score': 0,
            'volatility_expectation': 0,
            'high_impact_events': 0,
            'medium_impact_events': 0,
            'low_impact_events': 0,
            'upcoming_events': 0,
            'recent_events': 0,
            'market_sentiment': 'neutral',
            'trading_recommendation': 'normal_trading'
        }