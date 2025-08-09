import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import time
from datetime import datetime, timedelta
import re
import json

class SentimentAnalysis:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.news_sources = [
            'https://api.marketaux.com/v1/news',
            'https://newsapi.org/v2/everything',
            'https://api.polygon.io/v2/reference/news'
        ]
        self.sentiment_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
    def get_market_news(self, symbols=None, hours_back=24):
        """Fetch market news for sentiment analysis"""
        try:
            if symbols is None:
                symbols = ['FOREX', 'USD', 'EUR', 'GBP', 'JPY']
            
            news_data = []
            
            # Try multiple news sources
            for symbol in symbols:
                try:
                    # Simulate news fetching (in production, use actual APIs)
                    fake_news = self.generate_sample_news(symbol)
                    news_data.extend(fake_news)
                except Exception as e:
                    logging.warning(f"Error fetching news for {symbol}: {e}")
                    continue
            
            return news_data
            
        except Exception as e:
            logging.error(f"Error fetching market news: {e}")
            return []
    
    def generate_sample_news(self, symbol):
        """Generate sample news data for testing (replace with real APIs)"""
        sample_news = [
            {
                'title': f'{symbol} shows strong performance amid economic recovery',
                'description': f'Market analysts predict continued growth for {symbol} as economic indicators improve',
                'source': 'Financial Times',
                'published_at': datetime.now() - timedelta(hours=2),
                'url': 'https://example.com/news1'
            },
            {
                'title': f'Central bank policy impacts {symbol} trading volumes',
                'description': f'Recent monetary policy decisions are affecting {symbol} market dynamics',
                'source': 'Reuters',
                'published_at': datetime.now() - timedelta(hours=5),
                'url': 'https://example.com/news2'
            },
            {
                'title': f'{symbol} volatility increases due to geopolitical tensions',
                'description': f'International events are creating uncertainty in {symbol} markets',
                'source': 'Bloomberg',
                'published_at': datetime.now() - timedelta(hours=8),
                'url': 'https://example.com/news3'
            }
        ]
        return sample_news
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of a given text using multiple methods"""
        try:
            if not text or len(text.strip()) == 0:
                return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1}
            
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # VADER sentiment analysis
            vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
            
            # TextBlob sentiment analysis
            blob = TextBlob(cleaned_text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Combine both methods
            combined_sentiment = {
                'compound': (vader_scores['compound'] + textblob_polarity) / 2,
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'subjectivity': textblob_subjectivity,
                'confidence': abs(vader_scores['compound']) * textblob_subjectivity
            }
            
            return combined_sentiment
            
        except Exception as e:
            logging.error(f"Error analyzing text sentiment: {e}")
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1}
    
    def clean_text(self, text):
        """Clean and preprocess text for sentiment analysis"""
        try:
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove special characters but keep punctuation for sentiment
            text = re.sub(r'[^\w\s.,!?]', '', text)
            
            return text
            
        except Exception as e:
            logging.error(f"Error cleaning text: {e}")
            return text
    
    def calculate_news_sentiment(self, symbol):
        """Calculate overall sentiment from news articles"""
        try:
            # Check cache first
            cache_key = f"{symbol}_news"
            if self.is_cached_valid(cache_key):
                return self.sentiment_cache[cache_key]['data']
            
            # Fetch news
            news_data = self.get_market_news([symbol])
            
            if not news_data:
                # Return neutral sentiment if no news
                return {
                    'overall_sentiment': 0.0,
                    'sentiment_strength': 0.0,
                    'positive_ratio': 0.33,
                    'negative_ratio': 0.33,
                    'neutral_ratio': 0.34,
                    'articles_count': 0,
                    'confidence': 0.0
                }
            
            sentiments = []
            total_confidence = 0
            
            for article in news_data:
                # Combine title and description for analysis
                full_text = f"{article.get('title', '')} {article.get('description', '')}"
                
                sentiment = self.analyze_text_sentiment(full_text)
                sentiments.append(sentiment)
                total_confidence += sentiment.get('confidence', 0)
            
            # Calculate aggregate sentiment
            if sentiments:
                avg_compound = np.mean([s['compound'] for s in sentiments])
                avg_positive = np.mean([s['positive'] for s in sentiments])
                avg_negative = np.mean([s['negative'] for s in sentiments])
                avg_neutral = np.mean([s['neutral'] for s in sentiments])
                avg_confidence = total_confidence / len(sentiments) if sentiments else 0
                
                # Calculate sentiment strength (how strong the sentiment is)
                sentiment_strength = abs(avg_compound)
                
                result = {
                    'overall_sentiment': avg_compound,
                    'sentiment_strength': sentiment_strength,
                    'positive_ratio': avg_positive,
                    'negative_ratio': avg_negative,
                    'neutral_ratio': avg_neutral,
                    'articles_count': len(sentiments),
                    'confidence': min(avg_confidence, 1.0)
                }
            else:
                result = {
                    'overall_sentiment': 0.0,
                    'sentiment_strength': 0.0,
                    'positive_ratio': 0.33,
                    'negative_ratio': 0.33,
                    'neutral_ratio': 0.34,
                    'articles_count': 0,
                    'confidence': 0.0
                }
            
            # Cache the result
            self.cache_sentiment(cache_key, result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error calculating news sentiment: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'positive_ratio': 0.33,
                'negative_ratio': 0.33,
                'neutral_ratio': 0.34,
                'articles_count': 0,
                'confidence': 0.0
            }
    
    def get_social_media_sentiment(self, symbol):
        """Simulate social media sentiment analysis"""
        try:
            # In production, this would connect to Twitter API, Reddit API, etc.
            # For now, we'll simulate sentiment based on recent price action
            
            # Generate simulated social media sentiment
            base_sentiment = np.random.normal(0, 0.3)  # Random sentiment around neutral
            
            # Add some market-based adjustments
            time_factor = np.sin(time.time() / 3600) * 0.1  # Hourly variation
            
            overall_sentiment = np.clip(base_sentiment + time_factor, -1, 1)
            
            result = {
                'overall_sentiment': overall_sentiment,
                'sentiment_strength': abs(overall_sentiment),
                'positive_ratio': max(0, overall_sentiment + 0.5),
                'negative_ratio': max(0, -overall_sentiment + 0.5),
                'neutral_ratio': 1 - abs(overall_sentiment),
                'posts_count': np.random.randint(50, 200),
                'confidence': abs(overall_sentiment)
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error getting social media sentiment: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'positive_ratio': 0.33,
                'negative_ratio': 0.33,
                'neutral_ratio': 0.34,
                'posts_count': 0,
                'confidence': 0.0
            }
    
    def calculate_market_fear_greed_index(self):
        """Calculate a simplified fear and greed index"""
        try:
            # Simulate market fear/greed based on multiple factors
            factors = {
                'volatility': np.random.uniform(0.2, 0.8),
                'momentum': np.random.uniform(0.3, 0.7),
                'volume': np.random.uniform(0.4, 0.6),
                'safe_haven': np.random.uniform(0.3, 0.7),
                'sentiment': np.random.uniform(0.2, 0.8)
            }
            
            # Weight the factors
            weights = {
                'volatility': 0.25,
                'momentum': 0.25,
                'volume': 0.15,
                'safe_haven': 0.15,
                'sentiment': 0.20
            }
            
            # Calculate weighted average
            fear_greed_score = sum(factors[key] * weights[key] for key in factors)
            
            # Convert to 0-100 scale
            fear_greed_index = fear_greed_score * 100
            
            # Determine sentiment label
            if fear_greed_index <= 25:
                label = "Extreme Fear"
            elif fear_greed_index <= 45:
                label = "Fear"
            elif fear_greed_index <= 55:
                label = "Neutral"
            elif fear_greed_index <= 75:
                label = "Greed"
            else:
                label = "Extreme Greed"
            
            return {
                'index': fear_greed_index,
                'label': label,
                'factors': factors
            }
            
        except Exception as e:
            logging.error(f"Error calculating fear/greed index: {e}")
            return {
                'index': 50,
                'label': "Neutral",
                'factors': {}
            }
    
    def get_comprehensive_sentiment(self, symbol):
        """Get comprehensive sentiment analysis for a symbol"""
        try:
            # Get news sentiment
            news_sentiment = self.calculate_news_sentiment(symbol)
            
            # Get social media sentiment
            social_sentiment = self.get_social_media_sentiment(symbol)
            
            # Get market fear/greed index
            fear_greed = self.calculate_market_fear_greed_index()
            
            # Combine all sentiment sources
            combined_sentiment = self.combine_sentiment_sources(
                news_sentiment, 
                social_sentiment, 
                fear_greed
            )
            
            return combined_sentiment
            
        except Exception as e:
            logging.error(f"Error getting comprehensive sentiment: {e}")
            return self.get_neutral_sentiment()
    
    def combine_sentiment_sources(self, news_sentiment, social_sentiment, fear_greed):
        """Combine multiple sentiment sources into overall sentiment"""
        try:
            # Weight the different sources
            news_weight = 0.4
            social_weight = 0.35
            fear_greed_weight = 0.25
            
            # Normalize fear/greed index to -1 to 1 scale
            normalized_fear_greed = (fear_greed['index'] - 50) / 50
            
            # Calculate weighted sentiment
            overall_sentiment = (
                news_sentiment['overall_sentiment'] * news_weight +
                social_sentiment['overall_sentiment'] * social_weight +
                normalized_fear_greed * fear_greed_weight
            )
            
            # Calculate confidence based on all sources
            overall_confidence = (
                news_sentiment['confidence'] * news_weight +
                social_sentiment['confidence'] * social_weight +
                0.7 * fear_greed_weight  # Fixed confidence for fear/greed
            )
            
            # Determine sentiment direction and strength
            if overall_sentiment > 0.1:
                direction = "BULLISH"
            elif overall_sentiment < -0.1:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
            
            sentiment_strength = abs(overall_sentiment) * 10  # Scale to 0-10
            
            result = {
                'overall_sentiment': overall_sentiment,
                'direction': direction,
                'strength': min(sentiment_strength, 10),
                'confidence': overall_confidence,
                'sources': {
                    'news': news_sentiment,
                    'social_media': social_sentiment,
                    'fear_greed': fear_greed
                },
                'market_mood': self.interpret_sentiment(overall_sentiment),
                'trading_impact': self.assess_trading_impact(overall_sentiment, overall_confidence)
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error combining sentiment sources: {e}")
            return self.get_neutral_sentiment()
    
    def interpret_sentiment(self, sentiment_score):
        """Interpret sentiment score into human-readable format"""
        if sentiment_score >= 0.6:
            return "Very Positive"
        elif sentiment_score >= 0.2:
            return "Positive"
        elif sentiment_score >= -0.2:
            return "Neutral"
        elif sentiment_score >= -0.6:
            return "Negative"
        else:
            return "Very Negative"
    
    def assess_trading_impact(self, sentiment_score, confidence):
        """Assess potential trading impact of sentiment"""
        impact_score = abs(sentiment_score) * confidence
        
        if impact_score >= 0.7:
            return "High Impact"
        elif impact_score >= 0.4:
            return "Medium Impact"
        elif impact_score >= 0.2:
            return "Low Impact"
        else:
            return "Minimal Impact"
    
    def get_neutral_sentiment(self):
        """Return neutral sentiment when analysis fails"""
        return {
            'overall_sentiment': 0.0,
            'direction': "NEUTRAL",
            'strength': 0.0,
            'confidence': 0.0,
            'sources': {
                'news': {'overall_sentiment': 0.0, 'confidence': 0.0},
                'social_media': {'overall_sentiment': 0.0, 'confidence': 0.0},
                'fear_greed': {'index': 50, 'label': 'Neutral'}
            },
            'market_mood': "Neutral",
            'trading_impact': "Minimal Impact"
        }
    
    def is_cached_valid(self, cache_key):
        """Check if cached sentiment is still valid"""
        if cache_key not in self.sentiment_cache:
            return False
        
        cache_time = self.sentiment_cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_expiry
    
    def cache_sentiment(self, cache_key, data):
        """Cache sentiment data"""
        self.sentiment_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def get_sentiment_bias(self, sentiment_data):
        """Calculate sentiment bias for trading decisions"""
        try:
            sentiment_score = sentiment_data.get('overall_sentiment', 0)
            confidence = sentiment_data.get('confidence', 0)
            
            # Calculate bias factor (how much sentiment should influence trading)
            bias_factor = sentiment_score * confidence
            
            # Determine bias direction
            if bias_factor > 0.3:
                bias = "BULLISH_BIAS"
                strength = min(bias_factor * 100, 95)
            elif bias_factor < -0.3:
                bias = "BEARISH_BIAS"
                strength = min(abs(bias_factor) * 100, 95)
            else:
                bias = "NEUTRAL_BIAS"
                strength = 50
            
            return {
                'bias': bias,
                'strength': strength,
                'factor': bias_factor
            }
            
        except Exception as e:
            logging.error(f"Error calculating sentiment bias: {e}")
            return {
                'bias': "NEUTRAL_BIAS",
                'strength': 50,
                'factor': 0
            }