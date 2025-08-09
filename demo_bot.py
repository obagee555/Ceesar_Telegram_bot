#!/usr/bin/env python3
"""
Demo Simplified Advanced Binary Trading Bot
Enhanced AI-Powered Signal Generator with Core Components

Author: AI Trading Systems
Version: 2.0 - Demo Edition
License: Private Use Only
"""

import sys
import os
import asyncio
import logging
from datetime import datetime
import time

# Add workspace to Python path
sys.path.append('/workspace')

# Import simplified components
from simplified_signal_generator import SimplifiedSignalGenerator
from config import *

def setup_logging():
    """Setup comprehensive logging"""
    try:
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/demo_bot.log'),
                logging.StreamHandler()
            ]
        )
        
        logging.info("Demo logging system initialized")
        
    except Exception as e:
        print(f"Error setting up logging: {e}")

def print_demo_banner():
    """Print demo startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║      🤖 DEMO SIMPLIFIED ADVANCED BINARY TRADING BOT v2.0                    ║
    ║                                                                              ║
    ║      🎯 Enhanced AI-Powered Signal Generator                                ║
    ║      📊 Advanced Technical Analysis                                         ║
    ║      🧠 Simplified LSTM Ensemble (Random Forest, Gradient Boosting, SVR)    ║
    ║      📈 Market Sentiment Analysis                                           ║
    ║      ⚠️  Advanced Risk Management & Portfolio Optimization                  ║
    ║      📊 Real-Time Performance Tracking & Analytics                          ║
    ║      🎮 DEMO MODE - Simulated Market Data                                   ║
    ║                                                                              ║
    ║      🚀 Powered by Machine Learning & Advanced Analytics                    ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Configuration: {MIN_ACCURACY_THRESHOLD}% accuracy threshold")
    print(f"⚡ Signal timing: {SIGNAL_ADVANCE_TIME}s advance warning")
    print(f"🎯 Expiry time: {EXPIRY_TIME}s (2 minutes)")
    print(f"🧠 Simplified LSTM Ensemble: Random Forest, Gradient Boosting, SVR")
    print(f"📊 Technical Analysis: RSI, MACD, Bollinger Bands, Moving Averages")
    print(f"📈 Sentiment Analysis: News sentiment and market mood")
    print(f"⚠️  Risk Management: Dynamic position sizing and drawdown protection")
    print(f"📊 Performance Tracking: Real-time accuracy and profit monitoring")
    print(f"🎮 Demo Mode: Simulated market data for testing")
    print("-" * 80)

def validate_environment():
    """Validate environment and dependencies"""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher is required")
            return False
        
        # Check required directories
        required_dirs = ['logs', 'models', 'backup']
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
        
        print("✅ Environment validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False

def initialize_demo_components():
    """Initialize all demo components"""
    try:
        logging.info("🔧 Initializing demo components...")
        
        # Initialize simplified signal generator
        signal_generator = SimplifiedSignalGenerator()
        
        logging.info("✅ Demo components initialized successfully")
        
        return signal_generator
        
    except Exception as e:
        logging.error(f"❌ Error initializing demo components: {e}")
        raise

async def run_demo_bot():
    """Run the demo trading bot"""
    try:
        # Setup logging
        setup_logging()
        
        # Print startup banner
        print_demo_banner()
        
        # Validate environment
        if not validate_environment():
            logging.error("Environment validation failed")
            return False
        
        # Initialize demo components
        signal_generator = initialize_demo_components()
        
        # Start simplified signal generation
        logging.info("🚀 Starting Demo Signal Generation Engine...")
        if not signal_generator.start_signal_generation():
            logging.error("Failed to start demo signal generation")
            return False
        
        print("\n🎮 DEMO BOT IS NOW RUNNING!")
        print("📡 Signal generation engine is active")
        print("📊 Analyzing currency pairs in real-time")
        print("🎯 Generating high-accuracy trading signals")
        print("⏰ Press Ctrl+C to stop the bot")
        print("-" * 80)
        
        # Keep the bot running and show status updates
        signal_count = 0
        while True:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            # Show status update
            signal_count += 1
            print(f"⏰ Status Update #{signal_count}: Bot is running...")
            print(f"📊 Available pairs: {len(signal_generator.pocket_api.get_available_pairs())}")
            print(f"📈 Signal history: {len(signal_generator.signal_history)} signals generated")
            
            if signal_generator.signal_history:
                latest_signal = signal_generator.signal_history[-1]
                print(f"🎯 Latest signal: {latest_signal['pair']} {latest_signal['direction']} "
                      f"(Confidence: {latest_signal['confidence']:.1f}%)")
            
            print("-" * 50)
            
    except KeyboardInterrupt:
        logging.info("⛔ Demo bot stopped by user")
        print("\n⛔ Demo bot stopped by user")
    except Exception as e:
        logging.error(f"💥 Fatal error in demo bot: {e}")
        print(f"💥 Fatal error in demo bot: {e}")
        raise
    finally:
        logging.info("🛑 Demo bot shutdown complete")
        print("🛑 Demo bot shutdown complete")

def main():
    """Main function"""
    try:
        # Run the demo bot
        asyncio.run(run_demo_bot())
        
    except KeyboardInterrupt:
        print("\n⛔ Demo bot stopped by user")
    except Exception as e:
        print(f"💥 Failed to start demo bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()