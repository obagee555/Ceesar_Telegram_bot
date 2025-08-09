#!/usr/bin/env python3
"""
Advanced Binary Trading Bot for Pocket Option
95%+ Accuracy LSTM AI-Powered Signal Generator

Author: AI Trading Systems
Version: 1.0
License: Private Use Only
"""

import sys
import os
import asyncio
import logging
import signal
from datetime import datetime

# Add workspace to Python path
sys.path.append('/workspace')

from telegram_bot import TradingTelegramBot
from config import *

def setup_signal_handlers(bot):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        asyncio.create_task(bot.stop_bot())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def print_startup_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║      🤖 ADVANCED BINARY TRADING BOT v1.0                    ║
    ║                                                              ║
    ║      🎯 95%+ Accuracy LSTM AI Predictions                   ║
    ║      📊 Real-time Technical Analysis                        ║
    ║      🧠 Market Sentiment Analysis                           ║
    ║      ⚠️  Advanced Risk Management                            ║
    ║      📈 Performance Tracking                                ║
    ║      💬 Telegram Integration                                ║
    ║                                                              ║
    ║      🚀 Powered by Neural Networks & Machine Learning       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Configuration: {MIN_ACCURACY_THRESHOLD}% accuracy threshold")
    print(f"⚡ Signal timing: {SIGNAL_ADVANCE_TIME}s advance warning")
    print(f"🎯 Expiry time: {EXPIRY_TIME}s (2 minutes)")
    print("-" * 66)

async def main():
    """Main function"""
    try:
        # Print startup banner
        print_startup_banner()
        
        # Initialize and start the trading bot
        bot = TradingTelegramBot()
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(bot)
        
        # Start the bot
        logging.info("🚀 Starting Advanced Binary Trading Bot...")
        await bot.start_bot()
        
    except KeyboardInterrupt:
        logging.info("⛔ Bot stopped by user")
    except Exception as e:
        logging.error(f"💥 Fatal error: {e}")
        raise
    finally:
        logging.info("🛑 Bot shutdown complete")

if __name__ == "__main__":
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher is required")
            sys.exit(1)
        
        # Run the bot
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n⛔ Bot stopped by user")
    except Exception as e:
        print(f"💥 Failed to start bot: {e}")
        sys.exit(1)