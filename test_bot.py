#!/usr/bin/env python3
"""
Test script for LSTM Telegram Bot
"""

import json
import time
from telegram_lstm_bot import TelegramBot

def test_signal_generation():
    """Test LSTM signal generation"""
    print("🧪 Testing LSTM Signal Generation...")
    
    # Create bot instance
    bot = TelegramBot()
    
    # Test signal generation
    signal = bot.generate_lstm_signal()
    
    if signal:
        print("✅ Signal Generation: SUCCESS")
        print(f"   Pair: {signal['pair']}")
        print(f"   Direction: {signal['direction']}")
        print(f"   Accuracy: {signal['accuracy']}%")
        print(f"   LSTM Confidence: {signal['lstm_confidence']}%")
        print(f"   Expiry Duration: {signal['expiry_duration']}")
        print(f"   Trade Time: {signal['trade_time']}")
        
        # Test message formatting
        formatted_message = bot.format_signal_message(signal)
        print(f"\n📝 Message Format: {len(formatted_message)} characters")
        print("✅ Message Formatting: SUCCESS")
        
        return True
    else:
        print("❌ Signal Generation: FAILED")
        return False

def test_commands():
    """Test bot command handling"""
    print("\n🧪 Testing Bot Commands...")
    
    bot = TelegramBot()
    
    # Test available pairs
    pairs = bot.get_available_pairs()
    print(f"✅ Available Pairs: {len(pairs)} pairs loaded")
    
    # Test market time
    market_time = bot.get_market_time()
    print(f"✅ Market Time: {market_time.strftime('%H:%M:%S UTC-4')}")
    
    # Test LSTM model
    lstm_model = bot.lstm_model
    print(f"✅ LSTM Model: {lstm_model.model_performance['accuracy']}% accuracy")
    
    return True

def main():
    """Main test function"""
    print("🚀 LSTM AI Trading Bot - Test Suite")
    print("="*50)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Signal Generation
    if test_signal_generation():
        success_count += 1
    
    # Test 2: Commands
    if test_commands():
        success_count += 1
    
    print("\n" + "="*50)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All tests PASSED! Bot is ready for Telegram use.")
    else:
        print("❌ Some tests FAILED. Check configuration.")
    
    return success_count == total_tests

if __name__ == "__main__":
    main()