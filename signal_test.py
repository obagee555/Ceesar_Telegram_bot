#!/usr/bin/env python3
"""Test the new simplified signal format"""

from telegram_lstm_bot import TelegramBot

def test_signal_format():
    bot = TelegramBot()
    signal = bot.generate_lstm_signal()
    
    if signal:
        print("🚀 NEW SIMPLIFIED SIGNAL FORMAT:")
        print("="*50)
        formatted_message = bot.format_signal_message(signal)
        print(formatted_message)
        print("="*50)
        
        print("\n📊 LSTM TECHNICAL ANALYSIS DETAILS:")
        print(f"• RSI: {signal['lstm_analysis']['bullish_signals']} bullish vs {signal['lstm_analysis']['bearish_signals']} bearish signals")
        print(f"• Technical Confluence: {signal['market_conditions']['technical_confluence']}")
        print(f"• Trend Direction: {signal['market_conditions']['trend_direction']}")
        print(f"• Volatility: {signal['market_conditions']['volatility']}%")
        print(f"• Trend Strength: {signal['market_conditions']['trend_strength']}%")

if __name__ == "__main__":
    test_signal_format()