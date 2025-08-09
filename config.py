import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "8226952507:AAGPhIvSNikHOkDFTUAZnjTKQzxR4m9yIAU"
AUTHORIZED_USER_ID = 8093708320

# Pocket Option Configuration
POCKET_OPTION_SSID = '42["auth",{"session":"a:4:{s:10:\\"session_id\\";s:32:\\"8ddc70c84462c00f33c4e55cd07348c2\\";s:10:\\"ip_address\\";s:14:\\"102.88.110.242\\";s:10:\\"user_agent\\";s:120:\\"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.\\";s:13:\\"last_activity\\";i:1750856786;}5273f506ca5eac602df49436664bca19","isDemo":0,"uid":74793694,"platform":2,"isFastHistory":true}]'

# Trading Configuration
MIN_ACCURACY_THRESHOLD = 95.0
SIGNAL_ADVANCE_TIME = 60  # seconds before trade
EXPIRY_TIME = 120  # 2 minutes in seconds
VOLATILITY_THRESHOLD = 0.02  # Low volatility threshold

# LSTM Model Configuration
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_UNITS = 50

# Technical Analysis Configuration
TECHNICAL_INDICATORS = {
    'RSI': {'period': 14},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BB': {'period': 20, 'std': 2},
    'SMA': {'periods': [5, 10, 20, 50]},
    'EMA': {'periods': [12, 26]},
    'STOCH': {'k_period': 14, 'd_period': 3},
    'ATR': {'period': 14},
    'ADX': {'period': 14}
}

# Currency Pairs Configuration
CURRENCY_PAIRS = {
    'weekdays': [
        'GBP/USD OTC', 'EUR/USD OTC', 'USD/JPY OTC', 'AUD/USD OTC',
        'USD/CAD OTC', 'USD/CHF OTC', 'NZD/USD OTC', 'EUR/GBP OTC',
        'EUR/JPY OTC', 'GBP/JPY OTC', 'AUD/JPY OTC', 'EUR/CHF OTC'
    ],
    'weekends': [
        'GBP/USD', 'EUR/USD', 'USD/JPY', 'AUD/USD',
        'USD/CAD', 'USD/CHF', 'NZD/USD', 'EUR/GBP',
        'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'EUR/CHF'
    ]
}

# Database Configuration
DATABASE_PATH = 'trading_bot.db'

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_bot.log'

# Backup Configuration
BACKUP_PATH = '/workspace/backup/'
BACKUP_FREQUENCY = 3600  # 1 hour in seconds