import asyncio
import logging
from datetime import datetime, timedelta
import time
import threading
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, filters, ContextTypes
)
from signal_generator import SignalGenerator
from risk_management import RiskManager
from performance_tracker import PerformanceTracker
from backup_manager import BackupManager
from config import *

class TradingTelegramBot:
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.authorized_user = AUTHORIZED_USER_ID
        self.application = None
        
        # Initialize components
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.performance_tracker = PerformanceTracker()
        self.backup_manager = BackupManager()
        
        # Bot state
        self.is_running = False
        self.signals_enabled = True
        self.auto_mode = True
        
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=getattr(logging, LOG_LEVEL),
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if not self.is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access!")
            return
        
        welcome_message = """
🤖 **Advanced Binary Trading Bot**

Welcome to your AI-powered trading assistant!

🎯 **Features:**
• 95%+ accuracy LSTM AI predictions
• Real-time technical analysis
• Market sentiment analysis
• Risk management system
• Performance tracking

📊 **Commands:**
/status - Bot status and statistics
/signals - Toggle signal delivery
/stats - Performance statistics
/recent - Recent signals
/market - Market analysis
/settings - Bot settings
/help - Command help

🚀 **Ready to start trading!**
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        try:
            # Get system status
            signal_stats = self.signal_generator.get_signal_statistics()
            risk_status = self.risk_manager.get_risk_status()
            
            status_message = f"""
🤖 **Bot Status Report**

🔄 **System Status:**
• Bot Running: {'✅ Active' if self.is_running else '❌ Inactive'}
• Signals: {'✅ Enabled' if self.signals_enabled else '❌ Disabled'}
• Auto Mode: {'✅ On' if self.auto_mode else '❌ Off'}
• API Connection: {'✅ Connected' if self.signal_generator.pocket_api.is_connected else '❌ Disconnected'}

📊 **Signal Statistics:**
• Total Signals: {signal_stats['total_signals']}
• Daily Signals: {signal_stats['daily_signals']}
• Average Accuracy: {signal_stats['average_accuracy']}%
• Win Rate: {signal_stats['win_rate']}%
• Active Pairs: {signal_stats['active_pairs']}

⚠️ **Risk Management:**
• Risk Level: {risk_status['risk_level']}
• Daily Limit: {risk_status['daily_trades_left']} trades left
• Max Drawdown: {risk_status['max_drawdown']}%

🕐 **Last Updated:** {datetime.now().strftime('%H:%M:%S')}
            """
            
            await update.message.reply_text(status_message, parse_mode='Markdown')
            
        except Exception as e:
            logging.error(f"Error in status command: {e}")
            await update.message.reply_text("❌ Error retrieving status")
    
    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command - toggle signal delivery"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        self.signals_enabled = not self.signals_enabled
        status = "enabled" if self.signals_enabled else "disabled"
        emoji = "✅" if self.signals_enabled else "❌"
        
        await update.message.reply_text(f"{emoji} Signal delivery {status}")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command - show performance statistics"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        try:
            stats = self.performance_tracker.get_comprehensive_stats()
            
            stats_message = f"""
📈 **Performance Statistics**

🎯 **Accuracy Metrics:**
• Overall Accuracy: {stats['overall_accuracy']}%
• Win Rate: {stats['win_rate']}%
• Loss Rate: {stats['loss_rate']}%
• Success Streak: {stats['current_streak']}

💰 **Trading Performance:**
• Total Trades: {stats['total_trades']}
• Successful Trades: {stats['winning_trades']}
• Failed Trades: {stats['losing_trades']}
• Profit Factor: {stats['profit_factor']:.2f}

📊 **Daily Performance:**
• Today's Signals: {stats['today_signals']}
• Today's Win Rate: {stats['today_win_rate']}%
• Weekly Win Rate: {stats['weekly_win_rate']}%
• Monthly Win Rate: {stats['monthly_win_rate']}%

🏆 **Best Performance:**
• Best Day: {stats['best_day_win_rate']}%
• Longest Streak: {stats['longest_streak']} wins
• Average Signal Strength: {stats['avg_signal_strength']:.1f}/10

📅 **Period:** {stats['tracking_period']} days
            """
            
            await update.message.reply_text(stats_message, parse_mode='Markdown')
            
        except Exception as e:
            logging.error(f"Error in stats command: {e}")
            await update.message.reply_text("❌ Error retrieving statistics")
    
    async def recent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /recent command - show recent signals"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        try:
            recent_signals = self.signal_generator.get_recent_signals(5)
            
            if not recent_signals:
                await update.message.reply_text("📭 No recent signals available")
                return
            
            message = "📋 **Recent Signals**\n\n"
            
            for i, signal in enumerate(reversed(recent_signals), 1):
                time_str = signal['generated_at'].strftime('%H:%M')
                message += f"""
**{i}. {signal['pair']}** - {time_str}
Direction: {signal['direction']}
Accuracy: {signal['accuracy']}%
AI Confidence: {signal['ai_confidence']}%
Expiry: {signal['expiry_time']}
---
                """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logging.error(f"Error in recent command: {e}")
            await update.message.reply_text("❌ Error retrieving recent signals")
    
    async def market_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /market command - show market analysis"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        try:
            # Get market overview
            available_pairs = self.signal_generator.pocket_api.get_available_pairs()
            
            message = "📊 **Market Analysis**\n\n"
            
            # Show analysis for top 3 pairs
            for pair in available_pairs[:3]:
                try:
                    market_data = self.signal_generator.pocket_api.get_market_data_df(pair, limit=50)
                    if market_data is not None:
                        analyzed_data = self.signal_generator.technical_analyzer.calculate_all_indicators(market_data)
                        if analyzed_data is not None:
                            current_price = analyzed_data['close'].iloc[-1]
                            volatility = self.signal_generator.pocket_api.get_volatility(pair)
                            strength = self.signal_generator.technical_analyzer.get_signal_strength(analyzed_data)
                            
                            message += f"""
**{pair}**
Price: {current_price:.5f}
Volatility: {volatility:.4f}
Signal Strength: {strength:.1f}/10
Trend: {'🔹' if strength > 6 else '🔸' if strength > 4 else '🔹'}
---
                            """
                except Exception:
                    continue
            
            # Add market sentiment
            sentiment = self.signal_generator.sentiment_analyzer.calculate_market_fear_greed_index()
            message += f"""
🌡️ **Market Sentiment**
Fear & Greed Index: {sentiment['index']:.0f}/100
Status: {sentiment['label']}

🕐 **Updated:** {datetime.now().strftime('%H:%M:%S')}
            """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logging.error(f"Error in market command: {e}")
            await update.message.reply_text("❌ Error retrieving market analysis")
    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command - show bot settings"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        keyboard = [
            [InlineKeyboardButton("🔄 Toggle Auto Mode", callback_data="toggle_auto")],
            [InlineKeyboardButton("📊 Toggle Signals", callback_data="toggle_signals")],
            [InlineKeyboardButton("🎯 Set Accuracy Threshold", callback_data="set_accuracy")],
            [InlineKeyboardButton("⚠️ Risk Settings", callback_data="risk_settings")],
            [InlineKeyboardButton("🔄 Restart Bot", callback_data="restart_bot")],
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        settings_message = f"""
⚙️ **Bot Settings**

Current Configuration:
• Auto Mode: {'✅ On' if self.auto_mode else '❌ Off'}
• Signals: {'✅ Enabled' if self.signals_enabled else '❌ Disabled'}
• Accuracy Threshold: {MIN_ACCURACY_THRESHOLD}%
• Signal Interval: {self.signal_generator.min_signal_interval // 60} minutes
• Risk Level: {self.risk_manager.risk_level}

Select an option to modify:
        """
        
        await update.message.reply_text(
            settings_message, 
            reply_markup=reply_markup, 
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        help_message = """
📚 **Command Help**

🤖 **Basic Commands:**
/start - Initialize bot
/status - System status
/help - This help message

📊 **Trading Commands:**
/signals - Toggle signal delivery
/recent - Show recent signals
/market - Market analysis
/stats - Performance statistics

⚙️ **Management Commands:**
/settings - Bot configuration
/backup - Create backup
/restore - Restore from backup
/retrain - Retrain AI models

🔧 **Advanced Commands:**
/risk - Risk management settings
/pairs - Available currency pairs
/test - Test signal generation
/logs - System logs

📈 **Signal Format:**
Each signal includes:
• Currency pair (GBP/USD OTC)
• Direction (BUY/SELL)
• Accuracy percentage
• Expiry time (real market time)
• AI confidence level

🎯 **Features:**
• 95%+ accuracy AI predictions
• LSTM neural networks
• Technical analysis (RSI, MACD, BB, etc.)
• Market sentiment analysis
• Real-time data from Pocket Option
• Risk management system
• Performance tracking

💡 **Tips:**
• Signals are generated only during low volatility
• Wait for high accuracy signals (95%+)
• Follow risk management guidelines
• Monitor performance statistics

🆘 **Support:**
Contact for technical support or issues.
        """
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def backup_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /backup command"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        try:
            backup_file = self.backup_manager.create_backup()
            if backup_file:
                await update.message.reply_text(f"✅ Backup created: {backup_file}")
            else:
                await update.message.reply_text("❌ Backup creation failed")
        except Exception as e:
            logging.error(f"Error in backup command: {e}")
            await update.message.reply_text("❌ Error creating backup")
    
    async def retrain_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /retrain command"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        await update.message.reply_text("🔄 Starting model retraining...")
        
        try:
            success = self.signal_generator.train_models_with_new_data()
            if success:
                await update.message.reply_text("✅ Model retraining completed successfully")
            else:
                await update.message.reply_text("❌ Model retraining failed")
        except Exception as e:
            logging.error(f"Error in retrain command: {e}")
            await update.message.reply_text("❌ Error during model retraining")
    
    async def pairs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pairs command - show available currency pairs"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        try:
            pairs = self.signal_generator.pocket_api.get_available_pairs()
            
            message = "💱 **Available Currency Pairs**\n\n"
            
            current_day = datetime.now().weekday()
            is_weekend = current_day >= 5
            
            if is_weekend:
                message += "📅 **Weekend Pairs:**\n"
            else:
                message += "📅 **Weekday OTC Pairs:**\n"
            
            for i, pair in enumerate(pairs, 1):
                message += f"{i}. {pair}\n"
            
            message += f"\n🕐 Total: {len(pairs)} pairs available"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logging.error(f"Error in pairs command: {e}")
            await update.message.reply_text("❌ Error retrieving currency pairs")
    
    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /test command - test signal generation"""
        if not self.is_authorized(update.effective_user.id):
            return
        
        await update.message.reply_text("🧪 Testing signal generation...")
        
        try:
            # Test signal generation for GBP/USD OTC
            test_pair = "GBP/USD OTC"
            signal = self.signal_generator.generate_signal(test_pair)
            
            if signal:
                message = "✅ **Test Signal Generated**\n\n"
                message += self.signal_generator.format_signal_message(signal)
                await update.message.reply_text(message, parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ No signal generated (conditions not met)")
                
        except Exception as e:
            logging.error(f"Error in test command: {e}")
            await update.message.reply_text("❌ Error testing signal generation")
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if not self.is_authorized(query.from_user.id):
            return
        
        data = query.data
        
        if data == "toggle_auto":
            self.auto_mode = not self.auto_mode
            status = "enabled" if self.auto_mode else "disabled"
            await query.edit_message_text(f"🔄 Auto mode {status}")
            
        elif data == "toggle_signals":
            self.signals_enabled = not self.signals_enabled
            status = "enabled" if self.signals_enabled else "disabled"
            await query.edit_message_text(f"📊 Signals {status}")
            
        elif data == "restart_bot":
            await query.edit_message_text("🔄 Restarting bot...")
            # Implement restart logic
            
    def is_authorized(self, user_id):
        """Check if user is authorized"""
        return user_id == self.authorized_user
    
    async def send_signal_to_user(self, signal):
        """Send trading signal to authorized user"""
        try:
            if not self.signals_enabled:
                return
            
            message = self.signal_generator.format_signal_message(signal)
            
            # Add timing information
            current_time = datetime.now()
            signal_delay = (signal['expiry_timestamp'] - current_time).total_seconds()
            
            if signal_delay > SIGNAL_ADVANCE_TIME:
                # Schedule signal to be sent 1 minute before expiry
                delay = signal_delay - SIGNAL_ADVANCE_TIME
                await asyncio.sleep(delay)
            
            # Send the signal
            await self.application.bot.send_message(
                chat_id=self.authorized_user,
                text=message,
                parse_mode='Markdown'
            )
            
            logging.info(f"Signal sent to user: {signal['pair']} {signal['direction']}")
            
        except Exception as e:
            logging.error(f"Error sending signal to user: {e}")
    
    def setup_signal_callback(self):
        """Setup callback for signal broadcasting"""
        original_broadcast = self.signal_generator.broadcast_signal
        
        def new_broadcast(signal):
            # Call original broadcast
            original_broadcast(signal)
            
            # Send to Telegram user
            if self.application and self.signals_enabled:
                asyncio.create_task(self.send_signal_to_user(signal))
        
        self.signal_generator.broadcast_signal = new_broadcast
    
    async def start_bot(self):
        """Start the Telegram bot"""
        try:
            # Create application
            self.application = Application.builder().token(self.bot_token).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("signals", self.signals_command))
            self.application.add_handler(CommandHandler("stats", self.stats_command))
            self.application.add_handler(CommandHandler("recent", self.recent_command))
            self.application.add_handler(CommandHandler("market", self.market_command))
            self.application.add_handler(CommandHandler("settings", self.settings_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("backup", self.backup_command))
            self.application.add_handler(CommandHandler("retrain", self.retrain_command))
            self.application.add_handler(CommandHandler("pairs", self.pairs_command))
            self.application.add_handler(CommandHandler("test", self.test_command))
            
            # Add callback query handler
            self.application.add_handler(CallbackQueryHandler(self.button_callback))
            
            # Setup signal callback
            self.setup_signal_callback()
            
            # Start signal generation
            if self.signal_generator.start_signal_generation():
                logging.info("Signal generation started")
            else:
                logging.error("Failed to start signal generation")
            
            # Start other components
            self.performance_tracker.start_tracking()
            self.backup_manager.start_auto_backup()
            
            self.is_running = True
            
            # Start the bot
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            logging.info("Telegram bot started successfully")
            
            # Send startup notification
            await self.application.bot.send_message(
                chat_id=self.authorized_user,
                text="🤖 **Trading Bot Started**\n\nAll systems operational!\nReady to generate trading signals.",
                parse_mode='Markdown'
            )
            
            # Keep the bot running
            await self.application.updater.idle()
            
        except Exception as e:
            logging.error(f"Error starting Telegram bot: {e}")
            raise
    
    async def stop_bot(self):
        """Stop the Telegram bot"""
        try:
            self.is_running = False
            
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
            
            # Stop components
            self.signal_generator.stop_signal_generation()
            self.performance_tracker.stop_tracking()
            self.backup_manager.stop_auto_backup()
            
            logging.info("Telegram bot stopped")
            
        except Exception as e:
            logging.error(f"Error stopping Telegram bot: {e}")

async def main():
    """Main function to run the bot"""
    bot = TradingTelegramBot()
    
    try:
        await bot.start_bot()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Bot error: {e}")
    finally:
        await bot.stop_bot()

if __name__ == "__main__":
    asyncio.run(main())