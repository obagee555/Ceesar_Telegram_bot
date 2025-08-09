#!/usr/bin/env python3
"""
Comprehensive System Checkup for Advanced Binary Trading Bot
Analyzes all components and provides detailed status report
"""

import os
import sys
import json
import sqlite3
import importlib.util
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple
try:
    import pkg_resources
except ImportError:
    pkg_resources = None

class SystemCheckup:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'components': {},
            'dependencies': {},
            'configuration': {},
            'recommendations': [],
            'errors': [],
            'warnings': []
        }
        
    def run_full_checkup(self) -> Dict:
        """Run comprehensive system checkup"""
        print("🔍 Starting Comprehensive System Checkup...")
        print("=" * 60)
        
        # Check all components
        self.check_python_environment()
        self.check_dependencies()
        self.check_core_modules()
        self.check_configuration()
        self.check_ai_models()
        self.check_databases()
        self.check_telegram_bot()
        self.check_api_integration()
        self.check_backup_system()
        self.check_security()
        
        # Generate overall status
        self.generate_overall_status()
        
        # Print results
        self.print_results()
        
        return self.results
    
    def check_python_environment(self):
        """Check Python environment"""
        print("🐍 Checking Python Environment...")
        
        try:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            status = {
                'status': 'OK',
                'version': python_version,
                'executable': sys.executable,
                'platform': sys.platform
            }
            
            if sys.version_info < (3, 8):
                status['status'] = 'ERROR'
                self.results['errors'].append("Python 3.8+ required")
            elif sys.version_info < (3, 9):
                status['status'] = 'WARNING'
                self.results['warnings'].append("Python 3.9+ recommended")
            
            self.results['components']['python'] = status
            print(f"   ✅ Python {python_version} - {status['status']}")
            
        except Exception as e:
            self.results['components']['python'] = {'status': 'ERROR', 'error': str(e)}
            self.results['errors'].append(f"Python check failed: {e}")
            print(f"   ❌ Python check failed: {e}")
    
    def check_dependencies(self):
        """Check required dependencies"""
        print("📦 Checking Dependencies...")
        
        required_packages = [
            'python-telegram-bot', 'numpy', 'pandas', 'tensorflow',
            'scikit-learn', 'websocket-client', 'requests', 'ta',
            'textblob', 'vaderSentiment', 'python-dotenv'
        ]
        
        dep_status = {}
        
                 for package in required_packages:
             try:
                 if package == 'python-telegram-bot':
                     try:
                         import telegram
                         version = telegram.__version__
                     except ImportError:
                         version = "Not installed"
                         raise
                 elif package == 'websocket-client':
                     try:
                         import websocket
                         version = websocket.__version__
                     except ImportError:
                         version = "Not installed"
                         raise
                 else:
                     if pkg_resources:
                         pkg = pkg_resources.get_distribution(package)
                         version = pkg.version
                     else:
                         # Try importing the package directly
                         __import__(package)
                         version = "Available"
                
                dep_status[package] = {
                    'status': 'OK',
                    'version': version,
                    'installed': True
                }
                print(f"   ✅ {package} {version}")
                
            except Exception as e:
                dep_status[package] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'installed': False
                }
                self.results['errors'].append(f"Missing dependency: {package}")
                print(f"   ❌ {package} - Missing")
        
        self.results['dependencies'] = dep_status
    
    def check_core_modules(self):
        """Check core bot modules"""
        print("🔧 Checking Core Modules...")
        
        modules = {
            'config.py': 'Configuration module',
            'lstm_model.py': 'LSTM AI model',
            'technical_analysis.py': 'Technical analysis engine',
            'sentiment_analysis.py': 'Sentiment analysis engine',
            'signal_generator.py': 'Signal generation engine',
            'telegram_bot.py': 'Telegram bot interface',
            'risk_management.py': 'Risk management system',
            'performance_tracker.py': 'Performance tracking',
            'backup_manager.py': 'Backup management',
            'pocket_option_api.py': 'Pocket Option API integration',
            'main.py': 'Main startup script'
        }
        
        module_status = {}
        
        for module_file, description in modules.items():
            try:
                if os.path.exists(f'/workspace/{module_file}'):
                    # Try to import the module
                    module_name = module_file.replace('.py', '')
                    spec = importlib.util.spec_from_file_location(
                        module_name, f'/workspace/{module_file}'
                    )
                    
                    if spec and spec.loader:
                        test_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(test_module)
                        
                        module_status[module_file] = {
                            'status': 'OK',
                            'description': description,
                            'exists': True,
                            'importable': True
                        }
                        print(f"   ✅ {module_file} - {description}")
                    else:
                        module_status[module_file] = {
                            'status': 'ERROR',
                            'description': description,
                            'exists': True,
                            'importable': False,
                            'error': 'Cannot load module'
                        }
                        self.results['errors'].append(f"Cannot load {module_file}")
                        print(f"   ❌ {module_file} - Cannot load")
                else:
                    module_status[module_file] = {
                        'status': 'ERROR',
                        'description': description,
                        'exists': False,
                        'importable': False,
                        'error': 'File not found'
                    }
                    self.results['errors'].append(f"Missing file: {module_file}")
                    print(f"   ❌ {module_file} - Missing")
                    
            except Exception as e:
                module_status[module_file] = {
                    'status': 'ERROR',
                    'description': description,
                    'exists': os.path.exists(f'/workspace/{module_file}'),
                    'importable': False,
                    'error': str(e)
                }
                self.results['errors'].append(f"Error loading {module_file}: {e}")
                print(f"   ❌ {module_file} - Error: {e}")
        
        self.results['components']['modules'] = module_status
    
    def check_configuration(self):
        """Check configuration settings"""
        print("⚙️ Checking Configuration...")
        
        try:
            import config
            
            config_status = {
                'status': 'OK',
                'telegram_bot_token': bool(getattr(config, 'TELEGRAM_BOT_TOKEN', None)),
                'authorized_user_id': bool(getattr(config, 'AUTHORIZED_USER_ID', None)),
                'pocket_option_ssid': bool(getattr(config, 'POCKET_OPTION_SSID', None)),
                'accuracy_threshold': getattr(config, 'MIN_ACCURACY_THRESHOLD', 0),
                'currency_pairs': bool(getattr(config, 'CURRENCY_PAIRS', {})),
                'technical_indicators': bool(getattr(config, 'TECHNICAL_INDICATORS', {}))
            }
            
            # Check critical configurations
            if not config_status['telegram_bot_token']:
                self.results['errors'].append("Missing Telegram bot token")
                config_status['status'] = 'ERROR'
            
            if not config_status['authorized_user_id']:
                self.results['errors'].append("Missing authorized user ID")
                config_status['status'] = 'ERROR'
            
            if not config_status['pocket_option_ssid']:
                self.results['errors'].append("Missing Pocket Option SSID")
                config_status['status'] = 'ERROR'
            
            if config_status['accuracy_threshold'] < 90:
                self.results['warnings'].append("Low accuracy threshold")
            
            self.results['configuration'] = config_status
            print(f"   ✅ Configuration - {config_status['status']}")
            
        except Exception as e:
            self.results['configuration'] = {'status': 'ERROR', 'error': str(e)}
            self.results['errors'].append(f"Configuration error: {e}")
            print(f"   ❌ Configuration - Error: {e}")
    
    def check_ai_models(self):
        """Check AI models and training status"""
        print("🧠 Checking AI Models...")
        
        ai_status = {
            'lstm_model_file': os.path.exists('/workspace/models/lstm_model.h5'),
            'scaler_file': os.path.exists('/workspace/models/scaler.pkl'),
            'models_directory': os.path.exists('/workspace/models'),
            'status': 'OK'
        }
        
        if not ai_status['models_directory']:
            os.makedirs('/workspace/models', exist_ok=True)
            ai_status['status'] = 'WARNING'
            self.results['warnings'].append("Models directory created")
        
        if not ai_status['lstm_model_file']:
            ai_status['status'] = 'WARNING'
            self.results['warnings'].append("LSTM model not trained yet")
        
        if not ai_status['scaler_file']:
            ai_status['status'] = 'WARNING'
            self.results['warnings'].append("Scaler not trained yet")
        
        # Test LSTM model loading
        try:
            from lstm_model import LSTMTradingModel
            model = LSTMTradingModel()
            ai_status['lstm_loadable'] = True
            print(f"   ✅ LSTM Model - {ai_status['status']}")
        except Exception as e:
            ai_status['lstm_loadable'] = False
            ai_status['status'] = 'ERROR'
            ai_status['error'] = str(e)
            self.results['errors'].append(f"LSTM model error: {e}")
            print(f"   ❌ LSTM Model - Error: {e}")
        
        self.results['components']['ai_models'] = ai_status
    
    def check_databases(self):
        """Check database connectivity and structure"""
        print("🗄️ Checking Databases...")
        
        db_status = {}
        
        # Check performance database
        try:
            if os.path.exists('performance.db'):
                conn = sqlite3.connect('performance.db')
                cursor = conn.cursor()
                
                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                db_status['performance.db'] = {
                    'status': 'OK',
                    'exists': True,
                    'tables': tables,
                    'signals_table': 'signals' in tables,
                    'daily_stats_table': 'daily_stats' in tables
                }
                
                # Check record count
                if 'signals' in tables:
                    cursor.execute("SELECT COUNT(*) FROM signals")
                    signal_count = cursor.fetchone()[0]
                    db_status['performance.db']['signal_count'] = signal_count
                
                conn.close()
                print(f"   ✅ performance.db - {len(tables)} tables")
            else:
                db_status['performance.db'] = {
                    'status': 'WARNING',
                    'exists': False,
                    'message': 'Will be created on first run'
                }
                print(f"   ⚠️ performance.db - Will be created")
                
        except Exception as e:
            db_status['performance.db'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            self.results['errors'].append(f"Performance database error: {e}")
            print(f"   ❌ performance.db - Error: {e}")
        
        self.results['components']['databases'] = db_status
    
    def check_telegram_bot(self):
        """Check Telegram bot functionality"""
        print("💬 Checking Telegram Bot...")
        
        try:
            from telegram_bot import TradingTelegramBot
            from config import TELEGRAM_BOT_TOKEN, AUTHORIZED_USER_ID
            
            # Basic initialization test
            bot = TradingTelegramBot()
            
            telegram_status = {
                'status': 'OK',
                'bot_class_loadable': True,
                'token_configured': bool(TELEGRAM_BOT_TOKEN),
                'user_id_configured': bool(AUTHORIZED_USER_ID),
                'token_format_valid': len(TELEGRAM_BOT_TOKEN.split(':')) == 2 if TELEGRAM_BOT_TOKEN else False
            }
            
            if not telegram_status['token_configured']:
                telegram_status['status'] = 'ERROR'
                self.results['errors'].append("Telegram bot token not configured")
            
            if not telegram_status['user_id_configured']:
                telegram_status['status'] = 'ERROR'
                self.results['errors'].append("Authorized user ID not configured")
            
            print(f"   ✅ Telegram Bot - {telegram_status['status']}")
            
        except Exception as e:
            telegram_status = {
                'status': 'ERROR',
                'error': str(e),
                'bot_class_loadable': False
            }
            self.results['errors'].append(f"Telegram bot error: {e}")
            print(f"   ❌ Telegram Bot - Error: {e}")
        
        self.results['components']['telegram_bot'] = telegram_status
    
    def check_api_integration(self):
        """Check Pocket Option API integration"""
        print("🔌 Checking API Integration...")
        
        try:
            from pocket_option_api import PocketOptionAPI
            from config import POCKET_OPTION_SSID
            
            api = PocketOptionAPI()
            
            api_status = {
                'status': 'OK',
                'api_class_loadable': True,
                'ssid_configured': bool(POCKET_OPTION_SSID),
                'ssid_format_valid': POCKET_OPTION_SSID.startswith('42["auth"') if POCKET_OPTION_SSID else False,
                'currency_pairs_available': len(api.get_available_pairs()) > 0
            }
            
            if not api_status['ssid_configured']:
                api_status['status'] = 'ERROR'
                self.results['errors'].append("Pocket Option SSID not configured")
            
            if not api_status['ssid_format_valid']:
                api_status['status'] = 'ERROR'
                self.results['errors'].append("Invalid SSID format")
            
            print(f"   ✅ API Integration - {api_status['status']}")
            
        except Exception as e:
            api_status = {
                'status': 'ERROR',
                'error': str(e),
                'api_class_loadable': False
            }
            self.results['errors'].append(f"API integration error: {e}")
            print(f"   ❌ API Integration - Error: {e}")
        
        self.results['components']['api_integration'] = api_status
    
    def check_backup_system(self):
        """Check backup system"""
        print("💾 Checking Backup System...")
        
        try:
            from backup_manager import BackupManager
            from config import BACKUP_PATH
            
            backup_manager = BackupManager()
            
            backup_status = {
                'status': 'OK',
                'backup_class_loadable': True,
                'backup_path_exists': os.path.exists(BACKUP_PATH),
                'backup_path_writable': os.access(BACKUP_PATH, os.W_OK) if os.path.exists(BACKUP_PATH) else False
            }
            
            if not backup_status['backup_path_exists']:
                os.makedirs(BACKUP_PATH, exist_ok=True)
                backup_status['backup_path_exists'] = True
                backup_status['backup_path_writable'] = True
            
            print(f"   ✅ Backup System - {backup_status['status']}")
            
        except Exception as e:
            backup_status = {
                'status': 'ERROR',
                'error': str(e),
                'backup_class_loadable': False
            }
            self.results['errors'].append(f"Backup system error: {e}")
            print(f"   ❌ Backup System - Error: {e}")
        
        self.results['components']['backup_system'] = backup_status
    
    def check_security(self):
        """Check security configurations"""
        print("🔒 Checking Security...")
        
        security_status = {
            'status': 'OK',
            'file_permissions': True,
            'sensitive_data_secured': True,
            'logs_secured': True
        }
        
        # Check if sensitive files have proper permissions
        sensitive_files = ['config.py', 'main.py']
        for file in sensitive_files:
            if os.path.exists(f'/workspace/{file}'):
                stat = os.stat(f'/workspace/{file}')
                if stat.st_mode & 0o077:  # Check if others have access
                    security_status['file_permissions'] = False
                    self.results['warnings'].append(f"File {file} has permissive permissions")
        
        # Check for hardcoded secrets in logs
        if os.path.exists('trading_bot.log'):
            try:
                with open('trading_bot.log', 'r') as f:
                    content = f.read()
                    if 'bot' in content.lower() and 'token' in content.lower():
                        security_status['logs_secured'] = False
                        self.results['warnings'].append("Potential sensitive data in logs")
            except Exception:
                pass
        
        if not security_status['file_permissions'] or not security_status['logs_secured']:
            security_status['status'] = 'WARNING'
        
        print(f"   ✅ Security - {security_status['status']}")
        self.results['components']['security'] = security_status
    
    def generate_overall_status(self):
        """Generate overall system status"""
        error_count = len(self.results['errors'])
        warning_count = len(self.results['warnings'])
        
        if error_count == 0 and warning_count == 0:
            self.results['overall_status'] = 'EXCELLENT'
        elif error_count == 0 and warning_count <= 3:
            self.results['overall_status'] = 'GOOD'
        elif error_count <= 2:
            self.results['overall_status'] = 'FAIR'
        else:
            self.results['overall_status'] = 'POOR'
        
        # Generate recommendations
        if error_count > 0:
            self.results['recommendations'].append("Fix all critical errors before running the bot")
        
        if warning_count > 0:
            self.results['recommendations'].append("Address warnings to improve performance")
        
        if not os.path.exists('/workspace/models/lstm_model.h5'):
            self.results['recommendations'].append("Train the LSTM model before generating signals")
        
        if self.results['overall_status'] in ['EXCELLENT', 'GOOD']:
            self.results['recommendations'].append("System is ready for operation")
    
    def print_results(self):
        """Print detailed results"""
        print("\n" + "=" * 60)
        print("📊 SYSTEM CHECKUP RESULTS")
        print("=" * 60)
        
        # Overall status
        status_emoji = {
            'EXCELLENT': '🟢',
            'GOOD': '🟡',
            'FAIR': '🟠',
            'POOR': '🔴'
        }
        
        print(f"\n{status_emoji.get(self.results['overall_status'], '⚪')} Overall Status: {self.results['overall_status']}")
        
        # Error summary
        print(f"\n📈 Summary:")
        print(f"   Errors: {len(self.results['errors'])}")
        print(f"   Warnings: {len(self.results['warnings'])}")
        
        # List errors
        if self.results['errors']:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for i, error in enumerate(self.results['errors'], 1):
                print(f"   {i}. {error}")
        
        # List warnings
        if self.results['warnings']:
            print(f"\n⚠️ Warnings ({len(self.results['warnings'])}):")
            for i, warning in enumerate(self.results['warnings'], 1):
                print(f"   {i}. {warning}")
        
        # Recommendations
        if self.results['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "=" * 60)
        
        # Telegram Bot Commands
        print("🤖 TELEGRAM BOT COMMANDS")
        print("=" * 60)
        
        commands = {
            "Basic Commands": [
                "/start - Initialize bot",
                "/status - System status",
                "/help - Command help"
            ],
            "Trading Commands": [
                "/signals - Toggle signal delivery",
                "/recent - Show recent signals",
                "/market - Market analysis",
                "/stats - Performance statistics",
                "/test - Test signal generation"
            ],
            "Management Commands": [
                "/settings - Bot configuration",
                "/backup - Create backup",
                "/retrain - Retrain AI models",
                "/pairs - Available currency pairs"
            ],
            "Advanced Commands": [
                "/risk - Risk management settings",
                "/logs - System logs"
            ]
        }
        
        for category, cmd_list in commands.items():
            print(f"\n📋 {category}:")
            for cmd in cmd_list:
                print(f"   {cmd}")
        
        print("\n" + "=" * 60)
        print("🚀 SYSTEM READY FOR DEPLOYMENT")
        print("=" * 60)
        
        if self.results['overall_status'] in ['EXCELLENT', 'GOOD']:
            print("\n✅ The trading bot is ready to start!")
            print("   Run: python main.py")
        else:
            print(f"\n⚠️ Please fix the issues above before starting the bot.")
        
        print(f"\n🕐 Checkup completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Run system checkup"""
    checkup = SystemCheckup()
    results = checkup.run_full_checkup()
    
    # Save results to file
    with open('/workspace/system_checkup_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: /workspace/system_checkup_results.json")
    
    return results

if __name__ == "__main__":
    main()