import os
import shutil
import json
import sqlite3
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
import zipfile
from config import BACKUP_PATH, BACKUP_FREQUENCY

class BackupManager:
    def __init__(self):
        self.backup_path = BACKUP_PATH
        self.backup_frequency = BACKUP_FREQUENCY
        self.is_auto_backup_running = False
        self.backup_thread = None
        
        # Ensure backup directory exists
        os.makedirs(self.backup_path, exist_ok=True)
        
    def start_auto_backup(self):
        """Start automatic backup process"""
        try:
            self.is_auto_backup_running = True
            self.backup_thread = threading.Thread(target=self.auto_backup_loop)
            self.backup_thread.daemon = True
            self.backup_thread.start()
            logging.info("Auto backup started")
            
        except Exception as e:
            logging.error(f"Error starting auto backup: {e}")
    
    def stop_auto_backup(self):
        """Stop automatic backup process"""
        try:
            self.is_auto_backup_running = False
            if self.backup_thread:
                self.backup_thread.join(timeout=5)
            logging.info("Auto backup stopped")
            
        except Exception as e:
            logging.error(f"Error stopping auto backup: {e}")
    
    def auto_backup_loop(self):
        """Auto backup loop"""
        while self.is_auto_backup_running:
            try:
                self.create_backup()
                time.sleep(self.backup_frequency)
                
            except Exception as e:
                logging.error(f"Error in auto backup loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def create_backup(self) -> Optional[str]:
        """Create a complete backup of the trading bot"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"trading_bot_backup_{timestamp}.zip"
            backup_filepath = os.path.join(self.backup_path, backup_filename)
            
            with zipfile.ZipFile(backup_filepath, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
                # Backup configuration
                if os.path.exists('config.py'):
                    backup_zip.write('config.py', 'config.py')
                
                # Backup LSTM model
                if os.path.exists('/workspace/models/lstm_model.h5'):
                    backup_zip.write('/workspace/models/lstm_model.h5', 'models/lstm_model.h5')
                
                if os.path.exists('/workspace/models/scaler.pkl'):
                    backup_zip.write('/workspace/models/scaler.pkl', 'models/scaler.pkl')
                
                # Backup databases
                if os.path.exists('performance.db'):
                    backup_zip.write('performance.db', 'performance.db')
                
                if os.path.exists('trading_bot.db'):
                    backup_zip.write('trading_bot.db', 'trading_bot.db')
                
                # Backup logs
                if os.path.exists('trading_bot.log'):
                    backup_zip.write('trading_bot.log', 'trading_bot.log')
                
                # Create backup manifest
                manifest = self.create_backup_manifest()
                backup_zip.writestr('backup_manifest.json', json.dumps(manifest, indent=2))
            
            # Clean old backups
            self.cleanup_old_backups()
            
            logging.info(f"Backup created: {backup_filename}")
            return backup_filename
            
        except Exception as e:
            logging.error(f"Error creating backup: {e}")
            return None
    
    def create_backup_manifest(self) -> Dict:
        """Create backup manifest with metadata"""
        try:
            manifest = {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'components': {
                    'lstm_model': os.path.exists('/workspace/models/lstm_model.h5'),
                    'scaler': os.path.exists('/workspace/models/scaler.pkl'),
                    'performance_db': os.path.exists('performance.db'),
                    'trading_db': os.path.exists('trading_bot.db'),
                    'config': os.path.exists('config.py'),
                    'logs': os.path.exists('trading_bot.log')
                },
                'statistics': self.get_backup_statistics()
            }
            
            return manifest
            
        except Exception as e:
            logging.error(f"Error creating backup manifest: {e}")
            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}
    
    def get_backup_statistics(self) -> Dict:
        """Get statistics for backup manifest"""
        try:
            stats = {
                'total_signals': 0,
                'model_trained': False,
                'config_valid': False
            }
            
            # Get signal count from performance database
            if os.path.exists('performance.db'):
                try:
                    conn = sqlite3.connect('performance.db')
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM signals')
                    result = cursor.fetchone()
                    stats['total_signals'] = result[0] if result else 0
                    conn.close()
                except Exception:
                    pass
            
            # Check if LSTM model exists
            stats['model_trained'] = os.path.exists('/workspace/models/lstm_model.h5')
            
            # Check if config is valid
            stats['config_valid'] = os.path.exists('config.py')
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting backup statistics: {e}")
            return {}
    
    def restore_backup(self, backup_filename: str) -> bool:
        """Restore from a backup file"""
        try:
            backup_filepath = os.path.join(self.backup_path, backup_filename)
            
            if not os.path.exists(backup_filepath):
                logging.error(f"Backup file not found: {backup_filename}")
                return False
            
            with zipfile.ZipFile(backup_filepath, 'r') as backup_zip:
                # Read manifest
                try:
                    manifest_data = backup_zip.read('backup_manifest.json')
                    manifest = json.loads(manifest_data)
                    logging.info(f"Restoring backup from: {manifest['timestamp']}")
                except Exception:
                    logging.warning("No manifest found in backup")
                
                # Restore files
                backup_zip.extractall('/workspace/')
            
            logging.info(f"Backup restored: {backup_filename}")
            return True
            
        except Exception as e:
            logging.error(f"Error restoring backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List available backups"""
        try:
            backups = []
            
            for filename in os.listdir(self.backup_path):
                if filename.endswith('.zip') and filename.startswith('trading_bot_backup_'):
                    filepath = os.path.join(self.backup_path, filename)
                    
                    # Get file stats
                    stat = os.stat(filepath)
                    size_mb = stat.st_size / (1024 * 1024)
                    created = datetime.fromtimestamp(stat.st_ctime)
                    
                    # Try to read manifest
                    manifest_info = {}
                    try:
                        with zipfile.ZipFile(filepath, 'r') as backup_zip:
                            manifest_data = backup_zip.read('backup_manifest.json')
                            manifest = json.loads(manifest_data)
                            manifest_info = manifest.get('statistics', {})
                    except Exception:
                        pass
                    
                    backup_info = {
                        'filename': filename,
                        'created': created.isoformat(),
                        'size_mb': round(size_mb, 2),
                        'signals_count': manifest_info.get('total_signals', 0),
                        'has_model': manifest_info.get('model_trained', False),
                        'valid': True
                    }
                    
                    backups.append(backup_info)
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created'], reverse=True)
            
            return backups
            
        except Exception as e:
            logging.error(f"Error listing backups: {e}")
            return []
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """Remove old backup files, keeping only the most recent ones"""
        try:
            backups = self.list_backups()
            
            if len(backups) > keep_count:
                backups_to_delete = backups[keep_count:]
                
                for backup in backups_to_delete:
                    filepath = os.path.join(self.backup_path, backup['filename'])
                    try:
                        os.remove(filepath)
                        logging.debug(f"Deleted old backup: {backup['filename']}")
                    except Exception as e:
                        logging.warning(f"Could not delete backup {backup['filename']}: {e}")
                
                logging.info(f"Cleaned up {len(backups_to_delete)} old backups")
            
        except Exception as e:
            logging.error(f"Error cleaning up old backups: {e}")
    
    def export_trading_data(self) -> Optional[str]:
        """Export trading data to JSON file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_filename = f"trading_data_export_{timestamp}.json"
            export_filepath = os.path.join(self.backup_path, export_filename)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'signals': [],
                'daily_stats': [],
                'configuration': {}
            }
            
            # Export signals from performance database
            if os.path.exists('performance.db'):
                try:
                    conn = sqlite3.connect('performance.db')
                    cursor = conn.cursor()
                    
                    # Export signals
                    cursor.execute('SELECT * FROM signals')
                    columns = [description[0] for description in cursor.description]
                    signals = cursor.fetchall()
                    
                    for signal in signals:
                        signal_dict = dict(zip(columns, signal))
                        export_data['signals'].append(signal_dict)
                    
                    # Export daily stats
                    cursor.execute('SELECT * FROM daily_stats')
                    columns = [description[0] for description in cursor.description]
                    daily_stats = cursor.fetchall()
                    
                    for stat in daily_stats:
                        stat_dict = dict(zip(columns, stat))
                        export_data['daily_stats'].append(stat_dict)
                    
                    conn.close()
                    
                except Exception as e:
                    logging.warning(f"Could not export database data: {e}")
            
            # Export configuration (sanitized)
            try:
                from config import MIN_ACCURACY_THRESHOLD, EXPIRY_TIME, VOLATILITY_THRESHOLD
                export_data['configuration'] = {
                    'min_accuracy_threshold': MIN_ACCURACY_THRESHOLD,
                    'expiry_time': EXPIRY_TIME,
                    'volatility_threshold': VOLATILITY_THRESHOLD,
                    # Don't export sensitive credentials
                }
            except Exception:
                pass
            
            # Write export file
            with open(export_filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logging.info(f"Trading data exported: {export_filename}")
            return export_filename
            
        except Exception as e:
            logging.error(f"Error exporting trading data: {e}")
            return None
    
    def get_backup_status(self) -> Dict:
        """Get backup system status"""
        try:
            backups = self.list_backups()
            latest_backup = backups[0] if backups else None
            
            status = {
                'auto_backup_running': self.is_auto_backup_running,
                'backup_frequency_hours': self.backup_frequency / 3600,
                'backup_path': self.backup_path,
                'total_backups': len(backups),
                'latest_backup': latest_backup,
                'backup_space_used_mb': self.calculate_backup_space_used(),
                'next_backup_in_seconds': self.get_next_backup_time() if self.is_auto_backup_running else None
            }
            
            return status
            
        except Exception as e:
            logging.error(f"Error getting backup status: {e}")
            return {
                'auto_backup_running': False,
                'error': str(e)
            }
    
    def calculate_backup_space_used(self) -> float:
        """Calculate total space used by backups in MB"""
        try:
            total_size = 0
            
            for filename in os.listdir(self.backup_path):
                if filename.endswith('.zip'):
                    filepath = os.path.join(self.backup_path, filename)
                    total_size += os.path.getsize(filepath)
            
            return round(total_size / (1024 * 1024), 2)
            
        except Exception as e:
            logging.error(f"Error calculating backup space: {e}")
            return 0
    
    def get_next_backup_time(self) -> Optional[int]:
        """Get seconds until next automatic backup"""
        try:
            # This is a simplified calculation
            # In a real implementation, you'd track the last backup time
            return self.backup_frequency
            
        except Exception as e:
            logging.error(f"Error calculating next backup time: {e}")
            return None
    
    def create_emergency_backup(self) -> Optional[str]:
        """Create an emergency backup (minimal, essential files only)"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"emergency_backup_{timestamp}.zip"
            backup_filepath = os.path.join(self.backup_path, backup_filename)
            
            with zipfile.ZipFile(backup_filepath, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
                # Backup only essential files
                essential_files = [
                    'performance.db',
                    '/workspace/models/lstm_model.h5',
                    '/workspace/models/scaler.pkl'
                ]
                
                for file_path in essential_files:
                    if os.path.exists(file_path):
                        arcname = os.path.basename(file_path)
                        backup_zip.write(file_path, arcname)
                
                # Add emergency manifest
                emergency_manifest = {
                    'type': 'emergency_backup',
                    'timestamp': datetime.now().isoformat(),
                    'files_backed_up': [f for f in essential_files if os.path.exists(f)]
                }
                
                backup_zip.writestr('emergency_manifest.json', 
                                  json.dumps(emergency_manifest, indent=2))
            
            logging.info(f"Emergency backup created: {backup_filename}")
            return backup_filename
            
        except Exception as e:
            logging.error(f"Error creating emergency backup: {e}")
            return None