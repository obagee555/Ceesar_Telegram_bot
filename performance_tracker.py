import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import threading
import time

class PerformanceTracker:
    def __init__(self):
        self.db_path = 'performance.db'
        self.is_tracking = False
        self.tracking_thread = None
        self.setup_database()
        
    def setup_database(self):
        """Setup SQLite database for performance tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    pair TEXT,
                    direction TEXT,
                    accuracy REAL,
                    ai_confidence REAL,
                    result TEXT,
                    profit_loss REAL,
                    signal_strength REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date DATE PRIMARY KEY,
                    total_signals INTEGER,
                    winning_signals INTEGER,
                    losing_signals INTEGER,
                    win_rate REAL,
                    total_profit_loss REAL,
                    average_accuracy REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Performance tracking database initialized")
            
        except Exception as e:
            logging.error(f"Error setting up database: {e}")
    
    def start_tracking(self):
        """Start performance tracking"""
        try:
            self.is_tracking = True
            self.tracking_thread = threading.Thread(target=self.tracking_loop)
            self.tracking_thread.daemon = True
            self.tracking_thread.start()
            logging.info("Performance tracking started")
            
        except Exception as e:
            logging.error(f"Error starting performance tracking: {e}")
    
    def stop_tracking(self):
        """Stop performance tracking"""
        try:
            self.is_tracking = False
            if self.tracking_thread:
                self.tracking_thread.join(timeout=5)
            logging.info("Performance tracking stopped")
            
        except Exception as e:
            logging.error(f"Error stopping performance tracking: {e}")
    
    def tracking_loop(self):
        """Main tracking loop"""
        while self.is_tracking:
            try:
                # Update daily statistics
                self.update_daily_stats()
                time.sleep(3600)  # Update every hour
                
            except Exception as e:
                logging.error(f"Error in tracking loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def record_signal(self, signal: Dict, result: str = 'PENDING', profit_loss: float = 0):
        """Record a signal in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO signals 
                (id, timestamp, pair, direction, accuracy, ai_confidence, result, profit_loss, signal_strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.get('id', ''),
                datetime.now(),
                signal.get('pair', ''),
                signal.get('direction', ''),
                signal.get('accuracy', 0),
                signal.get('ai_confidence', 0),
                result,
                profit_loss,
                signal.get('strength', 0)
            ))
            
            conn.commit()
            conn.close()
            logging.debug(f"Signal recorded: {signal.get('id')}")
            
        except Exception as e:
            logging.error(f"Error recording signal: {e}")
    
    def update_signal_result(self, signal_id: str, result: str, profit_loss: float = 0):
        """Update signal result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE signals 
                SET result = ?, profit_loss = ?
                WHERE id = ?
            ''', (result, profit_loss, signal_id))
            
            conn.commit()
            conn.close()
            logging.debug(f"Signal result updated: {signal_id} -> {result}")
            
        except Exception as e:
            logging.error(f"Error updating signal result: {e}")
    
    def update_daily_stats(self):
        """Update daily statistics"""
        try:
            today = datetime.now().date()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get today's signals
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                       AVG(accuracy) as avg_accuracy,
                       SUM(profit_loss) as total_pl
                FROM signals 
                WHERE DATE(timestamp) = ?
            ''', (today,))
            
            result = cursor.fetchone()
            
            if result and result[0] > 0:
                total, wins, losses, avg_accuracy, total_pl = result
                win_rate = (wins / total) * 100 if total > 0 else 0
                
                cursor.execute('''
                    INSERT OR REPLACE INTO daily_stats 
                    (date, total_signals, winning_signals, losing_signals, win_rate, total_profit_loss, average_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (today, total, wins, losses, win_rate, total_pl or 0, avg_accuracy or 0))
                
                conn.commit()
            
            conn.close()
            
        except Exception as e:
            logging.error(f"Error updating daily stats: {e}")
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Overall stats
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                       AVG(accuracy) as avg_accuracy,
                       SUM(profit_loss) as total_pl,
                       AVG(signal_strength) as avg_strength
                FROM signals 
                WHERE result IN ('WIN', 'LOSS')
            ''')
            
            overall = cursor.fetchone()
            
            # Today's stats
            today = datetime.now().date()
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                       AVG(accuracy) as avg_accuracy
                FROM signals 
                WHERE DATE(timestamp) = ? AND result IN ('WIN', 'LOSS')
            ''', (today,))
            
            today_stats = cursor.fetchone()
            
            # Weekly stats
            week_ago = datetime.now() - timedelta(days=7)
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins
                FROM signals 
                WHERE timestamp >= ? AND result IN ('WIN', 'LOSS')
            ''', (week_ago,))
            
            weekly_stats = cursor.fetchone()
            
            # Monthly stats
            month_ago = datetime.now() - timedelta(days=30)
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins
                FROM signals 
                WHERE timestamp >= ? AND result IN ('WIN', 'LOSS')
            ''', (month_ago,))
            
            monthly_stats = cursor.fetchone()
            
            conn.close()
            
            # Calculate metrics
            total_trades = overall[0] if overall else 0
            winning_trades = overall[1] if overall else 0
            losing_trades = overall[2] if overall else 0
            overall_accuracy = overall[3] if overall else 0
            total_pl = overall[4] if overall else 0
            avg_strength = overall[5] if overall else 0
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            loss_rate = (losing_trades / total_trades) * 100 if total_trades > 0 else 0
            
            today_total = today_stats[0] if today_stats else 0
            today_wins = today_stats[1] if today_stats else 0
            today_win_rate = (today_wins / today_total) * 100 if today_total > 0 else 0
            
            weekly_total = weekly_stats[0] if weekly_stats else 0
            weekly_wins = weekly_stats[1] if weekly_stats else 0
            weekly_win_rate = (weekly_wins / weekly_total) * 100 if weekly_total > 0 else 0
            
            monthly_total = monthly_stats[0] if monthly_stats else 0
            monthly_wins = monthly_stats[1] if monthly_stats else 0
            monthly_win_rate = (monthly_wins / monthly_total) * 100 if monthly_total > 0 else 0
            
            return {
                'overall_accuracy': round(overall_accuracy, 1),
                'win_rate': round(win_rate, 1),
                'loss_rate': round(loss_rate, 1),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'profit_factor': self.calculate_profit_factor(),
                'current_streak': self.get_current_streak(),
                'today_signals': today_total,
                'today_win_rate': round(today_win_rate, 1),
                'weekly_win_rate': round(weekly_win_rate, 1),
                'monthly_win_rate': round(monthly_win_rate, 1),
                'best_day_win_rate': self.get_best_day_win_rate(),
                'longest_streak': self.get_longest_streak(),
                'avg_signal_strength': round(avg_strength, 1),
                'tracking_period': self.get_tracking_period_days(),
                'total_profit_loss': round(total_pl, 2)
            }
            
        except Exception as e:
            logging.error(f"Error getting comprehensive stats: {e}")
            return self.get_default_stats()
    
    def get_default_stats(self) -> Dict:
        """Return default stats when error occurs"""
        return {
            'overall_accuracy': 0,
            'win_rate': 0,
            'loss_rate': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'profit_factor': 0,
            'current_streak': 0,
            'today_signals': 0,
            'today_win_rate': 0,
            'weekly_win_rate': 0,
            'monthly_win_rate': 0,
            'best_day_win_rate': 0,
            'longest_streak': 0,
            'avg_signal_strength': 0,
            'tracking_period': 0,
            'total_profit_loss': 0
        }
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as total_profit,
                       SUM(CASE WHEN profit_loss < 0 THEN ABS(profit_loss) ELSE 0 END) as total_loss
                FROM signals 
                WHERE result IN ('WIN', 'LOSS')
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[1] > 0:
                return result[0] / result[1]
            else:
                return 0
                
        except Exception as e:
            logging.error(f"Error calculating profit factor: {e}")
            return 0
    
    def get_current_streak(self) -> int:
        """Get current winning/losing streak"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT result FROM signals 
                WHERE result IN ('WIN', 'LOSS')
                ORDER BY timestamp DESC 
                LIMIT 20
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return 0
            
            current_result = results[0][0]
            streak = 0
            
            for result in results:
                if result[0] == current_result:
                    streak += 1
                else:
                    break
            
            return streak if current_result == 'WIN' else -streak
            
        except Exception as e:
            logging.error(f"Error getting current streak: {e}")
            return 0
    
    def get_longest_streak(self) -> int:
        """Get longest winning streak"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT result FROM signals 
                WHERE result IN ('WIN', 'LOSS')
                ORDER BY timestamp ASC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return 0
            
            max_streak = 0
            current_streak = 0
            
            for result in results:
                if result[0] == 'WIN':
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            
            return max_streak
            
        except Exception as e:
            logging.error(f"Error getting longest streak: {e}")
            return 0
    
    def get_best_day_win_rate(self) -> float:
        """Get best single day win rate"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT MAX(win_rate) FROM daily_stats
                WHERE total_signals >= 3
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result and result[0] else 0
            
        except Exception as e:
            logging.error(f"Error getting best day win rate: {e}")
            return 0
    
    def get_tracking_period_days(self) -> int:
        """Get number of days being tracked"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT MIN(timestamp) FROM signals
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                start_date = datetime.fromisoformat(result[0])
                return (datetime.now() - start_date).days
            else:
                return 0
                
        except Exception as e:
            logging.error(f"Error getting tracking period: {e}")
            return 0