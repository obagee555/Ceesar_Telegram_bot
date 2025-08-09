import logging
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class RiskParameters:
    max_daily_trades: int = 10
    max_consecutive_losses: int = 3
    max_drawdown_percent: float = 15.0
    min_accuracy_threshold: float = 95.0
    position_size_percent: float = 2.0  # % of account per trade
    stop_loss_percent: float = 5.0
    max_risk_per_trade: float = 1.0  # % of account

class RiskManager:
    def __init__(self):
        self.risk_params = RiskParameters()
        self.risk_level = "CONSERVATIVE"
        
        # Daily tracking
        self.daily_trades = 0
        self.daily_losses = 0
        self.consecutive_losses = 0
        self.current_drawdown = 0.0
        self.max_drawdown_reached = 0.0
        
        # Trade history for risk calculation
        self.trade_history = []
        self.daily_reset_time = None
        
        # Risk states
        self.is_trading_halted = False
        self.halt_reason = ""
        self.halt_timestamp = None
        
        # Account simulation (for risk calculation)
        self.account_balance = 10000  # Simulated starting balance
        self.peak_balance = 10000
        
        self.setup_risk_levels()
        
    def setup_risk_levels(self):
        """Setup different risk management levels"""
        self.risk_levels = {
            "CONSERVATIVE": {
                "max_daily_trades": 5,
                "max_consecutive_losses": 2,
                "max_drawdown_percent": 10.0,
                "position_size_percent": 1.0,
                "min_accuracy_threshold": 97.0
            },
            "MODERATE": {
                "max_daily_trades": 10,
                "max_consecutive_losses": 3,
                "max_drawdown_percent": 15.0,
                "position_size_percent": 2.0,
                "min_accuracy_threshold": 95.0
            },
            "AGGRESSIVE": {
                "max_daily_trades": 20,
                "max_consecutive_losses": 5,
                "max_drawdown_percent": 25.0,
                "position_size_percent": 3.0,
                "min_accuracy_threshold": 93.0
            }
        }
    
    def set_risk_level(self, level: str):
        """Set risk management level"""
        try:
            if level in self.risk_levels:
                self.risk_level = level
                params = self.risk_levels[level]
                
                self.risk_params.max_daily_trades = params["max_daily_trades"]
                self.risk_params.max_consecutive_losses = params["max_consecutive_losses"]
                self.risk_params.max_drawdown_percent = params["max_drawdown_percent"]
                self.risk_params.position_size_percent = params["position_size_percent"]
                self.risk_params.min_accuracy_threshold = params["min_accuracy_threshold"]
                
                logging.info(f"Risk level set to: {level}")
                return True
            else:
                logging.error(f"Invalid risk level: {level}")
                return False
                
        except Exception as e:
            logging.error(f"Error setting risk level: {e}")
            return False
    
    def evaluate_signal_risk(self, signal: Dict) -> Dict:
        """Evaluate risk for a given signal"""
        try:
            risk_score = 0
            risk_factors = []
            
            # Check accuracy
            accuracy = signal.get('accuracy', 0)
            if accuracy < self.risk_params.min_accuracy_threshold:
                risk_score += 30
                risk_factors.append(f"Low accuracy: {accuracy}%")
            
            # Check volatility
            volatility = signal.get('volatility', 0)
            if volatility > 0.02:  # High volatility threshold
                risk_score += 20
                risk_factors.append(f"High volatility: {volatility:.4f}")
            
            # Check signal strength
            strength = signal.get('strength', 0)
            if strength < 6:
                risk_score += 15
                risk_factors.append(f"Weak signal strength: {strength}/10")
            
            # Check AI confidence
            ai_confidence = signal.get('ai_confidence', 0)
            if ai_confidence < 85:
                risk_score += 25
                risk_factors.append(f"Low AI confidence: {ai_confidence}%")
            
            # Check consecutive losses
            if self.consecutive_losses >= self.risk_params.max_consecutive_losses:
                risk_score += 40
                risk_factors.append(f"Max consecutive losses reached: {self.consecutive_losses}")
            
            # Check daily trade limit
            if self.daily_trades >= self.risk_params.max_daily_trades:
                risk_score += 50
                risk_factors.append(f"Daily trade limit reached: {self.daily_trades}")
            
            # Check drawdown
            if self.current_drawdown >= self.risk_params.max_drawdown_percent:
                risk_score += 60
                risk_factors.append(f"Max drawdown reached: {self.current_drawdown:.1f}%")
            
            # Determine risk level
            if risk_score >= 50:
                risk_level = "HIGH"
                recommendation = "REJECT"
            elif risk_score >= 30:
                risk_level = "MEDIUM"
                recommendation = "CAUTION"
            else:
                risk_level = "LOW"
                recommendation = "ACCEPT"
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'risk_factors': risk_factors,
                'position_size': self.calculate_position_size(signal, risk_score)
            }
            
        except Exception as e:
            logging.error(f"Error evaluating signal risk: {e}")
            return {
                'risk_score': 100,
                'risk_level': "HIGH",
                'recommendation': "REJECT",
                'risk_factors': ["Risk evaluation failed"],
                'position_size': 0
            }
    
    def calculate_position_size(self, signal: Dict, risk_score: int) -> float:
        """Calculate recommended position size based on risk"""
        try:
            base_position = self.risk_params.position_size_percent
            
            # Adjust based on accuracy
            accuracy_multiplier = signal.get('accuracy', 95) / 100
            
            # Adjust based on risk score
            risk_multiplier = max(0.1, 1 - (risk_score / 100))
            
            # Adjust based on consecutive losses
            loss_multiplier = max(0.5, 1 - (self.consecutive_losses * 0.2))
            
            # Calculate final position size
            position_size = base_position * accuracy_multiplier * risk_multiplier * loss_multiplier
            
            # Cap at maximum risk per trade
            max_position = self.risk_params.max_risk_per_trade
            position_size = min(position_size, max_position)
            
            return round(position_size, 2)
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.5  # Conservative default
    
    def should_halt_trading(self) -> bool:
        """Check if trading should be halted"""
        try:
            # Check if already halted
            if self.is_trading_halted:
                return True
            
            # Check consecutive losses
            if self.consecutive_losses >= self.risk_params.max_consecutive_losses:
                self.halt_trading("Maximum consecutive losses reached")
                return True
            
            # Check daily trade limit
            if self.daily_trades >= self.risk_params.max_daily_trades:
                self.halt_trading("Daily trade limit reached")
                return True
            
            # Check drawdown
            if self.current_drawdown >= self.risk_params.max_drawdown_percent:
                self.halt_trading("Maximum drawdown reached")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking halt conditions: {e}")
            return True  # Halt on error for safety
    
    def halt_trading(self, reason: str):
        """Halt trading with specified reason"""
        try:
            self.is_trading_halted = True
            self.halt_reason = reason
            self.halt_timestamp = datetime.now()
            logging.warning(f"Trading halted: {reason}")
            
        except Exception as e:
            logging.error(f"Error halting trading: {e}")
    
    def resume_trading(self):
        """Resume trading (manual override)"""
        try:
            self.is_trading_halted = False
            self.halt_reason = ""
            self.halt_timestamp = None
            logging.info("Trading resumed manually")
            
        except Exception as e:
            logging.error(f"Error resuming trading: {e}")
    
    def record_trade_result(self, signal: Dict, result: str, profit_loss: float = 0):
        """Record the result of a trade"""
        try:
            trade_record = {
                'timestamp': datetime.now(),
                'pair': signal.get('pair', ''),
                'direction': signal.get('direction', ''),
                'accuracy': signal.get('accuracy', 0),
                'result': result,  # 'WIN', 'LOSS', 'PENDING'
                'profit_loss': profit_loss,
                'position_size': signal.get('position_size', 0)
            }
            
            self.trade_history.append(trade_record)
            
            # Update statistics
            if result == 'WIN':
                self.consecutive_losses = 0
                self.account_balance += abs(profit_loss)
                
                # Update peak balance
                if self.account_balance > self.peak_balance:
                    self.peak_balance = self.account_balance
                    
            elif result == 'LOSS':
                self.daily_losses += 1
                self.consecutive_losses += 1
                self.account_balance -= abs(profit_loss)
            
            # Update drawdown
            self.calculate_current_drawdown()
            
            # Check if trading should be halted
            self.should_halt_trading()
            
            # Limit history size
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-500:]
            
            logging.info(f"Trade result recorded: {result} for {signal.get('pair')}")
            
        except Exception as e:
            logging.error(f"Error recording trade result: {e}")
    
    def calculate_current_drawdown(self):
        """Calculate current drawdown percentage"""
        try:
            if self.peak_balance > 0:
                drawdown = ((self.peak_balance - self.account_balance) / self.peak_balance) * 100
                self.current_drawdown = max(0, drawdown)
                
                # Track maximum drawdown reached
                if self.current_drawdown > self.max_drawdown_reached:
                    self.max_drawdown_reached = self.current_drawdown
                    
        except Exception as e:
            logging.error(f"Error calculating drawdown: {e}")
    
    def reset_daily_counters(self):
        """Reset daily counters (call at start of new day)"""
        try:
            self.daily_trades = 0
            self.daily_losses = 0
            self.daily_reset_time = datetime.now()
            
            # Auto-resume trading if halted due to daily limits
            if self.is_trading_halted and "daily" in self.halt_reason.lower():
                self.resume_trading()
            
            logging.info("Daily risk counters reset")
            
        except Exception as e:
            logging.error(f"Error resetting daily counters: {e}")
    
    def get_risk_status(self) -> Dict:
        """Get current risk management status"""
        try:
            return {
                'risk_level': self.risk_level,
                'is_trading_halted': self.is_trading_halted,
                'halt_reason': self.halt_reason,
                'daily_trades': self.daily_trades,
                'daily_trades_left': max(0, self.risk_params.max_daily_trades - self.daily_trades),
                'consecutive_losses': self.consecutive_losses,
                'current_drawdown': round(self.current_drawdown, 2),
                'max_drawdown': round(self.max_drawdown_reached, 2),
                'account_balance': round(self.account_balance, 2),
                'peak_balance': round(self.peak_balance, 2),
                'total_trades': len(self.trade_history),
                'last_reset': self.daily_reset_time.isoformat() if self.daily_reset_time else None
            }
            
        except Exception as e:
            logging.error(f"Error getting risk status: {e}")
            return {
                'risk_level': 'UNKNOWN',
                'is_trading_halted': True,
                'halt_reason': 'Error retrieving status',
                'daily_trades': 0,
                'daily_trades_left': 0,
                'consecutive_losses': 0,
                'current_drawdown': 0,
                'max_drawdown': 0,
                'account_balance': 0,
                'peak_balance': 0,
                'total_trades': 0,
                'last_reset': None
            }
    
    def get_risk_metrics(self) -> Dict:
        """Get detailed risk metrics"""
        try:
            recent_trades = self.trade_history[-50:] if self.trade_history else []
            
            if not recent_trades:
                return {
                    'win_rate': 0,
                    'average_accuracy': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'max_consecutive_wins': 0,
                    'max_consecutive_losses': 0,
                    'average_trade_duration': 0
                }
            
            # Calculate win rate
            wins = len([t for t in recent_trades if t['result'] == 'WIN'])
            win_rate = (wins / len(recent_trades)) * 100 if recent_trades else 0
            
            # Calculate average accuracy
            avg_accuracy = np.mean([t['accuracy'] for t in recent_trades])
            
            # Calculate profit factor
            total_profit = sum([t['profit_loss'] for t in recent_trades if t['profit_loss'] > 0])
            total_loss = abs(sum([t['profit_loss'] for t in recent_trades if t['profit_loss'] < 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate consecutive streaks
            max_consecutive_wins = self.calculate_max_consecutive('WIN', recent_trades)
            max_consecutive_losses = self.calculate_max_consecutive('LOSS', recent_trades)
            
            return {
                'win_rate': round(win_rate, 2),
                'average_accuracy': round(avg_accuracy, 2),
                'profit_factor': round(profit_factor, 2),
                'sharpe_ratio': self.calculate_sharpe_ratio(recent_trades),
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'total_profit': round(total_profit, 2),
                'total_loss': round(total_loss, 2),
                'recent_trades_count': len(recent_trades)
            }
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def calculate_max_consecutive(self, result_type: str, trades: List) -> int:
        """Calculate maximum consecutive wins or losses"""
        try:
            max_consecutive = 0
            current_consecutive = 0
            
            for trade in trades:
                if trade['result'] == result_type:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            logging.error(f"Error calculating consecutive {result_type}: {e}")
            return 0
    
    def calculate_sharpe_ratio(self, trades: List) -> float:
        """Calculate Sharpe ratio for risk-adjusted returns"""
        try:
            if len(trades) < 2:
                return 0
            
            returns = [t['profit_loss'] for t in trades]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Risk-free rate (assumed 0 for binary options)
            risk_free_rate = 0
            
            if std_return == 0:
                return 0
            
            sharpe_ratio = (mean_return - risk_free_rate) / std_return
            return round(sharpe_ratio, 3)
            
        except Exception as e:
            logging.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def update_daily_trade_count(self):
        """Increment daily trade counter"""
        self.daily_trades += 1
    
    def get_risk_recommendations(self) -> List[str]:
        """Get risk management recommendations"""
        try:
            recommendations = []
            
            # Check consecutive losses
            if self.consecutive_losses >= 2:
                recommendations.append("Consider reducing position size due to consecutive losses")
            
            # Check drawdown
            if self.current_drawdown > 10:
                recommendations.append("Current drawdown is significant - consider taking a break")
            
            # Check daily trades
            remaining_trades = self.risk_params.max_daily_trades - self.daily_trades
            if remaining_trades <= 2:
                recommendations.append(f"Only {remaining_trades} trades remaining for today")
            
            # Check win rate
            recent_trades = self.trade_history[-10:] if len(self.trade_history) >= 10 else self.trade_history
            if recent_trades:
                recent_wins = len([t for t in recent_trades if t['result'] == 'WIN'])
                recent_win_rate = (recent_wins / len(recent_trades)) * 100
                
                if recent_win_rate < 70:
                    recommendations.append("Recent win rate is below optimal - review strategy")
            
            if not recommendations:
                recommendations.append("Risk levels are within acceptable parameters")
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error getting risk recommendations: {e}")
            return ["Error retrieving recommendations"]