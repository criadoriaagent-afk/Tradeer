"""
Motor de Backtest Estatístico para Avaliação de Performance.
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INITIAL_CAPITAL
from src.risk_manager import RiskManager

class Backtester:
    def __init__(self, df_signals: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL):
        self.df = df_signals.copy()
        self.initial_capital = initial_capital
        self.risk_manager = RiskManager(initial_capital)
        self.trades = []
        self.equity_curve = []
        
    def run(self) -> dict:
        """
        Executa a simulação bar a bar (candle por candle).
        """
        capital = self.initial_capital
        position = None  # None ou dict com informações da posição aberta
        
        current_day = None
        
        for index, row in self.df.iterrows():
            current_price = row['close']
            atr = row.get('atr', current_price * 0.02) # Fallback para 2% caso ATR seja NaN
            signal = row['signal']
            
            # Reset/Verificação Diária de Segurança
            row_day = index.date() if hasattr(index, 'date') else index
            if current_day != row_day:
                current_day = row_day
                self.risk_manager.daily_start_capital = capital
                self.risk_manager.is_daily_halted = False
                
            # Adiciona capital atual na curva de patrimônio
            self.equity_curve.append({
                'timestamp': index,
                'equity': capital + (position['units'] * current_price if position else 0)
            })
            
            # Se a trava de segurança diária disparou, ignora novas entradas no dia
            if self.risk_manager.is_daily_halted:
                continue

            # 1. GERENCIAMENTO DE POSIÇÃO ABERTA (Se já estiver posicionado)
            if position is not None:
                units = position['units']
                entry_price = position['entry_price']
                
                # Atualiza Trailing Stop dinâmico com a máxima da vela e ATR atual
                position['stop_loss'] = self.risk_manager.update_trailing_stop(
                    current_stop_loss=position['stop_loss'],
                    current_high=row['high'],
                    atr=atr
                )
                
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']
                
                # Checa Stop Loss (Garoa de perdas)
                if row['low'] <= stop_loss:
                    exit_price = stop_loss
                    gross_pnl = (exit_price - entry_price) * units
                    costs = self.risk_manager.apply_costs(units * exit_price)
                    net_pnl = gross_pnl - costs
                    capital += (units * exit_price) - costs
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': index,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'type': 'STOP_LOSS',
                        'pnl': net_pnl,
                        'pnl_pct': (net_pnl / position['invested_capital']) * 100
                    })
                    position = None
                    self.risk_manager.check_daily_halt(index, capital)
                    continue

                # Checa Take Profit (Alvo de Lucro)
                elif row['high'] >= take_profit:
                    exit_price = take_profit
                    gross_pnl = (exit_price - entry_price) * units
                    costs = self.risk_manager.apply_costs(units * exit_price)
                    net_pnl = gross_pnl - costs
                    capital += (units * exit_price) - costs
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': index,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'type': 'TAKE_PROFIT',
                        'pnl': net_pnl,
                        'pnl_pct': (net_pnl / position['invested_capital']) * 100
                    })
                    position = None
                    continue

                # Checa Sinal de Saída pela Estratégia
                elif signal == -1:
                    exit_price = current_price
                    gross_pnl = (exit_price - entry_price) * units
                    costs = self.risk_manager.apply_costs(units * exit_price)
                    net_pnl = gross_pnl - costs
                    capital += (units * exit_price) - costs
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': index,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'type': 'SIGNAL_EXIT',
                        'pnl': net_pnl,
                        'pnl_pct': (net_pnl / position['invested_capital']) * 100
                    })
                    position = None
                    continue

            # 2. ENTRADA EM NOVA POSIÇÃO (Se não estiver posicionado)
            elif signal == 1 and not np.isnan(atr):
                units, stop_loss, take_profit = self.risk_manager.calculate_position_size(
                    current_capital=capital,
                    entry_price=current_price,
                    atr=atr
                )
                
                if units > 0:
                    invested_capital = units * current_price
                    costs = self.risk_manager.apply_costs(invested_capital)
                    capital -= (invested_capital + costs)
                    
                    position = {
                        'entry_time': index,
                        'entry_price': current_price,
                        'units': units,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'invested_capital': invested_capital
                    }

        # Converte equity curve para DataFrame
        df_equity = pd.DataFrame(self.equity_curve).set_index('timestamp')
        return self._generate_metrics(df_equity, capital)

    def _generate_metrics(self, df_equity: pd.DataFrame, final_capital: float) -> dict:
        """
        Calcula as métricas estatísticas da performance do backtest.
        """
        total_trades = len(self.trades)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'final_capital': self.initial_capital,
                'total_return_pct': 0.0,
                'win_rate_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'profit_factor': 0.0,
                'trades_df': pd.DataFrame(),
                'equity_df': df_equity
            }

        df_trades = pd.DataFrame(self.trades)
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] < 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100.0
        total_gain = winning_trades['pnl'].sum()
        total_loss = abs(losing_trades['pnl'].sum())
        profit_factor = total_gain / total_loss if total_loss > 0 else np.inf
        
        # Max Drawdown
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak']
        max_drawdown = abs(df_equity['drawdown'].min()) * 100.0
        
        total_return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100.0

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'final_capital': round(final_capital, 2),
            'total_return_pct': round(total_return_pct, 2),
            'win_rate_pct': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'trades_df': df_trades,
            'equity_df': df_equity
        }
