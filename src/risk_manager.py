"""
Módulo de Gerenciamento de Risco e Dimensionamento de Posição.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RISK_PER_TRADE, MAX_DAILY_LOSS, RISK_REWARD_RATIO, FEE_RATE, SLIPPAGE
)

class RiskManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.is_daily_halted = False
        
    def check_daily_halt(self, current_time, current_capital: float) -> bool:
        """
        Verifica se a perda do dia ultrapassou o limite diário de segurança (MAX_DAILY_LOSS).
        """
        daily_loss = (self.daily_start_capital - current_capital) / self.daily_start_capital
        if daily_loss >= MAX_DAILY_LOSS:
            self.is_daily_halted = True
            print(f"[RiskManager] TRAVA DE SEGURANÇA ATIVADA! Perda diária de {daily_loss*100:.2f}% atingiu o limite de {MAX_DAILY_LOSS*100:.1f}%.")
            return True
        return False
        
    def calculate_position_size(self, current_capital: float, entry_price: float, atr: float) -> tuple:
        """
        Calcula o tamanho da ordem e os níveis exatos de Stop Loss e Take Profit.
        
        Risco Máximo em $ = Capital Atual * RISK_PER_TRADE (ex: 1%)
        Distância do Stop Loss = 1.5 * ATR (Volatilidade real)
        Preço do Stop Loss = Entry Price - Distância do Stop
        Preço do Take Profit = Entry Price + (Distância do Stop * RISK_REWARD_RATIO)
        Tamanho da Posição (unidades) = Risco Máximo em $ / Distância do Stop
        """
        if current_capital <= 0 or entry_price <= 0 or atr <= 0:
            return 0.0, 0.0, 0.0
            
        risk_amount = current_capital * RISK_PER_TRADE
        stop_distance = 1.5 * atr
        
        stop_loss_price = entry_price - stop_distance
        take_profit_price = entry_price + (stop_distance * RISK_REWARD_RATIO)
        
        # Evita divisão por zero
        if stop_distance <= 0:
            return 0.0, 0.0, 0.0
            
        units = risk_amount / stop_distance
        
        # Garante que o valor investido não ultrapasse o capital disponível com alavancagem de 1x
        max_units = (current_capital * 0.95) / entry_price # usa no máximo 95% do capital
        units = min(units, max_units)
        
        return units, stop_loss_price, take_profit_price

    def update_trailing_stop(self, current_stop_loss: float, current_high: float, atr: float) -> float:
        """
        Atualiza o Stop Loss dinamicamente (Trailing Stop) à medida que o preço avança a favor da operação.
        O Stop Loss nunca cai, apenas sobe.
        """
        if atr <= 0:
            return current_stop_loss
            
        new_stop = current_high - (1.5 * atr)
        return max(current_stop_loss, new_stop)

    def apply_costs(self, trade_value: float) -> float:
        """
        Aplica a taxa de corretagem + slippage à operação.
        """
        total_cost_pct = FEE_RATE + SLIPPAGE
        return trade_value * total_cost_pct
