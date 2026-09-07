"""
Módulo de Varredura por Faixa de ATR (ATR Sweep Test: 1.0x a 3.5x ATR).
Testa a sensibilidade do Stop Loss/Take Profit em incrementos de 0.25x ATR para identificar o planalto de robustez.
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INITIAL_CAPITAL
from src.data_loader import fetch_historical_data
from src.strategy import generate_signals, calculate_indicators
from src.backtester import Backtester

def run_atr_sweep_test(symbol: str = "BTC-USD", days: int = 730) -> dict:
    """
    Executa a varredura completa da faixa de multiplicadores de ATR de 1.0x a 3.5x.
    """
    print(f"\n========================================================")
    print(f"  VARREDURA POR FAIXA DE ATR (ATR Sweep Test: 1.0x - 3.5x)")
    print(f"  Ativo: {symbol} | Período: {days} dias | Taxas + Slippage: 0.15%")
    print(f"========================================================\n")
    
    df = fetch_historical_data(symbol=symbol, days=days)
    if df.empty:
        raise ValueError("DataFrame histórico vazio.")
        
    atr_multipliers = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5]
    sweep_results = {}
    
    for mult in atr_multipliers:
        df_mod = df.copy()
        df_mod = calculate_indicators(df_mod, entry_window=30, exit_window=10)
        
        # ATR com multiplicador testado
        high_low = df_mod['high'] - df_mod['low']
        high_close = (df_mod['high'] - df_mod['close'].shift(1)).abs()
        low_close = (df_mod['low'] - df_mod['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_mod['atr'] = tr.shift(1).rolling(window=14).mean() * (mult / 2.0)
        
        buy_cond = (df_mod['close'] > df_mod['donchian_high']) & (df_mod['close'] > df_mod['ema_200'])
        sell_cond = df_mod['close'] < df_mod['donchian_low']
        
        df_mod['signal'] = 0
        df_mod.loc[buy_cond, 'signal'] = 1
        df_mod.loc[sell_cond, 'signal'] = -1
        
        bt = Backtester(df_mod, initial_capital=INITIAL_CAPITAL)
        res = bt.run()
        
        label = f"{mult:.2f}x ATR"
        sweep_results[label] = {
            "multiplier": mult,
            "total_trades": res['total_trades'],
            "return_pct": res['total_return_pct'],
            "final_capital": res['final_capital'],
            "win_rate_pct": res['win_rate_pct'],
            "profit_factor": res['profit_factor'],
            "expectancy_usd": res['expectancy_usd'],
            "max_drawdown_pct": res['max_drawdown_pct'],
            "sharpe_ratio": res['sharpe_ratio']
        }
        
        print(f"[{label}]: Retorno: {res['total_return_pct']:+.2f}% | Cap: ${res['final_capital']:,.2f} | Trades: {res['total_trades']} | WR: {res['win_rate_pct']:.1f}% | PF: {res['profit_factor']:.2f} | Exp: ${res['expectancy_usd']:+.2f} | MaxDD: -{res['max_drawdown_pct']:.2f}%")
        
    return sweep_results

if __name__ == '__main__':
    run_atr_sweep_test()
