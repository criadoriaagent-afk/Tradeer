"""
Módulo de Estudo de Ablação dos Filtros de Segurança (Filter Ablation Study).
Avalia o valor estatístico (alpha) real adicionado por cada um dos filtros.
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

def run_filter_ablation_study(symbol: str = "BTC-USD", days: int = 730) -> dict:
    """
    Executa o estudo de ablação isolando o efeito de cada filtro.
    """
    print(f"\n========================================================")
    print(f"  ESTUDO DE ABLAÇÃO DOS FILTROS (Filter Ablation Study)")
    print(f"  Ativo: {symbol} | Período: {days} dias")
    print(f"========================================================\n")
    
    df = fetch_historical_data(symbol=symbol, days=days)
    if df.empty:
        raise ValueError("DataFrame histórico vazio.")
        
    configurations = {
        "1. Todos os Filtros Ativos": {"use_ema200": True, "use_volume": True, "use_adx": True, "use_donchian": True},
        "2. Sem Filtro de Volume": {"use_ema200": True, "use_volume": False, "use_adx": True, "use_donchian": True},
        "3. Sem Filtro de Regime ADX": {"use_ema200": True, "use_volume": True, "use_adx": False, "use_donchian": True},
        "4. Sem Filtro EMA 200 Macro": {"use_ema200": False, "use_volume": True, "use_adx": True, "use_donchian": True},
        "5. Donchian Puro (Sem Filtros)": {"use_ema200": False, "use_volume": False, "use_adx": False, "use_donchian": True}
    }
    
    ablation_results = {}
    
    for name, config in configurations.items():
        df_mod = df.copy()
        df_mod = calculate_indicators(df_mod, entry_window=30, exit_window=10)
        
        # Volume
        high_low = df_mod['high'] - df_mod['low']
        high_close = (df_mod['high'] - df_mod['close'].shift(1)).abs()
        low_close = (df_mod['low'] - df_mod['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_mod['atr'] = tr.shift(1).rolling(window=14).mean()
        df_mod['vol_sma'] = df_mod['volume'].shift(1).rolling(window=20).mean()
        
        buy_cond = df_mod['close'] > df_mod['donchian_high']
        if config["use_ema200"]:
            buy_cond = buy_cond & (df_mod['close'] > df_mod['ema_200'])
        if config["use_volume"]:
            buy_cond = buy_cond & (df_mod['volume'].shift(1) >= 1.20 * df_mod['vol_sma'])
            
        sell_cond = df_mod['close'] < df_mod['donchian_low']
        
        df_mod['signal'] = 0
        df_mod.loc[buy_cond, 'signal'] = 1
        df_mod.loc[sell_cond, 'signal'] = -1
        
        bt = Backtester(df_mod, initial_capital=INITIAL_CAPITAL)
        res = bt.run()
        
        ablation_results[name] = {
            "total_trades": res['total_trades'],
            "return_pct": res['total_return_pct'],
            "final_capital": res['final_capital'],
            "win_rate_pct": res['win_rate_pct'],
            "profit_factor": res['profit_factor'],
            "expectancy_usd": res['expectancy_usd'],
            "max_drawdown_pct": res['max_drawdown_pct'],
            "sharpe_ratio": res['sharpe_ratio']
        }
        
        print(f"[{name}]:")
        print(f"   Retorno: {res['total_return_pct']:+.2f}% | Capital: ${res['final_capital']:,.2f} | Trades: {res['total_trades']}")
        print(f"   Win Rate: {res['win_rate_pct']:.1f}% | Profit Factor: {res['profit_factor']:.2f} | Expectancy: ${res['expectancy_usd']:+.2f}")
        print(f"   Max Drawdown: -{res['max_drawdown_pct']:.2f}% | Sharpe: {res['sharpe_ratio']:.2f}\n")
        
    return ablation_results

if __name__ == '__main__':
    run_filter_ablation_study()
