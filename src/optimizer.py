"""
Módulo Otimizador de Parâmetros (Grid Search).
"""
import pandas as pd
import numpy as np
import os
import sys
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INITIAL_CAPITAL
from src.backtester import Backtester

def generate_custom_signals(df: pd.DataFrame, entry_window: int, exit_window: int, ema_trend_window: int) -> pd.DataFrame:
    """
    Gera sinais customizados para uma combinação específica de parâmetros.
    """
    df = df.copy()
    
    df['donchian_high'] = df['high'].shift(1).rolling(window=entry_window).max()
    df['donchian_low'] = df['low'].shift(1).rolling(window=exit_window).min()
    df['ema_trend'] = df['close'].ewm(span=ema_trend_window, adjust=False).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    df['signal'] = 0
    buy_condition = (df['close'] > df['donchian_high']) & (df['close'] > df['ema_trend'])
    sell_condition = (df['close'] < df['donchian_low'])
    
    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1
    
    return df

def run_grid_search(df: pd.DataFrame, 
                    entry_range=[10, 15, 20, 25, 30], 
                    exit_range=[5, 10, 15], 
                    ema_range=[100, 150, 200]) -> pd.DataFrame:
    """
    Executa busca em grade (Grid Search) testando todas as combinações de parâmetros.
    """
    results = []
    total_combinations = len(entry_range) * len(exit_range) * len(ema_range)
    print(f"[Otimizador] Iniciando Grid Search com {total_combinations} combinações possíveis...")
    
    count = 0
    for entry_w, exit_w, ema_w in product(entry_range, exit_range, ema_range):
        count += 1
        df_signals = generate_custom_signals(df, entry_w, exit_w, ema_w)
        backtester = Backtester(df_signals, initial_capital=INITIAL_CAPITAL)
        metrics = backtester.run()
        
        # Só considera combinações que realizaram operações
        if metrics['total_trades'] > 0:
            results.append({
                'entry_window': entry_w,
                'exit_window': exit_w,
                'ema_trend': ema_w,
                'total_return_pct': metrics['total_return_pct'],
                'win_rate_pct': metrics['win_rate_pct'],
                'profit_factor': metrics['profit_factor'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'total_trades': metrics['total_trades'],
                'final_capital': metrics['final_capital']
            })
            
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        # Ordena pelo Profit Factor e Retorno Total
        df_results = df_results.sort_values(by=['profit_factor', 'total_return_pct'], ascending=False).reset_index(drop=True)
    
    print(f"[Otimizador] Concluído! Melhores combinações encontradas.")
    return df_results

if __name__ == '__main__':
    from src.data_loader import fetch_historical_data
    df = fetch_historical_data()
    best_params = run_grid_search(df)
    print(best_params.head(10))
