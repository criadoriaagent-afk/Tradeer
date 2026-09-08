import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import INITIAL_CAPITAL
from src.core.data_loader import fetch_historical_data
from src.engines.self_healing import generate_calibrated_signals
from src.engines.backtester import Backtester

def test_multi_assets():
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    print("--- AVALIAÇÃO MULTI-ATIVOS CALIBRADA (730 DIAS) ---")
    for sym in symbols:
        df = fetch_historical_data(sym, days=730)
        df_cal = generate_calibrated_signals(df, entry_window=30, exit_window=10)
        bt = Backtester(df_cal, initial_capital=INITIAL_CAPITAL)
        res = bt.run()
        print(f"[{sym}]: Retorno: {res['total_return_pct']:+.2f}% | Cap: ${res['final_capital']:,.2f} | Trades: {res['total_trades']} | WR: {res['win_rate_pct']:.1f}% | PF: {res['profit_factor']:.2f} | MaxDD: -{res['max_drawdown_pct']:.2f}%")

if __name__ == '__main__':
    test_multi_assets()
