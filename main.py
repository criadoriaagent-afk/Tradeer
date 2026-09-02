"""
Script Principal de Execução do Sistema Quantitativo de Trading & Portfólio (Fase 2).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

from src.config import INITIAL_CAPITAL, SYMBOL, TIMEFRAME, DAYS_BACK
from src.data_loader import fetch_historical_data
from src.strategy import generate_signals
from src.backtester import Backtester
from src.optimizer import run_grid_search
from src.portfolio_backtester import PortfolioBacktester

def print_banner():
    print("=" * 65)
    print("   SISTEMA QUANTITATIVO DE TRADING & PORTFÓLIO MULTI-ATIVOS (FASE 2)   ")
    print("=" * 65)

def run_single_asset_backtest():
    print(f"\n---> [1/3] EXECUTANDO BACKTEST ÚNICO ({SYMBOL}) COM TRAILING STOP...")
    df = fetch_historical_data(symbol=SYMBOL, timeframe=TIMEFRAME, days=DAYS_BACK)
    df_signals = generate_signals(df)
    
    backtester = Backtester(df_signals, initial_capital=INITIAL_CAPITAL)
    metrics = backtester.run()
    
    print("\n" + "-" * 50)
    print(f"       RELATÓRIO INDIVIDUAL ({SYMBOL}) - COM TRAILING STOP       ")
    print("-" * 50)
    print(f" Capital Inicial         : ${INITIAL_CAPITAL:,.2f} USD")
    print(f" Capital Final           : ${metrics['final_capital']:,.2f} USD")
    print(f" Retorno Total           : {metrics['total_return_pct']:+.2f}%")
    print(f" Total de Operações      : {metrics['total_trades']}")
    print(f" Taxa de Acerto (WinRate): {metrics['win_rate_pct']:.2f}%")
    print(f" Fator de Lucro          : {metrics['profit_factor']:.2f}")
    print(f" Maior Queda (Max DD)    : -{metrics['max_drawdown_pct']:.2f}%")
    print("-" * 50)
    return df_signals, metrics

def run_parameter_optimization(df_signals: pd.DataFrame):
    print(f"\n---> [2/3] EXECUTANDO OTIMIZADOR DE PARÂMETROS (GRID SEARCH)...")
    results_df = run_grid_search(df_signals)
    
    print("\n" + "-" * 65)
    print("          TOP 5 MELHORES COMBINAÇÕES DE PARÂMETROS ENCONTRADAS         ")
    print("-" * 65)
    if not results_df.empty:
        top5 = results_df.head(5)
        for idx, row in top5.iterrows():
            print(f" #{idx+1} | Donchian Entrada: {int(row['entry_window'])}d | Saída: {int(row['exit_window'])}d | EMA Trend: {int(row['ema_trend'])}")
            print(f"      Lucro: {row['total_return_pct']:+.2f}% | WinRate: {row['win_rate_pct']:.1f}% | ProfitFactor: {row['profit_factor']:.2f} | MaxDD: -{row['max_drawdown_pct']:.2f}%\n")
    else:
        print(" Nenhuma combinação relevante encontrada.")
    print("-" * 65)

def run_portfolio_simulation():
    print(f"\n---> [3/3] EXECUTANDO BACKTEST DE PORTFÓLIO MULTI-ATIVOS (BTC, ETH, SOL)...")
    portfolio_tester = PortfolioBacktester(symbols=["BTC-USD", "ETH-USD", "SOL-USD"], initial_capital=INITIAL_CAPITAL)
    metrics = portfolio_tester.run()
    
    print("\n" + "=" * 50)
    print("      RELATÓRIO FINAL DO PORTFÓLIO DIVERSIFICADO      ")
    print("=" * 50)
    print(f" Capital Inicial Total   : ${INITIAL_CAPITAL:,.2f} USD")
    print(f" Capital Final Total     : ${metrics['final_capital']:,.2f} USD")
    print(f" Retorno Total da Carteira: {metrics['total_return_pct']:+.2f}%")
    print(f" Total de Trades         : {metrics['total_trades']}")
    print(f" WinRate Global          : {metrics['win_rate_pct']:.2f}%")
    print(f" Fator de Lucro Global   : {metrics['profit_factor']:.2f}")
    print(f" Maior Queda do Portfólio: -{metrics['max_drawdown_pct']:.2f}%")
    print("=" * 50)
    
    portfolio_tester.plot_portfolio(metrics)

def main():
    print_banner()
    
    # 1. Backtest Individual com Trailing Stop
    df_signals, single_metrics = run_single_asset_backtest()
    
    # 2. Otimizador de Parâmetros
    run_parameter_optimization(df_signals)
    
    # 3. Portfólio Multi-Ativos
    run_portfolio_simulation()

if __name__ == '__main__':
    main()
