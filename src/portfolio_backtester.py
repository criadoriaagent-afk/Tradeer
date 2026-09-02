"""
Motor de Backtest de Portfólio Multi-Ativos (Diversificação).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INITIAL_CAPITAL, TIMEFRAME, DAYS_BACK
from src.data_loader import fetch_historical_data
from src.strategy import generate_signals
from src.backtester import Backtester

class PortfolioBacktester:
    def __init__(self, symbols: list = ["BTC-USD", "ETH-USD", "SOL-USD"], initial_capital: float = INITIAL_CAPITAL):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.capital_per_asset = initial_capital / len(symbols)
        
    def run(self) -> dict:
        print(f"\n[Portfólio] Iniciando Backtest de Carteira Multi-Ativos: {self.symbols}")
        print(f"[Portfólio] Capital por ativo: ${self.capital_per_asset:,.2f} USD")
        
        asset_results = {}
        all_trades = []
        combined_equity = None
        
        for symbol in self.symbols:
            try:
                df = fetch_historical_data(symbol=symbol, timeframe=TIMEFRAME, days=DAYS_BACK)
                df_signals = generate_signals(df)
                
                backtester = Backtester(df_signals, initial_capital=self.capital_per_asset)
                metrics = backtester.run()
                
                asset_results[symbol] = metrics
                
                # Coleta trades com identificação do ativo
                trades = metrics['trades_df']
                if not trades.empty:
                    trades['symbol'] = symbol
                    all_trades.append(trades)
                    
                # Combina curvas de patrimônio
                eq = metrics['equity_df']['equity']
                if combined_equity is None:
                    combined_equity = pd.DataFrame({symbol: eq})
                else:
                    combined_equity[symbol] = eq
                    
            except Exception as e:
                print(f"[Erro] Falha ao processar {symbol}: {e}")
                
        if combined_equity is None or combined_equity.empty:
            raise ValueError("Não foi possível gerar backtest de portfólio.")
            
        # Preenche falhas e calcula a curva total somando o patrimônio dos ativos
        combined_equity = combined_equity.ffill().bfill()
        combined_equity['total_portfolio'] = combined_equity.sum(axis=1)
        
        # Consolidação de Trades
        df_all_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        
        # Métricas Globais da Carteira
        final_portfolio_capital = combined_equity['total_portfolio'].iloc[-1]
        portfolio_return_pct = ((final_portfolio_capital - self.initial_capital) / self.initial_capital) * 100.0
        
        # Drawdown do Portfólio
        combined_equity['peak'] = combined_equity['total_portfolio'].cummax()
        combined_equity['drawdown'] = (combined_equity['total_portfolio'] - combined_equity['peak']) / combined_equity['peak']
        max_portfolio_dd = abs(combined_equity['drawdown'].min()) * 100.0
        
        # Taxa de acerto global
        if not df_all_trades.empty:
            wins = df_all_trades[df_all_trades['pnl'] > 0]
            win_rate = (len(wins) / len(df_all_trades)) * 100.0
            total_gain = wins['pnl'].sum()
            total_loss = abs(df_all_trades[df_all_trades['pnl'] < 0]['pnl'].sum())
            profit_factor = total_gain / total_loss if total_loss > 0 else np.inf
        else:
            win_rate, profit_factor = 0.0, 0.0
            
        return {
            'final_capital': round(final_portfolio_capital, 2),
            'total_return_pct': round(portfolio_return_pct, 2),
            'max_drawdown_pct': round(max_portfolio_dd, 2),
            'total_trades': len(df_all_trades),
            'win_rate_pct': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'asset_results': asset_results,
            'combined_equity': combined_equity,
            'all_trades': df_all_trades
        }

    def plot_portfolio(self, results: dict, filename="portfolio_equity.png"):
        """
        Gera gráfico comparativo de patrimônio por ativo e total do portfólio.
        """
        plt.figure(figsize=(12, 6))
        combined_equity = results['combined_equity']
        
        for symbol in self.symbols:
            if symbol in combined_equity.columns:
                plt.plot(combined_equity.index, combined_equity[symbol], label=f'{symbol}', alpha=0.6, linestyle='--')
                
        plt.plot(combined_equity.index, combined_equity['total_portfolio'], label='PORTFÓLIO TOTAL DIVERSIFICADO', color='green', linewidth=2.5)
        plt.axhline(self.initial_capital, color='black', linestyle=':', label='Capital Inicial')
        
        plt.title(f'Evolução do Patrimônio do Portfólio ({", ".join(self.symbols)})')
        plt.xlabel('Data')
        plt.ylabel('Capital Total ($ USD)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(filename, dpi=150)
        print(f"\n[Gráfico] Gráfico de portfólio salvo em: {os.path.abspath(filename)}")

if __name__ == '__main__':
    p_tester = PortfolioBacktester()
    res = p_tester.run()
    print(f"Retorno Portfólio: {res['total_return_pct']}% | Max DD: {res['max_drawdown_pct']}%")
    p_tester.plot_portfolio(res)
