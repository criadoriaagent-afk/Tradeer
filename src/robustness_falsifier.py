"""
Módulo de Falsificação Científica e Testes de Robustez Quantitativa (Tradeer Quant).
Executa testes de estresse para tentar provar que os resultados do passado são nulos ou overfitted.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INITIAL_CAPITAL
from src.data_loader import fetch_historical_data
from src.strategy import generate_signals, calculate_indicators
from src.backtester import Backtester

def run_local_perturbation_test(symbol: str = "BTC-USD", days: int = 730) -> dict:
    """
    1. Teste de Perturbação Local (Vizinhança do ATR):
    Verifica se pequenas alterações ao redor de 1.25x e 3.00x produzem planalto de estabilidade ou colapsam.
    """
    print("\n========================================================")
    print(" 1. ANÁLISE DE PERTURBAÇÃO LOCAL (Planalto vs. Agulha)")
    print("========================================================")
    
    df = fetch_historical_data(symbol=symbol, days=days)
    neighborhood = [1.15, 1.20, 1.25, 1.30, 1.35, 2.75, 2.85, 3.00, 3.15, 3.25]
    results = {}
    
    for mult in neighborhood:
        df_mod = df.copy()
        df_mod = calculate_indicators(df_mod, entry_window=30, exit_window=10)
        
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
        results[f"{mult:.2f}x ATR"] = res
        
        print(f" Multiplicador {mult:.2f}x ATR -> Retorno: {res['total_return_pct']:+.2f}% | Trades: {res['total_trades']} | PF: {res['profit_factor']:.2f} | Exp: ${res['expectancy_usd']:+.2f} | MaxDD: -{res['max_drawdown_pct']:.2f}%")
        
    return results

def run_1d_vs_4h_comparison(symbol: str = "BTC-USD", days: int = 730) -> dict:
    """
    2. Comparativo Estrito 1D vs 4H (Mesmas Regras e Custos):
    Compara o comportamento do robô entre 1D e 4H com métricas completas.
    """
    print("\n========================================================")
    print(" 2. COMPARATIVO ESTRITO 1D vs 4H (Mesmo Período e Regras)")
    print("========================================================")
    
    # 1D
    df_1d = fetch_historical_data(symbol=symbol, days=days)
    df_1d_sig = generate_signals(df_1d)
    bt_1d = Backtester(df_1d_sig, initial_capital=INITIAL_CAPITAL)
    res_1d = bt_1d.run()
    
    # 4H
    df_raw = yf.download(symbol, period=f"{days}d", interval="1h", progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw = df_raw.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}).dropna()
    df_4h = df_raw.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    
    df_4h_sig = generate_signals(df_4h)
    bt_4h = Backtester(df_4h_sig, initial_capital=INITIAL_CAPITAL)
    res_4h = bt_4h.run()
    
    # CAGR
    years = days / 365.25
    cagr_1d = ((res_1d['final_capital'] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100.0
    cagr_4h = ((res_4h['final_capital'] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100.0
    
    # % Tempo Exposto
    eq_1d = res_1d['equity_df']
    eq_4h = res_4h['equity_df']
    
    # Retorno por Unidade de Risco (Return / Max DD)
    r_risk_1d = res_1d['total_return_pct'] / (res_1d['max_drawdown_pct'] + 1e-5)
    r_risk_4h = res_4h['total_return_pct'] / (res_4h['max_drawdown_pct'] + 1e-5)
    
    comp = {
        "1D": {
            "trades_per_year": round(res_1d['total_trades'] / years, 1),
            "cagr_pct": round(cagr_1d, 2),
            "return_pct": res_1d['total_return_pct'],
            "profit_factor": res_1d['profit_factor'],
            "expectancy_usd": res_1d['expectancy_usd'],
            "max_drawdown_pct": res_1d['max_drawdown_pct'],
            "sharpe_ratio": res_1d['sharpe_ratio'],
            "return_per_unit_risk": round(r_risk_1d, 2)
        },
        "4H": {
            "trades_per_year": round(res_4h['total_trades'] / years, 1),
            "cagr_pct": round(cagr_4h, 2),
            "return_pct": res_4h['total_return_pct'],
            "profit_factor": res_4h['profit_factor'],
            "expectancy_usd": res_4h['expectancy_usd'],
            "max_drawdown_pct": res_4h['max_drawdown_pct'],
            "sharpe_ratio": res_4h['sharpe_ratio'],
            "return_per_unit_risk": round(r_risk_4h, 2)
        }
    }
    
    print(f" [Gráfico 1D]: Trades/Ano: {comp['1D']['trades_per_year']} | CAGR: {comp['1D']['cagr_pct']:+.2f}% | Retorno: {comp['1D']['return_pct']:+.2f}% | PF: {comp['1D']['profit_factor']} | Exp: ${comp['1D']['expectancy_usd']} | MaxDD: -{comp['1D']['max_drawdown_pct']}% | Sharpe: {comp['1D']['sharpe_ratio']}")
    print(f" [Gráfico 4H]: Trades/Ano: {comp['4H']['trades_per_year']} | CAGR: {comp['4H']['cagr_pct']:+.2f}% | Retorno: {comp['4H']['return_pct']:+.2f}% | PF: {comp['4H']['profit_factor']} | Exp: ${comp['4H']['expectancy_usd']} | MaxDD: -{comp['4H']['max_drawdown_pct']}% | Sharpe: {comp['4H']['sharpe_ratio']}")
    
    return comp

def run_trade_by_trade_outlier_inspection(symbol: str = "BTC-USD", days: int = 730):
    """
    3. Inspeção Trade-by-Trade & Detecção de Outliers:
    Verifica se a performance é distorcida por 1 ou 2 trades gigantes desproporcionais.
    """
    print("\n========================================================")
    print(" 3. INSPEÇÃO TRADE-BY-TRADE E DETECÇÃO DE OUTLIERS")
    print("========================================================")
    
    df = fetch_historical_data(symbol=symbol, days=days)
    df_sig = generate_signals(df)
    bt = Backtester(df_sig, initial_capital=INITIAL_CAPITAL)
    res = bt.run()
    trades_df = res['trades_df']
    
    if trades_df.empty:
        print(" Nenhum trade realizado no período.")
        return
        
    print(f" Total de Trades: {len(trades_df)}")
    print(f" Maior Lucro Único: ${trades_df['pnl'].max():+.2f} ({trades_df['pnl_pct'].max():+.2f}%)")
    print(f" Maior Perda Única: ${trades_df['pnl'].min():+.2f} ({trades_df['pnl_pct'].min():+.2f}%)")
    print(f" Média por Trade (Expectancy): ${trades_df['pnl'].mean():+.2f}")
    print(f" Mediana por Trade: ${trades_df['pnl'].median():+.2f}")
    
    # Checagem de Outliers: Se remover o melhor trade, a estratégia continua positiva?
    df_no_top = trades_df.sort_values(by='pnl', ascending=False).iloc[1:]
    return_without_top = df_no_top['pnl'].sum()
    print(f" Lucro Acumulado SEM o melhor trade: ${return_without_top:+.2f}")
    if return_without_top <= 0:
        print(" ALERTA DE OUTLIER: O resultado depende 100% de um único trade atípico!")
    else:
        print(" ROBUSTEZ CONFIRMADA: A estratégia se mantém rentável mesmo sem o melhor trade.")

def run_math_expectancy_audit(symbol: str = "BTC-USD", days: int = 730):
    """
    4. Auditoria da Matemática da Expectancy:
    Explica exatamente a relação entre Expectancy ($), Tamanho de Posição ($) e Retorno Total.
    """
    print("\n========================================================")
    print(" 4. AUDITORIA MATEMÁTICA DA EXPECTANCY E CAPITAL")
    print("========================================================")
    
    df = fetch_historical_data(symbol=symbol, days=days)
    df_sig = generate_signals(df)
    bt = Backtester(df_sig, initial_capital=INITIAL_CAPITAL)
    res = bt.run()
    
    trades = res['trades_df']
    if trades.empty:
        return
        
    total_trades = len(trades)
    sum_pnl = trades['pnl'].sum()
    mean_pnl = trades['pnl'].mean()
    initial_cap = INITIAL_CAPITAL
    final_cap = res['final_capital']
    
    print(f" Capital Inicial: ${initial_cap:,.2f}")
    print(f" Capital Final: ${final_cap:,.2f}")
    print(f" Soma Bruta de PnL dos Trades: ${sum_pnl:+.2f}")
    print(f" Expectancy Real Média por Trade: ${mean_pnl:+.2f}")
    print(f" Fórmula Explicada: Capital Final = Capital Inicial (${initial_cap}) + (Expectancy Média ${mean_pnl:.2f} * {total_trades} Trades)")
    print(f" Verificação da Fórmula: ${initial_cap + (mean_pnl * total_trades):,.2f} == ${final_cap:,.2f}")

def run_common_baseline_multi_asset(days: int = 730):
    """
    5. Baseline Comum Multi-Ativo (Sem Ajustes Prematuros por Moeda):
    Testa uma única regra idêntica em BTC, ETH e SOL.
    """
    print("\n========================================================")
    print(" 5. BASELINE COMUM MULTI-ATIVO (BTC, ETH, SOL SEM OTIMIZAÇÃO)")
    print("========================================================")
    
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    for sym in symbols:
        df = fetch_historical_data(sym, days=days)
        df_sig = generate_signals(df)
        bt = Backtester(df_sig, initial_capital=INITIAL_CAPITAL)
        res = bt.run()
        print(f" [{sym} Baseline Idêntico]: Retorno: {res['total_return_pct']:+.2f}% | Trades: {res['total_trades']} | WR: {res['win_rate_pct']:.1f}% | PF: {res['profit_factor']:.2f} | Exp: ${res['expectancy_usd']:+.2f} | MaxDD: -{res['max_drawdown_pct']:.2f}%")

if __name__ == '__main__':
    run_local_perturbation_test()
    run_1d_vs_4h_comparison()
    run_trade_by_trade_outlier_inspection()
    run_math_expectancy_audit()
    run_common_baseline_multi_asset()
