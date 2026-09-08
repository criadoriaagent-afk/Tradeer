"""
Módulo de Auditoria e Validação da Versão V1.0 Congelada (Tradeer Quant).
Executa testes de 5 anos (2021-2026), remoção de outliers e auditoria matemática financeira sem qualquer otimização.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import INITIAL_CAPITAL
from src.core.data_loader import fetch_historical_data
from src.core.strategy import generate_signals, calculate_indicators
from src.engines.backtester import Backtester

def audit_financial_math(symbol: str = "BTC-USD", days: int = 730):
    """
    1. Auditoria Matemática da Divergência Financeira:
    Explica exatamente a conexão entre o Capital Inicial, o Sizing por trade, os custos de 0.15% e o Retorno Final %.
    """
    print("\n========================================================")
    print(" 1. AUDITORIA DA DIVERGÊNCIA MATEMÁTICA FINANCEIRA")
    print("========================================================")
    
    df = fetch_historical_data(symbol=symbol, days=days)
    df_sig = generate_signals(df)
    bt = Backtester(df_sig, initial_capital=INITIAL_CAPITAL)
    res = bt.run()
    
    trades = res['trades_df']
    if trades.empty:
        print(" Nenhum trade no período.")
        return
        
    initial_cap = INITIAL_CAPITAL
    final_cap = res['final_capital']
    total_pnl_usd = trades['pnl'].sum()
    total_trades = len(trades)
    mean_pnl = trades['pnl'].mean()
    
    # Explicativo matemático
    print(f" Capital Base Total: ${initial_cap:,.2f}")
    print(f" Total de Operações Realizadas: {total_trades}")
    print(f" Soma Líquida em Dólar de Todos os Trades (após 0.15% taxas): ${total_pnl_usd:+.2f}")
    print(f" Expectancy Média por Trade ($): ${mean_pnl:+.2f}")
    print(f" Capital Final Apurado: ${final_cap:,.2f}")
    print(f" Retorno Percentual em Relação ao Capital Inicial: {res['total_return_pct']:+.2f}%")
    print(f"\n [Fórmula Conectada]: Retorno % = (Soma Líquida PnL ${total_pnl_usd:.2f} / Capital Inicial ${initial_cap:.2f}) * 100")
    print(f" [Verificação]: (${total_pnl_usd:.2f} / ${initial_cap:.2f}) * 100 = {((total_pnl_usd / initial_cap) * 100):+.2f}%")

def audit_outlier_dependency(symbol: str = "BTC-USD", days: int = 730):
    """
    2. Análise de Dependência de Outliers:
    Mede a performance da V1.0 sob 4 cenários (Completo, Sem Top 1, Sem Top 3, Sem Top 5).
    """
    print("\n========================================================")
    print(" 2. ANÁLISE DE DEPENDÊNCIA DE OUTLIERS (TOP VENCEDORES)")
    print("========================================================")
    
    df = fetch_historical_data(symbol=symbol, days=days)
    df_sig = generate_signals(df)
    bt = Backtester(df_sig, initial_capital=INITIAL_CAPITAL)
    res = bt.run()
    
    trades = res['trades_df']
    if trades.empty:
        return
        
    sorted_trades = trades.sort_values(by='pnl', ascending=False).reset_index(drop=True)
    total_pnl = sorted_trades['pnl'].sum()
    
    top_1_val = sorted_trades['pnl'].iloc[0]
    top_3_val = sorted_trades['pnl'].iloc[:3].sum()
    top_5_val = sorted_trades['pnl'].iloc[:5].sum()
    
    pnl_no_top1 = total_pnl - top_1_val
    pnl_no_top3 = total_pnl - top_3_val
    pnl_no_top5 = total_pnl - top_5_val
    
    ret_full = res['total_return_pct']
    ret_no_top1 = round((pnl_no_top1 / INITIAL_CAPITAL) * 100, 2)
    ret_no_top3 = round((pnl_no_top3 / INITIAL_CAPITAL) * 100, 2)
    ret_no_top5 = round((pnl_no_top5 / INITIAL_CAPITAL) * 100, 2)
    
    pct_top1 = (top_1_val / total_pnl * 100) if total_pnl > 0 else 0.0
    pct_top3 = (top_3_val / total_pnl * 100) if total_pnl > 0 else 0.0
    
    print(f" Retorno Total Completo (15 Trades): {ret_full:+.2f}% (${total_pnl:+.2f})")
    print(f" Top 1 Melhor Trade: ${top_1_val:+.2f} (Representa {pct_top1:.1f}% do lucro total)")
    print(f" Retorno SEM o 1º Melhor Trade: {ret_no_top1:+.2f}% (${pnl_no_top1:+.2f})")
    print(f" Retorno SEM os 3 Melhores Trades: {ret_no_top3:+.2f}% (${pnl_no_top3:+.2f})")
    print(f" Retorno SEM os 5 Melhores Trades: {ret_no_top5:+.2f}% (${pnl_no_top5:+.2f})")
    print(f" Média por Trade: ${sorted_trades['pnl'].mean():+.2f} | Mediana: ${sorted_trades['pnl'].median():+.2f}")
    
    if ret_no_top1 <= 0:
        print(" [ALERTA DE FRAGILIDADE DE OUTLIER]: A V1.0 no período de 730d depende 100% do maior trade isolado para fechar no positivo.")

def run_long_5year_backtest(symbol: str = "BTC-USD", days: int = 1825):
    """
    3. Backtest de Longo Prazo (5 Anos: 2021 - 2026 / 1.825 Dias):
    Mede a performance da V1.0 no maior histórico confiável passando por bull markets, crashs e consolidacoes.
    """
    print("\n========================================================")
    print(" 3. BACKTEST DE LONGO PRAZO - 5 ANOS (2021 A 2026 - 1.825 DIAS)")
    print("========================================================")
    
    df = fetch_historical_data(symbol=symbol, days=days)
    if df.empty:
        print(" Falha ao carregar 5 anos de dados.")
        return
        
    df_sig = generate_signals(df)
    bt = Backtester(df_sig, initial_capital=INITIAL_CAPITAL)
    res = bt.run()
    
    trades = res['trades_df']
    eq = res['equity_df']
    
    years = days / 365.25
    cagr = ((res['final_capital'] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100.0
    
    # Expectancy em R (múltiplo do risco médio)
    avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if not trades[trades['pnl'] < 0].empty else 1.0
    expectancy_r = round(res['expectancy_usd'] / avg_loss, 2) if avg_loss > 0 else 0.0
    
    # Pior Sequência de Perdas Consecutivas
    pnl_series = trades['pnl'] > 0
    loss_streaks = (~pnl_series).astype(int).groupby((pnl_series).cumsum()).sum()
    worst_loss_streak = int(loss_streaks.max()) if not loss_streaks.empty else 0
    
    # % Tempo Exposto no Mercado
    in_position_count = len(trades) * 15 # Estimativa de dias posicionados
    exposure_pct = round(min(100.0, (in_position_count / days) * 100.0), 1)
    
    # Sortino Ratio (apenas volatilidade negativa)
    daily_ret = eq['equity'].pct_change().dropna()
    downside_std = daily_ret[daily_ret < 0].std()
    sortino_ratio = round((daily_ret.mean() / downside_std) * np.sqrt(365), 2) if downside_std > 0 else 0.0
    
    # Retorno por Unidade de Risco
    r_risk = round(res['total_return_pct'] / (res['max_drawdown_pct'] + 1e-5), 2)
    
    print(f" [V1.0 BTC-USD 5 ANOS]:")
    print(f"   Período Avaliado: {days} dias ({years:.1f} anos) | Capital Inicial: ${INITIAL_CAPITAL:,.2f}")
    print(f"   Capital Final: ${res['final_capital']:,.2f} | Retorno Acumulado: {res['total_return_pct']:+.2f}% | CAGR: {cagr:+.2f}%/ano")
    print(f"   Total de Trades: {res['total_trades']} (Média {res['total_trades']/years:.1f} trades/ano)")
    print(f"   Win Rate: {res['win_rate_pct']:.1f}% | Profit Factor: {res['profit_factor']:.2f}")
    print(f"   Expectancy por Trade: ${res['expectancy_usd']:+.2f} ({expectancy_r:+.2f} R)")
    print(f"   Max Drawdown: -{res['max_drawdown_pct']:.2f}% | Duração Máx. Drawdown: {res['max_dd_duration_days']} dias")
    print(f"   Pior Sequência de Perdas Consecutivas: {worst_loss_streak} derrotas seguidas")
    print(f"   Sharpe Ratio: {res['sharpe_ratio']:.2f} | Sortino Ratio: {sortino_ratio:.2f}")
    print(f"   Tempo Exposto no Mercado: ~{exposure_pct}% | Retorno por Unidade de Risco: {r_risk:.2f}")

def run_multi_asset_v1_baseline(days: int = 1825):
    """
    4. Avaliação Baseline V1.0 nos 3 Ativos (BTC, ETH, SOL) em 5 Anos:
    Verifica se a V1.0 possui robustez multi-ativo sem qualquer otimização específica.
    """
    print("\n========================================================")
    print(" 4. AVALIAÇÃO BASELINE V1.0 NOS 3 ATIVOS (5 ANOS - SEM TUNING)")
    print("========================================================")
    
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    for sym in symbols:
        df = fetch_historical_data(sym, days=days)
        if df.empty:
            continue
        df_sig = generate_signals(df)
        bt = Backtester(df_sig, initial_capital=INITIAL_CAPITAL)
        res = bt.run()
        years = days / 365.25
        cagr = ((res['final_capital'] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100.0
        print(f" [{sym} 5 Anos]: Retorno: {res['total_return_pct']:+.2f}% | CAGR: {cagr:+.2f}%/ano | Trades: {res['total_trades']} | WR: {res['win_rate_pct']:.1f}% | PF: {res['profit_factor']:.2f} | Exp: ${res['expectancy_usd']:+.2f} | MaxDD: -{res['max_drawdown_pct']:.2f}% | Sharpe: {res['sharpe_ratio']:.2f}")

def record_4h_generalization_failure():
    """
    5. Registro Científico da Não-Generalização no 4H:
    Documenta formalmente a falha de generalização da V1.0 no gráfico intraday de 4H.
    """
    print("\n========================================================")
    print(" 5. REGISTRO CIENTÍFICO DE NÃO-GENERALIZAÇÃO NO 4H")
    print("========================================================")
    print(" [EVIDÊNCIA CONGELADA]: A versão V1.0 apresentou performance de -29.43% com Profit Factor de 0.47 no gráfico de 4H (730d).")
    print(" [REGISTRO FORMAL]: A V1.0 NÃO generaliza para o gráfico intraday de 4H. Nenhuma modificação artificial será feita para 'consertar' o resultado de 4H.")

if __name__ == '__main__':
    audit_financial_math()
    audit_outlier_dependency()
    run_long_5year_backtest()
    run_multi_asset_v1_baseline()
    record_4h_generalization_failure()
