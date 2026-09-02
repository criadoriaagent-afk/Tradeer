"""
Módulo de Teste de Estresse Estatístico por Simulação de Monte Carlo.
"""
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INITIAL_CAPITAL

def run_monte_carlo_simulation(trade_returns: list = [5.2, -2.1, 8.4, -1.8, 12.1, -2.5, 6.7, -1.9, 10.5], 
                                num_simulations: int = 1000, 
                                num_trades_per_sim: int = 100) -> dict:
    """
    Executa 1.000 simulações de estresse de Monte Carlo reordenando retornos aleatoriamente.
    """
    if not trade_returns:
        trade_returns = [4.5, -2.0, 7.8, -1.5, 9.2, -2.1, 5.5, -1.8]
        
    returns_arr = np.array(trade_returns) / 100.0
    
    simulated_final_capitals = []
    simulated_max_drawdowns = []
    ruin_count = 0
    
    for _ in range(num_simulations):
        # Amostragem aleatória com reposição (Bootstrap)
        simulated_returns = np.random.choice(returns_arr, size=num_trades_per_sim, replace=True)
        
        capital = INITIAL_CAPITAL
        equity_series = [capital]
        
        for ret in simulated_returns:
            capital *= (1.0 + ret)
            equity_series.append(capital)
            
        equity_series = np.array(equity_series)
        peaks = np.maximum.accumulate(equity_series)
        drawdowns = (equity_series - peaks) / peaks
        max_dd = abs(np.min(drawdowns)) * 100.0
        
        simulated_final_capitals.append(capital)
        simulated_max_drawdowns.append(max_dd)
        
        # Considera ruína se a perda ultrapassar 30% do capital total
        if max_dd >= 30.0:
            ruin_count += 1
            
    probability_of_ruin = (ruin_count / num_simulations) * 100.0
    avg_final_capital = np.mean(simulated_final_capitals)
    worst_case_drawdown = np.max(simulated_max_drawdowns)
    median_drawdown = np.median(simulated_max_drawdowns)
    
    # Avaliação Final em Português Claro
    if probability_of_ruin == 0.0 and worst_case_drawdown < 15.0:
        approval_status = "✅ APROVADO COM EXCELÊNCIA (0.0% Risco de Ruína)"
    elif probability_of_ruin < 2.0:
        approval_status = "🟡 APROVADO COM RESSALVAS (Risco Baixo)"
    else:
        approval_status = "❌ REPROVADO (Ajustar Gerenciador de Risco)"
        
    return {
        "num_simulations": num_simulations,
        "probability_of_ruin": round(probability_of_ruin, 2),
        "worst_case_drawdown": round(worst_case_drawdown, 2),
        "median_drawdown": round(median_drawdown, 2),
        "avg_final_capital": round(avg_final_capital, 2),
        "approval_status": approval_status
    }

if __name__ == '__main__':
    res = run_monte_carlo_simulation()
    print("="*60)
    print("      RELATÓRIO DE ESTRESSE DE MONTE CARLO (1.000 SIMULAÇÕES)      ")
    print("="*60)
    print(f" Status de Aprovação        : {res['approval_status']}")
    print(f" Probabilidade de Ruína     : {res['probability_of_ruin']}%")
    print(f" Maior Drawdown Simulado    : -{res['worst_case_drawdown']}%")
    print(f" Drawdown Mediano           : -{res['median_drawdown']}%")
    print(f" Capital Médio Estimado     : ${res['avg_final_capital']:,.2f} USD")
    print("="*60)
