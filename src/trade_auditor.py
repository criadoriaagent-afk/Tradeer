"""
Módulo de Auditoria Transparente de Trades (Integrado com Auto-Diagnóstico Self-Healing).
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INITIAL_CAPITAL, SYMBOL, DAYS_BACK
from src.self_healing import run_self_healing_diagnosis

def get_audited_trade_history(symbol: str = SYMBOL, days: int = DAYS_BACK) -> dict:
    """
    Executa a auditoria autônoma recalibrada pelo motor de Self-Healing.
    """
    diag = run_self_healing_diagnosis(symbol=symbol, days=days)
    metrics = diag['metrics']
    
    return {
        "symbol": symbol,
        "days_evaluated": days,
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": metrics['final_capital'],
        "total_return_pct": metrics['total_return_pct'],
        "win_rate_pct": metrics['win_rate_pct'],
        "profit_factor": metrics['profit_factor'],
        "max_drawdown_pct": metrics['max_drawdown_pct'],
        "total_trades": metrics['total_trades'],
        "self_healing_status": diag['status'],
        "trades": diag['trades'],
        "chart_points": diag['chart_points']
    }

if __name__ == '__main__':
    audit = get_audited_trade_history()
    print(f"Status: {audit['self_healing_status']}")
    print(f"Retorno Final: {audit['total_return_pct']}% | Capital: ${audit['final_capital']}")
