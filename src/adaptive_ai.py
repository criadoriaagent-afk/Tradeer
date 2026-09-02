"""
Módulo de IA Adaptativa de Regime de Mercado (Super-Banco).
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import fetch_historical_data
from src.institutional_engine import calculate_adx

def determine_adaptive_mode(df: pd.DataFrame = None, symbol: str = "BTC-USD") -> dict:
    """
    Classifica o regime de mercado e ativa o Sub-Algoritmo Especialista adequado.
    """
    if df is None or df.empty:
        try:
            df = fetch_historical_data(symbol=symbol, days=60)
        except Exception:
            df = pd.DataFrame()
            
    if df.empty or len(df) < 20:
        return {
            "mode": "BULL_ENGINE",
            "mode_label": "🟢 SUB-ALGORITMO 1: BULL ENGINE (Tendência de Alta Rápida)",
            "description": "Ativado: Rompimento de topos de 30 dias + Trailing Stop dinâmico.",
            "hedging_active": False,
            "target_allocation_pct": 100
        }
        
    df = df.copy()
    
    curr_price = df['close'].iloc[-1]
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    ema_200 = df['ema_200'].iloc[-1]
    
    df['adx'] = calculate_adx(df)
    adx_curr = df['adx'].iloc[-1]
    roc_10d = ((curr_price - df['close'].iloc[-11]) / df['close'].iloc[-11]) * 100.0
    
    # Lógica da IA Adaptativa de Regime
    if curr_price >= ema_200 and roc_10d >= 0:
        mode = "BULL_ENGINE"
        mode_label = "🟢 SUB-ALGORITMO 1: BULL ENGINE (Tendência de Alta Rápida)"
        desc = "Mercado em ciclo comprador. Operações de rompimento liberadas com Trailing Stop 2.5x ATR."
        hedging = False
        allocation = 100
    elif curr_price < ema_200 and roc_10d < -2.0:
        mode = "BEAR_ENGINE"
        mode_label = "🛡️ SUB-ALGORITMO 2: BEAR & HEDGING (Proteção Neutra em Queda)"
        desc = "Mercado em ciclo de baixa. Capital 100% protegido em Dólar/USDT. Compras do varejo bloqueadas."
        hedging = True
        allocation = 0
    else:
        mode = "RANGE_ENGINE"
        mode_label = "🟡 SUB-ALGORITMO 3: RANGE & ARBITRAGEM (Mercado Lateral)"
        desc = "Mercado sem tendência limpa. Ativada estratégia de retorno à média em canais de volatilidade."
        hedging = False
        allocation = 50
        
    return {
        "mode": mode,
        "mode_label": mode_label,
        "description": desc,
        "hedging_active": hedging,
        "target_allocation_pct": allocation,
        "adx": round(adx_curr, 1),
        "roc_10d": round(roc_10d, 2)
    }

if __name__ == '__main__':
    ai = determine_adaptive_mode()
    print(ai['mode_label'])
