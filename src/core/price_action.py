"""
Módulo de Reconhecimento de Padrões de Price Action Avançado.
"""
import pandas as pd
import numpy as np

def analyze_price_action(df: pd.DataFrame) -> dict:
    """
    Analisa a estrutura dos candles recentes para identificar padrões de Price Action.
    """
    if df.empty or len(df) < 5:
        return {
            "pattern_name": "SEM PADRÃO CLARO",
            "is_bullish_pattern": False,
            "score": 50,
            "description": "Aguardando formação de padrão de Price Action"
        }
        
    df = df.copy()
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    open_c, close_c, high_c, low_c = curr['open'], curr['close'], curr['high'], curr['low']
    body_size = abs(close_c - open_c)
    total_range = high_c - low_c if (high_c - low_c) > 0 else 1.0
    lower_shadow = min(open_c, close_c) - low_c
    upper_shadow = high_c - max(open_c, close_c)
    
    # 1. Padrão Pin Bar (Martelo de Rejeição de Queda)
    is_pinbar = (lower_shadow >= 2.0 * body_size) and (upper_shadow <= 0.5 * body_size)
    
    # 2. Padrão Engolfo de Alta (Bullish Engulfing)
    prev_open, prev_close = prev['open'], prev['close']
    is_engulfing = (prev_close < prev_open) and (close_c > open_c) and (close_c >= prev_open) and (open_c <= prev_close)
    
    # 3. Quebra de Estrutura de Alta (BOS - Break of Structure)
    last_high_max = df['high'].iloc[-5:-1].max()
    is_bos = close_c > last_high_max
    
    if is_pinbar:
        pattern = "PIN BAR DE ALTA (Rejeição de Fundo)"
        score = 90
        desc = "Vela Martelo com longa sombra inferior indica forte entrada de compradores."
        is_bullish = True
    elif is_engulfing:
        pattern = "ENGOLFO DE ALTA (Engolimento Comprador)"
        score = 85
        desc = "Vela de força compradora engoliu o corpo da vela anterior."
        is_bullish = True
    elif is_bos:
        pattern = "ROMPIMENTO DE ESTRUTURA (BOS)"
        score = 80
        desc = "Preço fechou acima da máxima dos últimos 5 candles."
        is_bullish = True
    elif close_c > open_c:
        pattern = "VELA COMPRADORA PADRÃO"
        score = 65
        desc = "Fechamento positivo sem padrão extremo."
        is_bullish = True
    else:
        pattern = "CONSOLIDADO / VELA VENDEDORA"
        score = 40
        desc = "Preço sem sinal comprador no candle atual."
        is_bullish = False
        
    return {
        "pattern_name": pattern,
        "is_bullish_pattern": is_bullish,
        "score": score,
        "description": desc
    }
