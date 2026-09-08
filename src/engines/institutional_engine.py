"""
Módulo Institucional (Filtro de Volume Bancário & Classificador de Regime de Mercado).
"""
import pandas as pd
import numpy as np

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcula o ADX (Average Directional Index) para determinar a FORÇA da tendência.
    ADX > 25 = Tendência Forte Institucional
    ADX < 20 = Mercado Lateral / Sem Direção (Ruído)
    """
    df = df.copy()
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    plus_di = 100 * (pd.Series(df['plus_dm']).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (pd.Series(df['minus_dm']).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx

def analyze_institutional_flow(df: pd.DataFrame) -> dict:
    """
    Analisa os dados mais recentes do ativo sob a ótica dos grandes fundos e bancos.
    """
    if df.empty or len(df) < 30:
        return {
            "market_regime": "ANALISANDO DADOS",
            "is_trend_strong": False,
            "volume_surge": False,
            "volume_ratio": 1.0,
            "adx": 0.0,
            "institutional_status": "MODO DE ESPERA (Aguardando Confirmações)"
        }
        
    df = df.copy()
    
    # 1. Análise de Volume Bancário
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    current_vol = df['volume'].iloc[-1]
    avg_vol = df['vol_ma_20'].iloc[-1]
    
    volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    volume_surge = volume_ratio >= 1.25 # Volume 25%+ maior que a média das últimas 20 velas
    
    # 2. Análise de Regime de Mercado via ADX
    df['adx'] = calculate_adx(df)
    current_adx = df['adx'].iloc[-1]
    
    if current_adx >= 25.0:
        market_regime = "TENDÊNCIA FORTE INSTITUCIONAL"
        is_trend_strong = True
    elif current_adx <= 18.0:
        market_regime = "MERCADO LATERAL (Ruído / Sem Direção)"
        is_trend_strong = False
    else:
        market_regime = "MERCADO EM TRANSIÇÃO"
        is_trend_strong = False
        
    # Status Consolidado em Português Claro
    if is_trend_strong and volume_surge:
        status = "🟢 FLUXO BANCÁRIO DETECTADO: Entrada Institucional Confirmada"
    elif is_trend_strong:
        status = "🔵 TENDÊNCIA CONFIRMADA: Aguardando Volume de Pico"
    else:
        status = "🛡️ MODO PROTEÇÃO: Mercado sem direção limpa (Operações Bloqueadas)"
        
    return {
        "market_regime": market_regime,
        "is_trend_strong": is_trend_strong,
        "volume_surge": volume_surge,
        "volume_ratio": round(volume_ratio, 2),
        "adx": round(current_adx, 1),
        "institutional_status": status
    }
