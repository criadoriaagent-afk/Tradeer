"""
Módulo da Estratégia Quantitativa (Retorno à Média: Bollinger + RSI + ATR).
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    BOLLINGER_PERIOD, BOLLINGER_STD,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT
)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os indicadores para a Estratégia Donchian Breakout (Turtle Trading).
    """
    df = df.copy()
    
    # Canais de Donchian (Máxima dos últimos 20 dias / Mínima dos últimos 10 dias)
    df['donchian_high'] = df['high'].shift(1).rolling(window=20).max()
    df['donchian_low'] = df['low'].shift(1).rolling(window=10).min()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # ATR (Volatilidade para Stop Loss)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sinal 1 (Compra): Fechamento rompe a máxima dos últimos 20 dias E preço > EMA 200.
    Sinal -1 (Venda): Fechamento perde a mínima dos últimos 10 dias.
    """
    df = calculate_indicators(df)
    df['signal'] = 0
    
    buy_condition = (df['close'] > df['donchian_high']) & (df['close'] > df['ema_200'])
    sell_condition = (df['close'] < df['donchian_low'])
    
    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1
    
    return df

if __name__ == '__main__':
    from src.data_loader import fetch_historical_data
    df = fetch_historical_data()
    df_signals = generate_signals(df)
    print(df_signals[['close', 'lower_band', 'upper_band', 'rsi', 'signal']].tail(20))
