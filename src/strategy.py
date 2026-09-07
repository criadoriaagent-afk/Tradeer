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

def calculate_indicators(df: pd.DataFrame, entry_window: int = 30, exit_window: int = 10) -> pd.DataFrame:
    """
    Calcula os indicadores para a Estratégia Donchian Breakout com Zero Look-Ahead Bias.
    Todos os indicadores são deslocados por .shift(1) para refletir estritamente velas fechadas.
    """
    df = df.copy()
    
    # Canais de Donchian ([1:31] - 30 velas anteriores excluindo o candle atual)
    df['donchian_high'] = df['high'].shift(1).rolling(window=entry_window).max()
    df['donchian_low'] = df['low'].shift(1).rolling(window=exit_window).min()
    
    # EMA 200 (Tendência Macro na vela fechada anterior)
    df['ema_200'] = df['close'].shift(1).ewm(span=200, adjust=False).mean()
    
    # ATR (Volatilidade para Stop Loss na vela fechada anterior)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.shift(1).rolling(window=14).mean()
    
    return df

def generate_signals(df: pd.DataFrame, entry_window: int = 30, exit_window: int = 10) -> pd.DataFrame:
    """
    Sinal 1 (Compra): Fechamento do candle i > Donchian High (30d fechados) E > EMA 200.
    Sinal -1 (Venda): Fechamento do candle i < Donchian Low (10d fechados).
    Ordem executada estritamente na Abertura do candle i+1.
    """
    df = calculate_indicators(df, entry_window=entry_window, exit_window=exit_window)
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
