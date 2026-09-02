"""
Módulo de Carregamento e Preparação de Dados Históricos de Mercado.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

# Adiciona o diretório raiz ao path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SYMBOL, TIMEFRAME, DAYS_BACK

def fetch_historical_data(symbol: str = SYMBOL, timeframe: str = TIMEFRAME, days: int = DAYS_BACK) -> pd.DataFrame:
    """
    Baixa dados históricos OHLCV usando yfinance com tratamento de erros.
    """
    print(f"[DataLoader] Baixando dados históricos para {symbol} ({days} dias, timeframe {timeframe})...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Faz download via yfinance
    df = yf.download(
        tickers=symbol,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval=timeframe,
        progress=False
    )
    
    if df.empty:
        raise ValueError(f"Não foi possível obter dados para {symbol}. Verifique sua conexão ou o código do ativo.")
    
    # Trata caso yfinance retorne MultiIndex nas colunas
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    print(f"[DataLoader] Sucesso: {len(df)} velas carregadas.")
    return df

if __name__ == '__main__':
    data = fetch_historical_data()
    print(data.tail())
