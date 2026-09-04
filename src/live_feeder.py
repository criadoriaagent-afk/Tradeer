"""
Módulo de Conexão de Mercado ao Vivo (Live Market Feeder).
"""
import pandas as pd
import yfinance as yf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.institutional_engine import analyze_institutional_flow
from src.strategy import generate_signals

class LiveMarketFeeder:
    def __init__(self, symbols=["BTC-USD", "ETH-USD", "SOL-USD"]):
        self.symbols = symbols
        self.cached_dfs = {}
        
    def fetch_symbol_dataframes(self, symbol: str = "BTC-USD") -> tuple:
        """
        Baixa os DataFrames de 1D e 4H para o ativo selecionado.
        """
        try:
            df_1d = yf.download(symbol, period="60d", interval="1d", progress=False)
            if isinstance(df_1d.columns, pd.MultiIndex):
                df_1d.columns = df_1d.columns.get_level_values(0)
            df_1d = df_1d.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }).dropna()
            
            # 4H data
            df_4h = yf.download(symbol, period="14d", interval="1h", progress=False)
            if isinstance(df_4h.columns, pd.MultiIndex):
                df_4h.columns = df_4h.columns.get_level_values(0)
            df_4h = df_4h.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }).dropna()
            
            return df_1d, df_4h
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    def get_live_market_summary(self) -> dict:
        """
        Consulta os servidores de dados em tempo real e retorna o relatório dos grandes bancos.
        """
        summary = {}
        for symbol in self.symbols:
            try:
                df, df_4h = self.fetch_symbol_dataframes(symbol)
                if df.empty:
                    raise ValueError("DataFrame vazio obtido do Yahoo Finance")

                self.cached_dfs[symbol] = {"1d": df, "4h": df_4h}

                df_signals = generate_signals(df)
                inst_analysis = analyze_institutional_flow(df_signals)
                
                current_price = float(df['close'].iloc[-1])
                prev_price = float(df['close'].iloc[-2]) if len(df) >= 2 else current_price
                change_24h = ((current_price - prev_price) / prev_price) * 100.0 if prev_price > 0 else 0.0
                
                summary[symbol] = {
                    "price": round(current_price, 2),
                    "change_24h": round(change_24h, 2),
                    "institutional": inst_analysis,
                    "signal": int(df_signals['signal'].iloc[-1]),
                    "df_1d": df,
                    "df_4h": df_4h
                }
            except Exception as e:
                summary[symbol] = {
                    "error": str(e),
                    "price": 0.0,
                    "change_24h": 0.0,
                    "institutional": analyze_institutional_flow(pd.DataFrame()),
                    "signal": 0,
                    "df_1d": pd.DataFrame(),
                    "df_4h": pd.DataFrame()
                }
        return summary

if __name__ == '__main__':
    feeder = LiveMarketFeeder()
    data = feeder.get_live_market_summary()
    print(data)
