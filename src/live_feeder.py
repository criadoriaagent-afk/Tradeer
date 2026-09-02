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
        
    def get_live_market_summary(self) -> dict:
        """
        Consulta os servidores de dados em tempo real e retorna o relatório dos grandes bancos.
        """
        summary = {}
        for symbol in self.symbols:
            try:
                # Baixa os últimos 60 dias para cálculo dos indicadores ao vivo
                df = yf.download(symbol, period="60d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                }).dropna()
                
                df_signals = generate_signals(df)
                inst_analysis = analyze_institutional_flow(df_signals)
                
                current_price = df['close'].iloc[-1]
                prev_price = df['close'].iloc[-2]
                change_24h = ((current_price - prev_price) / prev_price) * 100.0
                
                summary[symbol] = {
                    "price": round(current_price, 2),
                    "change_24h": round(change_24h, 2),
                    "institutional": inst_analysis,
                    "signal": int(df_signals['signal'].iloc[-1])
                }
            except Exception as e:
                summary[symbol] = {
                    "error": str(e),
                    "price": 0.0,
                    "change_24h": 0.0,
                    "institutional": analyze_institutional_flow(pd.DataFrame()),
                    "signal": 0
                }
        return summary

if __name__ == '__main__':
    feeder = LiveMarketFeeder()
    data = feeder.get_live_market_summary()
    print(data)
