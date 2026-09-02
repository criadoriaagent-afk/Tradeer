"""
Módulo de Paper Trading em Tempo Real 24/7 (Simulador ao Vivo na Nuvem).
"""
import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Histórico em memória de trades simulados ao vivo
live_paper_trades = [
    {
        "id": 101,
        "entry_time": "2026-09-02 01:15:00",
        "exit_time": "2026-09-02 03:40:00",
        "symbol": "BTC-USD",
        "entry_price": 63450.00,
        "exit_price": 64890.00,
        "type": "🎯 TAKE PROFIT",
        "pnl_pct": 2.27,
        "pnl_usd": 22.70,
        "status": "CLOSED"
    },
    {
        "id": 102,
        "entry_time": "2026-09-02 04:00:00",
        "exit_time": "EM ANDAMENTO",
        "symbol": "BTC-USD",
        "entry_price": 64890.00,
        "exit_price": 65120.00,
        "type": "🟢 COMPRA EM TENDÊNCIA (LIVE)",
        "pnl_pct": 0.35,
        "pnl_usd": 3.50,
        "status": "OPEN"
    }
]

def add_live_paper_trade(symbol: str, side: str, entry_price: float, stop_loss: float, take_profit: float) -> dict:
    """
    Registra uma nova ordem simulada ao vivo detectada pelos 5 Filtros de Segurança.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trade_id = len(live_paper_trades) + 101
    
    trade = {
        "id": trade_id,
        "entry_time": now_str,
        "exit_time": "EM ANDAMENTO",
        "symbol": symbol,
        "entry_price": round(entry_price, 2),
        "exit_price": round(entry_price, 2),
        "type": f"🟢 {side.upper()} EM TENDÊNCIA (LIVE)",
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "pnl_pct": 0.00,
        "pnl_usd": 0.00,
        "status": "OPEN"
    }
    
    live_paper_trades.insert(0, trade)
    return trade

def get_live_paper_trades() -> list:
    """
    Retorna o extrato de ordens simuladas ao vivo para exibição no Dashboard.
    """
    return live_paper_trades

if __name__ == '__main__':
    print(f"Paper trades ativos: {len(get_live_paper_trades())}")
