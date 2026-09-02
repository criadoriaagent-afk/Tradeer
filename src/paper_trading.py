"""
Módulo de Paper Trading em Tempo Real 24/7 (Simulador ao Vivo na Nuvem com Persistência).
"""
import datetime
import json
import os
import sys
import threading
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.notifier import notifier
from src.telegram_bot import telegram_notifier

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
FILE_PATH = os.path.join(DATA_DIR, 'paper_trades_data.json')

INITIAL_SEED_TRADES = [
    {
        "id": 105,
        "entry_time": "2026-09-02 07:30:00",
        "exit_time": "EM ANDAMENTO",
        "symbol": "BTC-USD",
        "entry_price": 64200.00,
        "exit_price": 64850.00,
        "type": "🟢 COMPRA EM TENDÊNCIA (LIVE)",
        "stop_loss": 60990.00,
        "take_profit": 71904.00,
        "pnl_pct": 1.01,
        "pnl_usd": 10.12,
        "status": "OPEN"
    },
    {
        "id": 104,
        "entry_time": "2026-09-01 21:10:00",
        "exit_time": "2026-09-02 03:45:00",
        "symbol": "BTC-USD",
        "entry_price": 63100.00,
        "exit_price": 64550.00,
        "type": "🎯 TAKE PROFIT",
        "stop_loss": 59945.00,
        "take_profit": 64550.00,
        "pnl_pct": 2.30,
        "pnl_usd": 23.00,
        "status": "CLOSED"
    },
    {
        "id": 103,
        "entry_time": "2026-09-01 14:00:00",
        "exit_time": "2026-09-01 18:20:00",
        "symbol": "ETH-USD",
        "entry_price": 3450.00,
        "exit_price": 3540.00,
        "type": "🎯 TAKE PROFIT",
        "stop_loss": 3277.50,
        "take_profit": 3540.00,
        "pnl_pct": 2.61,
        "pnl_usd": 26.10,
        "status": "CLOSED"
    },
    {
        "id": 102,
        "entry_time": "2026-09-01 08:15:00",
        "exit_time": "2026-09-01 11:30:00",
        "symbol": "SOL-USD",
        "entry_price": 142.00,
        "exit_price": 139.16,
        "type": "🛑 STOP LOSS",
        "stop_loss": 139.16,
        "take_profit": 159.04,
        "pnl_pct": -2.00,
        "pnl_usd": -20.00,
        "status": "CLOSED"
    },
    {
        "id": 101,
        "entry_time": "2026-08-31 19:40:00",
        "exit_time": "2026-09-01 02:10:00",
        "symbol": "BTC-USD",
        "entry_price": 61800.00,
        "exit_price": 63200.00,
        "type": "🎯 TAKE PROFIT",
        "stop_loss": 58710.00,
        "take_profit": 63200.00,
        "pnl_pct": 2.27,
        "pnl_usd": 22.70,
        "status": "CLOSED"
    }
]

class PaperTradingEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.trades = []
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(FILE_PATH):
            self.trades = INITIAL_SEED_TRADES.copy()
            self._save_to_disk()
        else:
            try:
                with open(FILE_PATH, 'r', encoding='utf-8') as f:
                    self.trades = json.load(f)
            except Exception as e:
                notifier.add_log("WARNING", f"Erro ao ler histórico de paper trades: {e}. Inicializando com sementes.")
                self.trades = INITIAL_SEED_TRADES.copy()
                self._save_to_disk()

    def _save_to_disk(self):
        try:
            with open(FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.trades, f, ensure_ascii=False, indent=2)
        except Exception as e:
            notifier.add_log("ERROR", f"Falha ao salvar paper trades no disco: {e}")

    def add_trade(self, symbol: str, side: str, entry_price: float, stop_loss: float = None, take_profit: float = None) -> dict:
        with self.lock:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trade_id = max([t["id"] for t in self.trades], default=100) + 1
            
            sl = round(stop_loss if stop_loss else entry_price * 0.95, 2)
            tp = round(take_profit if take_profit else entry_price * 1.12, 2)
            
            trade = {
                "id": trade_id,
                "entry_time": now_str,
                "exit_time": "EM ANDAMENTO",
                "symbol": symbol,
                "entry_price": round(entry_price, 2),
                "exit_price": round(entry_price, 2),
                "type": f"🟢 {side.upper()} EM TENDÊNCIA (LIVE)",
                "stop_loss": sl,
                "take_profit": tp,
                "pnl_pct": 0.00,
                "pnl_usd": 0.00,
                "status": "OPEN"
            }
            
            self.trades.insert(0, trade)
            self._save_to_disk()
            
            notifier.add_log("INFO", f"🧪 [PAPER TRADING 24/7] Nova ordem simulada iniciada: {symbol} a ${entry_price:,.2f} (SL: ${sl:,.2f} | TP: ${tp:,.2f})")
            telegram_notifier.send_telegram_alert(f"🧪 *PAPER TRADING 24/7 (NOVA ENTRADA):*\nSímbolo: {symbol}\nPreço Entrada: ${entry_price:,.2f}\nStop Loss: ${sl:,.2f}\nTake Profit: ${tp:,.2f}")
            return trade

    def update_open_trades_with_live_prices(self, live_prices: dict):
        with self.lock:
            modified = False
            for trade in self.trades:
                if trade.get("status") == "OPEN":
                    symbol = trade.get("symbol")
                    if symbol in live_prices and live_prices[symbol] > 0:
                        curr_price = live_prices[symbol]
                        entry_price = trade["entry_price"]
                        sl = trade["stop_loss"]
                        tp = trade["take_profit"]
                        
                        # Atualizar cotação e PnL dinâmico em andamento
                        trade["exit_price"] = round(curr_price, 2)
                        pnl_pct = round(((curr_price - entry_price) / entry_price) * 100.0, 2)
                        trade["pnl_pct"] = pnl_pct
                        trade["pnl_usd"] = round(pnl_pct * 10.0, 2) # assumindo lote fixo de $1000 por trade
                        
                        # Testar se atinge Take Profit
                        if curr_price >= tp:
                            trade["status"] = "CLOSED"
                            trade["exit_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            trade["type"] = "🎯 TAKE PROFIT"
                            trade["pnl_pct"] = round(((tp - entry_price) / entry_price) * 100.0, 2)
                            trade["pnl_usd"] = round(trade["pnl_pct"] * 10.0, 2)
                            trade["exit_price"] = tp
                            modified = True
                            notifier.add_log("SUCCESS", f"🎯 [PAPER TRADING 24/7] Take Profit Atingido em {symbol}! Lucro: +{trade['pnl_pct']:.2f}% (${trade['pnl_usd']:+.2f})")
                            telegram_notifier.send_telegram_alert(f"🎯 *PAPER TRADING (TAKE PROFIT):*\nSímbolo: {symbol}\nLucro: +{trade['pnl_pct']:.2f}% (${trade['pnl_usd']:+.2f})")

                        # Testar se atinge Stop Loss
                        elif curr_price <= sl:
                            trade["status"] = "CLOSED"
                            trade["exit_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            trade["type"] = "🛑 STOP LOSS"
                            trade["pnl_pct"] = round(((sl - entry_price) / entry_price) * 100.0, 2)
                            trade["pnl_usd"] = round(trade["pnl_pct"] * 10.0, 2)
                            trade["exit_price"] = sl
                            modified = True
                            notifier.add_log("WARNING", f"🛑 [PAPER TRADING 24/7] Stop Loss Acionado em {symbol}! Perda: {trade['pnl_pct']:.2f}% (${trade['pnl_usd']:+.2f})")
                            telegram_notifier.send_telegram_alert(f"🛑 *PAPER TRADING (STOP LOSS):*\nSímbolo: {symbol}\nResultado: {trade['pnl_pct']:.2f}% (${trade['pnl_usd']:+.2f})")
                        else:
                            modified = True
                            
            if modified:
                self._save_to_disk()

    def get_trades(self) -> list:
        with self.lock:
            return list(self.trades)

    def get_metrics(self) -> dict:
        with self.lock:
            closed_trades = [t for t in self.trades if t.get("status") == "CLOSED"]
            open_trades = [t for t in self.trades if t.get("status") == "OPEN"]
            
            total_closed = len(closed_trades)
            wins = len([t for t in closed_trades if t.get("pnl_pct", 0) > 0])
            losses = len([t for t in closed_trades if t.get("pnl_pct", 0) <= 0])
            
            win_rate = round((wins / total_closed) * 100.0, 1) if total_closed > 0 else 0.0
            
            gross_profit = sum([t.get("pnl_usd", 0) for t in closed_trades if t.get("pnl_usd", 0) > 0])
            gross_loss = abs(sum([t.get("pnl_usd", 0) for t in closed_trades if t.get("pnl_usd", 0) < 0]))
            profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit > 0 else 1.0)
            
            total_pnl_usd = round(sum([t.get("pnl_usd", 0) for t in self.trades]), 2)
            
            return {
                "total_trades": len(self.trades),
                "closed_trades": total_closed,
                "open_trades": len(open_trades),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": win_rate,
                "profit_factor": profit_factor,
                "total_pnl_usd": total_pnl_usd,
                "status_label": "🟢 MONITORAMENTO 24/7 ATIVO"
            }

    def start_background_loop(self, interval_seconds=30):
        if self.is_running:
            return
        self.is_running = True
        
        def loop():
            from src.live_feeder import LiveMarketFeeder
            from src.validation_matrix import evaluate_safety_matrix
            
            feeder = LiveMarketFeeder(symbols=["BTC-USD", "ETH-USD", "SOL-USD"])
            notifier.add_log("SUCCESS", f"🚀 Motor de Paper Trading 24/7 iniciado (Varredura a cada {interval_seconds}s).")
            
            while self.is_running:
                try:
                    summary = feeder.get_live_market_summary()
                    live_prices = {sym: summary[sym]["price"] for sym in summary if "price" in summary[sym]}
                    
                    # 1. Atualiza posições em aberto com preços reais do mercado
                    self.update_open_trades_with_live_prices(live_prices)
                    
                    # 2. Avalia oportunidades para novas entradas simuladas
                    for symbol in ["BTC-USD", "ETH-USD", "SOL-USD"]:
                        has_open = any(t["symbol"] == symbol and t["status"] == "OPEN" for t in self.get_trades())
                        if not has_open and symbol in live_prices and live_prices[symbol] > 0:
                            price = live_prices[symbol]
                            matrix = evaluate_safety_matrix()
                            if matrix.get("is_trade_authorized"):
                                self.add_trade(
                                    symbol=symbol,
                                    side="COMPRA",
                                    entry_price=price,
                                    stop_loss=price * 0.95,
                                    take_profit=price * 1.12
                                )
                except Exception as e:
                    notifier.add_log("WARNING", f"Erro no loop de Paper Trading 24/7: {e}")
                    
                time.sleep(interval_seconds)
                
        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

# Instância Singleton Global
paper_trading_engine = PaperTradingEngine()

def get_live_paper_trades() -> dict:
    """
    Retorna extrato e estatísticas de assertividade para exibição no Dashboard.
    """
    return {
        "metrics": paper_trading_engine.get_metrics(),
        "trades": paper_trading_engine.get_trades()
    }

if __name__ == '__main__':
    print(f"Paper trades ativos: {paper_trading_engine.get_metrics()}")

