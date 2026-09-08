"""
AUDITORIA SANBOX ISOLADA DO FORWARD OOS (Spot & Futuros 4 Carteiras).
Arquivado em tests/sandbox/ sem qualquer conexão com a execução de produção 24/7.
Este script NÃO roda automaticamente e NÃO altera os estados de produção.
"""
import os
import sys
import json
import shutil
import datetime
import pandas as pd

sys.path.append(r"c:\Users\fdfm1\OneDrive\Desktop\Tradeer")
from src.engines.forward_oos_cloud_engine import ForwardOOSCloudEngine, FROZEN_PARAMS
from src.engines.forward_oos_futures_cloud_engine import ForwardOOSFuturesCloudEngine, FROZEN_FUTURES_PARAMS, BYBIT_RISK_TIERS

BASE_DIR = r"c:\Users\fdfm1\OneDrive\Desktop\Tradeer"
REAL_DATA_DIR = os.path.join(BASE_DIR, "data")
REAL_SPOT_STATE_FILE = os.path.join(REAL_DATA_DIR, "forward_oos_state.json")
REAL_FUTURES_STATE_FILE = os.path.join(REAL_DATA_DIR, "forward_oos_futures_state.json")

SANDBOX_DIR = os.path.join(REAL_DATA_DIR, "sandbox_test_env")
SANDBOX_SPOT_STATE_FILE = os.path.join(SANDBOX_DIR, "forward_oos_state_sandbox.json")
SANDBOX_FUTURES_STATE_FILE = os.path.join(SANDBOX_DIR, "forward_oos_futures_state_sandbox.json")

class SandboxSpotEngine(ForwardOOSCloudEngine):
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.is_running = False
        self.symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        self.state = {}
        self._ensure_sandbox_storage()

    def _ensure_storage(self):
        self._ensure_sandbox_storage()

    def _save_to_disk(self):
        self._save_sandbox_disk()

    def _ensure_sandbox_storage(self):
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        if not os.path.exists(SANDBOX_SPOT_STATE_FILE):
            self.state = self._get_initial_state()
            self._save_sandbox_disk()
        else:
            with open(SANDBOX_SPOT_STATE_FILE, 'r', encoding='utf-8') as f:
                self.state = json.load(f)

    def _save_sandbox_disk(self):
        with open(SANDBOX_SPOT_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

class SandboxFuturesEngine(ForwardOOSFuturesCloudEngine):
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.is_running = False
        self.symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        self.state = {}
        self.mock_intraday_data = {}
        self._ensure_sandbox_storage()

    def _ensure_storage(self):
        self._ensure_sandbox_storage()

    def _save_to_disk(self):
        self._save_sandbox_disk()

    def _ensure_sandbox_storage(self):
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        if not os.path.exists(SANDBOX_FUTURES_STATE_FILE):
            self.state = self._get_initial_state()
            self._save_sandbox_disk()
        else:
            with open(SANDBOX_FUTURES_STATE_FILE, 'r', encoding='utf-8') as f:
                self.state = json.load(f)

    def _save_sandbox_disk(self):
        with open(SANDBOX_FUTURES_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def _get_spot_signals(self) -> list:
        if not os.path.exists(SANDBOX_SPOT_STATE_FILE):
            return []
        with open(SANDBOX_SPOT_STATE_FILE, 'r', encoding='utf-8') as f:
            spot_data = json.load(f)
        
        signals = []
        for pos in spot_data.get("open_positions", []):
            sym = pos["symbol"]
            date_str = pos["entry_date"]
            sig_id = f"SPOT_SIG_{sym}_{date_str}"
            signals.append({
                "spot_signal_id": sig_id,
                "symbol": sym,
                "entry_date": date_str,
                "spot_entry_price": pos["entry_price"],
                "stop_loss": pos["stop_loss"],
                "status": "OPEN",
                "spot_pos_data": pos
            })
        for tr in spot_data.get("closed_trades", []):
            sym = tr["symbol"]
            date_str = tr["entry_date"]
            sig_id = f"SPOT_SIG_{sym}_{date_str}"
            signals.append({
                "spot_signal_id": sig_id,
                "symbol": sym,
                "entry_date": date_str,
                "spot_entry_price": tr["entry_price"],
                "exit_date": tr["exit_date"],
                "spot_exit_price": tr["exit_price"],
                "stop_loss": tr["stop_loss"],
                "exit_reason": tr["exit_reason"],
                "status": "CLOSED",
                "spot_trade_data": tr
            })
        return signals

    def _fetch_intraday_ohlc(self, symbol: str) -> pd.DataFrame:
        if symbol in self.mock_intraday_data:
            return self.mock_intraday_data[symbol]
        return pd.DataFrame()

def mock_get_futures_price(symbol_bybit, target_time):
    dt_str = str(target_time)
    if "BTC" in symbol_bybit:
        price = 80000.0 if "08" in dt_str else 88000.0
    elif "ETH" in symbol_bybit:
        price = 3000.0 if "15" in dt_str else 2800.0
    elif "SOL" in symbol_bybit:
        price = 135.0
    else:
        price = 100.0
    ts_ms = 1788806400000
    return (price, ts_ms, "2026-09-08 00:00:00 UTC")

def mock_calculate_real_funding(symbol_bybit, entry_ts_ms, exit_ts_ms, notional_usd):
    return {
        "events_detail": [{"funding_event_timestamp": "2026-09-08 08:00:00 UTC", "funding_rate": 0.0001, "position_notional": notional_usd, "funding_paid_received": -0.03}],
        "funding_events_count": 1,
        "funding_paid_usd": 0.03,
        "funding_received_usd": 0.0,
        "funding_net_usd": -0.03
    }

def run_sandbox_verification():
    print("=" * 80)
    print(" SUÍTE DE ARQUIVAMENTO/AUDITORIA DO SANDBOX")
    print("=" * 80)
    print("Este script é mantido em tests/sandbox/ apenas para fins de documentação histórica.")

if __name__ == "__main__":
    run_sandbox_verification()
