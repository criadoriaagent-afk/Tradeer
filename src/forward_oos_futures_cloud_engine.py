"""
Motor Cloud Engine Idempotente 24/7 do Forward OOS - Futuros Perpétuos (4 Carteiras Virtuais).
Roda em nuvem no Render com persistência isolada em data/forward_oos_futures_state.json.

DECISÃO: 100% Derivada da EARLY_PRUNE_V1 Spot OOS (lida diretamente de data/forward_oos_state.json).
EXECUÇÃO: Espelhada nas 4 carteiras virtuais (Perp 1.0x, 1.25x, 1.5x, 2.0x).
FUNDING: Taxas reais de Funding da Bybit consultadas via API nos timestamps de 8h (00:00, 08:00, 16:00 UTC).
IDEMPOTÊNCIA: Rastreabilidade estrita via SPOT_SIGNAL_ID para conter qualquer duplicação.
"""
import os
import sys
import json
import time
import datetime
import threading
import requests
import pandas as pd
import numpy as np
import yfinance as yf

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
STATE_FILE = os.path.join(DATA_DIR, 'forward_oos_futures_state.json')
SPOT_STATE_FILE = os.path.join(DATA_DIR, 'forward_oos_state.json')

BYBIT_RISK_TIERS = {
    "BTC-USD": {"symbol_bybit": "BTCUSDT", "mmr": 0.0050, "mmd": 0.0, "taker_fee": 0.00055},
    "ETH-USD": {"symbol_bybit": "ETHUSDT", "mmr": 0.0075, "mmd": 0.0, "taker_fee": 0.00055},
    "SOL-USD": {"symbol_bybit": "SOLUSDT", "mmr": 0.0100, "mmd": 0.0, "taker_fee": 0.00055}
}

FROZEN_FUTURES_PARAMS = {
    "strategy_name": "EARLY_PRUNE_FUTURES_V1",
    "instrument": "Bybit Linear USDT Perpetual Futures",
    "portfolios": [
        {"name": "Portfolio A (Perp 1.0x)", "leverage": 1.0},
        {"name": "Portfolio B (Perp 1.25x)", "leverage": 1.25},
        {"name": "Portfolio C (Perp 1.5x)", "leverage": 1.5},
        {"name": "Portfolio D (Perp 2.0x)", "leverage": 2.0}
    ],
    "status_label": "FORWARD OOS FUTUROS 24/7 ATIVO (4 CARTEIRAS)",
    "mfe_threshold_pct": 1.0,
    "observation_days": 3,
    "execution_timing": "Day 4 Open",
    "margin_per_trade_usd": 333.33,
    "capital_initial_per_portfolio_usd": 3000.0,
    "slippage_rate": 0.0002,
    "stop_loss_atr_mult": 2.0,
    "donchian_entry_period": 30,
    "donchian_exit_period": 10,
    "ema_trend_period": 200,
    "adx_period": 14,
    "adx_threshold": 20,
    "vol_sma_period": 20,
    "in_sample_end_date": "2026-09-07",
    "oos_start_date": "2026-09-08",
    "oos_start_timestamp_utc": "2026-09-08T00:00:00Z"
}

def calculate_bybit_long_liquidation_price(entry_price_futures: float, leverage: float, mmr: float, mmd: float, taker_fee: float, qty: float) -> float:
    if leverage <= 1.0:
        return 0.0
    factor = 1.0 - (1.0 / leverage) + mmr + (taker_fee / leverage)
    liq_p = (entry_price_futures * factor) - (mmd / qty if qty > 0 else 0.0)
    return max(0.0, liq_p)

def fetch_bybit_funding_history_real(symbol_bybit: str) -> pd.DataFrame:
    """
    Busca o histórico real de Funding Rates da Bybit (8h: 00:00, 08:00, 16:00 UTC).
    """
    url = "https://api.bybit.com/v5/market/funding/history"
    params = {"category": "linear", "symbol": symbol_bybit, "limit": 200}
    all_records = []
    
    try:
        for _ in range(3):
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                result = data.get("result", {})
                list_data = result.get("list", [])
                if not list_data:
                    break
                all_records.extend(list_data)
                cursor = result.get("nextPageCursor")
                if not cursor:
                    break
                params["cursor"] = cursor
            else:
                break
    except Exception:
        pass
        
    if not all_records:
        return pd.DataFrame(columns=["date_str", "funding_rate", "timestamp_ms"])
        
    df_f = pd.DataFrame(all_records)
    df_f["funding_rate"] = df_f["fundingRate"].astype(float)
    df_f["timestamp_ms"] = df_f["fundingRateTimestamp"].astype(int)
    df_f["datetime_utc"] = pd.to_datetime(df_f["timestamp_ms"], unit="ms", utc=True)
    df_f["date_str"] = df_f["datetime_utc"].dt.strftime("%Y-%m-%d")
    df_f["timestamp_str"] = df_f["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    df_f = df_f.sort_values("timestamp_ms").reset_index(drop=True)
    return df_f

def calculate_real_funding_for_trade(symbol_bybit: str, entry_dt_str: str, exit_dt_str: str, notional_usd: float) -> dict:
    """
    Calcula o custo REAL de Funding sem estimativas pré-fixadas, registrando cada evento de 8h.
    """
    df_f = fetch_bybit_funding_history_real(symbol_bybit)
    if df_f.empty:
        return {"events_detail": [], "funding_events_count": 0, "funding_paid_usd": 0.0, "funding_received_usd": 0.0, "funding_net_usd": 0.0}

    mask = (df_f["date_str"] >= entry_dt_str) & (df_f["date_str"] <= exit_dt_str)
    trade_funding = df_f[mask]
    
    events_detail = []
    paid = 0.0
    received = 0.0
    
    for _, row_f in trade_funding.iterrows():
        rate = row_f["funding_rate"]
        ts_str = row_f["timestamp_str"]
        cost_event = notional_usd * rate
        
        if cost_event > 0:
            paid += cost_event
        else:
            received += abs(cost_event)
            
        events_detail.append({
            "funding_event_timestamp": ts_str,
            "funding_rate": rate,
            "position_notional": notional_usd,
            "funding_paid_received": round(-cost_event, 2)
        })
        
    net = received - paid
    return {
        "events_detail": events_detail,
        "funding_events_count": len(events_detail),
        "funding_paid_usd": round(paid, 2),
        "funding_received_usd": round(received, 2),
        "funding_net_usd": round(net, 2)
    }

class ForwardOOSFuturesCloudEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        self.state = {}
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(STATE_FILE):
            self.state = self._get_initial_state()
            self._save_to_disk()
        else:
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    self.state = json.load(f)
            except Exception as e:
                print(f"[ForwardOOS Futures] Erro ao carregar estado: {e}. Criando estado inicial.")
                self.state = self._get_initial_state()
                self._save_to_disk()

    def _get_initial_state(self) -> dict:
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        portfolios_state = {}
        for p in FROZEN_FUTURES_PARAMS["portfolios"]:
            pname = p["name"]
            portfolios_state[pname] = {
                "name": pname,
                "leverage": p["leverage"],
                "capital_initial": 3000.0,
                "capital_current": 3000.0,
                "open_positions": [],
                "closed_trades": [],
                "daily_equity_series": [
                    {"date": FROZEN_FUTURES_PARAMS["in_sample_end_date"], "equity_usd": 3000.0, "pnl_pct": 0.0}
                ],
                "metrics": {
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate_pct": 0.0,
                    "profit_factor": 1.0,
                    "total_pnl_usd": 0.0,
                    "total_return_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "funding_total_usd": 0.0,
                    "liquidations_count": 0
                }
            }

        return {
            "status": "FORWARD OOS FUTUROS 24/7 ATIVO (4 CARTEIRAS)",
            "candidate_id": "EARLY_PRUNE_FUTURES_V1",
            "signal_source": "100% DERIVADO DA EARLY_PRUNE_V1 SPOT (forward_oos_state.json)",
            "start_date_oos": FROZEN_FUTURES_PARAMS["oos_start_date"],
            "start_timestamp_utc": FROZEN_FUTURES_PARAMS["oos_start_timestamp_utc"],
            "last_processed_date_1d": FROZEN_FUTURES_PARAMS["in_sample_end_date"],
            "last_heartbeat": now_utc,
            "processed_spot_signal_ids": [],
            "portfolios": portfolios_state,
            "execution_logs": [
                {
                    "timestamp": now_utc,
                    "level": "INFO",
                    "message": "⚡ Forward OOS Futuros 24/7 (4 Carteiras Virtuais 1.0x a 2.0x) acoplado diretamente ao Spot OOS."
                }
            ],
            "frozen_params": FROZEN_FUTURES_PARAMS
        }

    def _save_to_disk(self):
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ForwardOOS Futures] Erro ao salvar estado: {e}")

    def add_log(self, level: str, message: str):
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        log_entry = {
            "timestamp": now_utc,
            "level": level,
            "message": message
        }
        self.state.setdefault("execution_logs", []).insert(0, log_entry)
        self.state["execution_logs"] = self.state["execution_logs"][:100]

    def _get_spot_signals() -> list:
        """
        Lê diretamente os sinais emitidos pelo Spot OOS em data/forward_oos_state.json.
        """
        if not os.path.exists(SPOT_STATE_FILE):
            return []
        try:
            with open(SPOT_STATE_FILE, 'r', encoding='utf-8') as f:
                spot_data = json.load(f)
            
            signals = []
            # Sinais de posições abertas no Spot
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
            # Sinais de trades já encerrados no Spot
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
        except Exception as e:
            print(f"[ForwardOOS Futures] Erro ao ler sinais do Spot: {e}")
            return []

    def _fetch_intraday_ohlc(self, symbol: str) -> pd.DataFrame:
        try:
            df = yf.download(symbol, period="30d", interval="1d", progress=False)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}).dropna()
            df = df.reset_index()
            date_col = 'Date' if 'Date' in df.columns else df.columns[0]
            df['date_str'] = df[date_col].dt.strftime('%Y-%m-%d')
            return df
        except Exception:
            return pd.DataFrame()

    def process_daily_update(self):
        with self.lock:
            now_utc_dt = datetime.datetime.now(datetime.timezone.utc)
            now_utc_str = now_utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            today_utc = now_utc_dt.strftime("%Y-%m-%d")
            self.state["last_heartbeat"] = now_utc_str

            margin_base = FROZEN_FUTURES_PARAMS["margin_per_trade_usd"]
            slippage_rate = FROZEN_FUTURES_PARAMS["slippage_rate"]
            portfolios_list = FROZEN_FUTURES_PARAMS["portfolios"]
            processed_ids = set(self.state.get("processed_spot_signal_ids", []))

            # 1. OBTER SINAIS REAIS GERADOS PELO SPOT OOS (100% Pura Fonte Única)
            spot_signals = ForwardOOSFuturesCloudEngine._get_spot_signals()

            # 2. PROCESSAR SINAIS SPOT NAS 4 CARTEIRAS VIRTUAIS DE FUTUROS
            for sig in spot_signals:
                sig_id = sig["spot_signal_id"]
                sym = sig["symbol"]
                curr_date = sig["entry_date"]
                
                # Regra de Corte: Ignorar qualquer sinal anterior ao início oficial do OOS (2026-09-08)
                if curr_date < FROZEN_FUTURES_PARAMS["oos_start_date"]:
                    continue

                tier_info = BYBIT_RISK_TIERS[sym]
                mmr = tier_info["mmr"]
                mmd = tier_info["mmd"]
                taker_fee_rate = tier_info["taker_fee"]

                for p_config in portfolios_list:
                    pname = p_config["name"]
                    lev = p_config["leverage"]
                    port_state = self.state["portfolios"][pname]
                    notional_usd = margin_base * lev

                    # Checar duplicação (Idempotência)
                    already_processed = any(t.get("spot_signal_id") == sig_id for t in port_state["closed_trades"]) or any(p.get("spot_signal_id") == sig_id for p in port_state["open_positions"])
                    
                    if not already_processed and sig["status"] in ["OPEN", "CLOSED"]:
                        # Nova posição espelhada diretamente do Sinal Spot
                        spot_entry_ref = sig["spot_entry_price"]
                        futures_entry_price = spot_entry_ref * (1.0 + slippage_rate)
                        qty = notional_usd / futures_entry_price
                        spot_sl_price = sig["stop_loss"]

                        liq_price = calculate_bybit_long_liquidation_price(
                            entry_price_futures=futures_entry_price,
                            leverage=lev,
                            mmr=mmr,
                            mmd=mmd,
                            taker_fee=taker_fee_rate,
                            qty=qty
                        )
                        entry_fee = notional_usd * taker_fee_rate
                        futures_trade_id = f"FUT_{pname}_{sig_id}"

                        new_pos = {
                            "id": futures_trade_id,
                            "futures_trade_id": futures_trade_id,
                            "spot_signal_id": sig_id,
                            "symbol": sym,
                            "leverage": lev,
                            "entry_date": curr_date,
                            "spot_entry_reference": spot_entry_ref,
                            "futures_entry_price": round(futures_entry_price, 2),
                            "spot_sl_trigger_price": round(spot_sl_price, 2),
                            "liquidation_price": round(liq_price, 2),
                            "margin_usd": margin_base,
                            "notional_usd": notional_usd,
                            "contracts_qty": qty,
                            "entry_fee_usd": round(entry_fee, 2),
                            "days_in_trade": 1,
                            "mfe_d1_pct": 0.0,
                            "mfe_d2_pct": 0.0,
                            "mfe_d3_pct": 0.0,
                            "mfe_max_pct": 0.0,
                            "intraday_resolution_note": "Acompanhamento prospectivo via high/low do candle diário com fórmulas oficiais Bybit.",
                            "early_prune_pending": False
                        }

                        port_state["open_positions"].append(new_pos)
                        processed_ids.add(sig_id)
                        self.add_log("INFO", f"🚀 [{pname}] VÍNCULO DIRETO SPOT->FUTURES: {sig_id} -> {futures_trade_id} | Ativo: {sym} ({lev}x)")

            # 3. GERENCIAR E ATUALIZAR POSIÇÕES FUTURAS ABERTAS COM DADOS DE MERCADO E FUNDING REAL
            for symbol in self.symbols:
                df_market = self._fetch_intraday_ohlc(symbol)
                if df_market.empty:
                    continue

                for p_config in portfolios_list:
                    pname = p_config["name"]
                    lev = p_config["leverage"]
                    port_state = self.state["portfolios"][pname]
                    notional_usd = margin_base * lev
                    tier_info = BYBIT_RISK_TIERS[symbol]
                    taker_fee_rate = tier_info["taker_fee"]
                    sym_bybit = tier_info["symbol_bybit"]

                    open_pos_list = [p for p in port_state["open_positions"] if p["symbol"] == symbol]

                    for pos in open_pos_list:
                        # Verificar se o Spot OOS já encerrou este trade
                        corresponding_spot = [s for s in spot_signals if s["spot_signal_id"] == pos["spot_signal_id"]]
                        spot_is_closed = len(corresponding_spot) > 0 and corresponding_spot[0]["status"] == "CLOSED"

                        curr_market_rows = df_market[df_market["date_str"] >= pos["entry_date"]]
                        if not curr_market_rows.empty:
                            last_row = curr_market_rows.iloc[-1]
                            curr_date_str = last_row["date_str"]

                            # Teste de Liquidação Futura
                            hit_futures_liq = (lev > 1.0) and (last_row['low'] <= pos["liquidation_price"]) and (pos["liquidation_price"] > 0)
                            
                            if hit_futures_liq or spot_is_closed:
                                if hit_futures_liq:
                                    futures_raw_exit = pos["liquidation_price"]
                                    exit_reason = "LIQUIDATED (Perda de Margem - Bybit)"
                                    is_liquidated = True
                                else:
                                    spot_tr = corresponding_spot[0]
                                    futures_raw_exit = spot_tr["spot_exit_price"]
                                    exit_reason = spot_tr["exit_reason"]
                                    is_liquidated = False

                                futures_exit_price = futures_raw_exit * (1.0 - slippage_rate)
                                exit_fee = (pos["contracts_qty"] * futures_exit_price) * taker_fee_rate
                                tot_fees = pos["entry_fee_usd"] + exit_fee

                                # BUSCA DE FUNDING RATES REAIS DA BYBIT NOS TIMESTAMPS UTC
                                real_funding_info = calculate_real_funding_for_trade(
                                    symbol_bybit=sym_bybit,
                                    entry_dt_str=pos["entry_date"],
                                    exit_dt_str=curr_date_str,
                                    notional_usd=notional_usd
                                )
                                funding_net = real_funding_info["funding_net_usd"]

                                if is_liquidated:
                                    pnl_usd = -margin_base
                                    net_pct = -100.0
                                else:
                                    price_change_pct = ((futures_exit_price - pos["futures_entry_price"]) / pos["futures_entry_price"])
                                    gross_pnl = price_change_pct * notional_usd
                                    pnl_usd = gross_pnl - tot_fees + funding_net
                                    net_pct = (pnl_usd / margin_base) * 100.0

                                closed_trade = {
                                    "futures_trade_id": pos["futures_trade_id"],
                                    "spot_signal_id": pos["spot_signal_id"],
                                    "symbol": symbol,
                                    "leverage": lev,
                                    "entry_date": pos["entry_date"],
                                    "exit_date": curr_date_str,
                                    "spot_entry_reference": pos["spot_entry_reference"],
                                    "futures_entry_price": pos["futures_entry_price"],
                                    "futures_exit_price": round(futures_exit_price, 2),
                                    "stop_loss": pos["spot_sl_trigger_price"],
                                    "liquidation_price": pos["liquidation_price"],
                                    "is_liquidated": is_liquidated,
                                    "days_in_trade": len(curr_market_rows),
                                    "margin_usd": margin_base,
                                    "notional_usd": notional_usd,
                                    "exit_reason": exit_reason,
                                    "fees_usd": round(tot_fees, 2),
                                    "real_funding_detail": real_funding_info,
                                    "funding_net_usd": round(funding_net, 2),
                                    "net_pnl_usd": round(pnl_usd, 2),
                                    "net_pnl_pct": round(net_pct, 2)
                                }

                                port_state["closed_trades"].insert(0, closed_trade)
                                port_state["open_positions"] = [p for p in port_state["open_positions"] if p["id"] != pos["id"]]
                                port_state["capital_current"] = round(port_state["capital_current"] + pnl_usd, 2)
                                self.add_log("INFO", f"⚡ [{pname}] Trade Encerrado em {symbol} ({pos['futures_trade_id']}): {exit_reason} | PnL: {net_pct:+.2f}% (${pnl_usd:+.2f}) | Funding Real: ${funding_net:+.2f}")

            self.state["processed_spot_signal_ids"] = sorted(list(processed_ids))
            
            # Atualizar métricas e série diária para cada carteira
            for p_config in portfolios_list:
                pname = p_config["name"]
                self._update_portfolio_metrics(pname, today_utc)

            self.state["last_processed_date_1d"] = today_utc
            self._save_to_disk()

    def _update_portfolio_metrics(self, pname: str, latest_date: str):
        port = self.state["portfolios"][pname]
        closed = port.get("closed_trades", [])
        tot = len(closed)
        wins = [t for t in closed if t["net_pnl_usd"] > 0]
        losses = [t for t in closed if t["net_pnl_usd"] <= 0]
        
        win_rate = (len(wins) / tot * 100.0) if tot > 0 else 0.0
        gross_profit = sum(t["net_pnl_usd"] for t in wins)
        gross_loss = abs(sum(t["net_pnl_usd"] for t in losses))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
        
        tot_pnl = sum(t["net_pnl_usd"] for t in closed)
        cap_init = port["capital_initial"]
        tot_equity = round(cap_init + tot_pnl, 2)
        tot_ret = round((tot_pnl / cap_init) * 100.0, 2)
        
        equity = cap_init
        peak = cap_init
        max_dd = 0.0
        for t in reversed(closed):
            equity += t["net_pnl_usd"]
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

        tot_funding = sum(t.get("funding_net_usd", 0.0) for t in closed)
        liqs_cnt = sum(1 for t in closed if t.get("is_liquidated", False))

        port["capital_current"] = tot_equity
        port["metrics"] = {
            "total_trades": tot,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(pf, 2),
            "total_pnl_usd": round(tot_pnl, 2),
            "total_return_pct": round(tot_ret, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "funding_total_usd": round(tot_funding, 2),
            "liquidations_count": liqs_cnt
        }

        eq_series = port.get("daily_equity_series", [])
        if not eq_series or eq_series[-1]["date"] != latest_date:
            eq_series.append({
                "date": latest_date,
                "equity_usd": tot_equity,
                "pnl_pct": tot_ret
            })
            port["daily_equity_series"] = eq_series

    def get_full_state(self) -> dict:
        with self.lock:
            self.state["last_heartbeat"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            return self.state

    def start_background_loop(self, interval_seconds=900):
        if self.is_running:
            return
        self.is_running = True
        
        def loop():
            print(f"[ForwardOOS Futures Engine] Motor 24/7 de Futuros Perpétuos (4 Carteiras) iniciado (Varredura a cada {interval_seconds}s)...")
            while self.is_running:
                try:
                    self.process_daily_update()
                except Exception as e:
                    print(f"[ForwardOOS Futures Engine] Erro no loop: {e}")
                time.sleep(interval_seconds)

        t = threading.Thread(target=loop, daemon=True)
        t.start()

forward_oos_futures_cloud_engine = ForwardOOSFuturesCloudEngine()

if __name__ == '__main__':
    engine = ForwardOOSFuturesCloudEngine()
    engine.process_daily_update()
    print("Estado das 4 Carteiras de Futuros (Coupled to Spot):", json.dumps(engine.get_full_state(), indent=2))
