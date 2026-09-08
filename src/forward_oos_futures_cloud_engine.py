"""
Motor Cloud Engine Idempotente 24/7 do Forward OOS - Futuros Perpétuos (4 Carteiras Virtuais).
Roda em nuvem no Render com persistência isolada em data/forward_oos_futures_state.json.
Decisão: 100% Derivada da EARLY_PRUNE_V1 Spot.
Execução: Espelhada nas 4 carteiras virtuais (Perp 1.0x, 1.25x, 1.5x, 2.0x).
Idempotência: Rastreabilidade estrita via SPOT_SIGNAL_ID para impedir qualquer duplicação de ordens.
"""
import os
import sys
import json
import time
import datetime
import threading
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
                    "message": "⚡ Forward OOS Futuros 24/7 (4 Carteiras Virtuais 1.0x a 2.0x) iniciado oficialmente no Render."
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

    def _fetch_indicator_data(self, symbol: str) -> pd.DataFrame:
        try:
            df = yf.download(symbol, period="365d", interval="1d", progress=False)
            if df.empty:
                return pd.DataFrame()
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }).dropna()
            
            df = df.reset_index()
            date_col = 'Date' if 'Date' in df.columns else df.columns[0]
            df['date_str'] = df[date_col].dt.strftime('%Y-%m-%d')
            
            # Shift(1) Zero Look-Ahead no Spot
            df['donchian_high_30'] = df['high'].shift(1).rolling(window=30).max()
            df['donchian_low_10'] = df['low'].shift(1).rolling(window=10).min()
            df['ema_200'] = df['close'].shift(1).ewm(span=200, adjust=False).mean()
            
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = tr.shift(1).rolling(window=14).mean()
            
            up_move = df['high'] - df['high'].shift(1)
            down_move = df['low'].shift(1) - df['low']
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / df['atr_14'])
            minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / df['atr_14'])
            dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
            df['adx_14'] = dx.shift(1).rolling(14).mean()
            df['vol_sma_20'] = df['volume'].shift(1).rolling(window=20).mean()
            
            return df
        except Exception as e:
            print(f"[ForwardOOS Futures] Erro ao baixar dados de {symbol}: {e}")
            return pd.DataFrame()

    def process_daily_update(self):
        with self.lock:
            now_utc_dt = datetime.datetime.now(datetime.timezone.utc)
            now_utc_str = now_utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            today_utc = now_utc_dt.strftime("%Y-%m-%d")
            self.state["last_heartbeat"] = now_utc_str
            
            btc_df = self._fetch_indicator_data("BTC-USD")
            if btc_df.empty:
                self._save_to_disk()
                return

            # Considerar apenas candles 1D estritamente FECHADOS (datas < data atual UTC)
            closed_candles_df = btc_df[btc_df['date_str'] < today_utc]
            if closed_candles_df.empty:
                self._save_to_disk()
                return

            latest_closed_date = closed_candles_df['date_str'].iloc[-1]
            last_processed = self.state.get("last_processed_date_1d", FROZEN_FUTURES_PARAMS["in_sample_end_date"])
            
            if latest_closed_date <= last_processed:
                self._save_to_disk()
                return

            self.add_log("INFO", f"🔄 Novo candle 1D fechado detectado: {latest_closed_date}. Processando Forward OOS das 4 Carteiras...")

            margin_base = FROZEN_FUTURES_PARAMS["margin_per_trade_usd"]
            slippage_rate = FROZEN_FUTURES_PARAMS["slippage_rate"]
            portfolios_list = FROZEN_FUTURES_PARAMS["portfolios"]
            processed_ids = set(self.state.get("processed_spot_signal_ids", []))

            for symbol in self.symbols:
                df = self._fetch_indicator_data(symbol)
                if df.empty or len(df) < 30:
                    continue

                # Filtrar apenas velas estritamente fechadas e posteriores ao último candle processado
                closed_df = df[df['date_str'] < today_utc].reset_index(drop=True)
                unprocessed_indices = closed_df[closed_df['date_str'] > last_processed].index.tolist()
                
                tier_info = BYBIT_RISK_TIERS[symbol]
                mmr = tier_info["mmr"]
                mmd = tier_info["mmd"]
                taker_fee_rate = tier_info["taker_fee"]

                for idx in unprocessed_indices:
                    row = closed_df.iloc[idx]
                    prev = closed_df.iloc[idx-1]
                    curr_date = row['date_str']
                    
                    # Ignorar velas anteriores ao início oficial do OOS (2026-09-08)
                    if curr_date < FROZEN_FUTURES_PARAMS["oos_start_date"]:
                        continue

                    # 1. VERIFICAR SE O SINAL SPOT OOS DISPAROU NESTE CANDLE FECHADO
                    c_trend = prev['close'] > prev['ema_200']
                    c_donchian = prev['close'] >= prev['donchian_high_30']
                    c_adx = prev['adx_14'] >= FROZEN_FUTURES_PARAMS["adx_threshold"]
                    c_vol = prev['volume'] >= prev['vol_sma_20']
                    spot_signal_triggered = c_trend and c_donchian and c_adx and c_vol
                    
                    spot_signal_id = f"SPOT_SIG_{symbol}_{curr_date}"

                    for p_config in portfolios_list:
                        pname = p_config["name"]
                        lev = p_config["leverage"]
                        port_state = self.state["portfolios"][pname]
                        notional_usd = margin_base * lev
                        
                        # A) Atualizar posições abertas nesta carteira
                        open_pos_list = [p for p in port_state["open_positions"] if p["symbol"] == symbol]
                        
                        for pos in open_pos_list:
                            pos["days_in_trade"] += 1
                            days = pos["days_in_trade"]
                            entry_p = pos["futures_entry_price"]
                            sl_p = pos["spot_sl_trigger_price"]
                            liq_p = pos["liquidation_price"]

                            curr_high_pnl = ((row['high'] - pos["spot_entry_reference"]) / pos["spot_entry_reference"]) * 100.0
                            if curr_high_pnl > pos["mfe_max_pct"]:
                                pos["mfe_max_pct"] = round(curr_high_pnl, 2)

                            if days == 1:
                                pos["mfe_d1_pct"] = round(curr_high_pnl, 2)
                            elif days == 2:
                                pos["mfe_d2_pct"] = round(max(pos["mfe_d1_pct"], curr_high_pnl), 2)
                            elif days == 3:
                                pos["mfe_d3_pct"] = round(max(pos["mfe_d2_pct"], curr_high_pnl), 2)

                            hit_futures_liq = (lev > 1.0) and (row['low'] <= liq_p) and (liq_p > 0)
                            hit_spot_sl = row['low'] <= sl_p
                            hit_spot_donchian = row['close'] < prev['donchian_low_10']
                            is_day4_prune = (days == 4) and pos.get("early_prune_pending", False)

                            if hit_futures_liq or hit_spot_sl or hit_spot_donchian or is_day4_prune:
                                if hit_futures_liq:
                                    futures_raw_exit = liq_p
                                    exit_reason = "LIQUIDATED (Perda de Margem)"
                                    is_liquidated = True
                                elif hit_spot_sl:
                                    futures_raw_exit = sl_p
                                    exit_reason = "Stop Loss (2.0x ATR)"
                                    is_liquidated = False
                                elif is_day4_prune:
                                    futures_raw_exit = row['open']
                                    exit_reason = "EARLY_PRUNE_V1 (Day 4 Open)"
                                    is_liquidated = False
                                else:
                                    futures_raw_exit = row['close']
                                    exit_reason = "Donchian Exit (10d Low)"
                                    is_liquidated = False

                                futures_exit_price = futures_raw_exit * (1.0 - slippage_rate)
                                exit_fee = (pos["contracts_qty"] * futures_exit_price) * taker_fee_rate
                                tot_fees = pos["entry_fee_usd"] + exit_fee
                                est_funding_usd = notional_usd * (0.0001 * days)

                                if is_liquidated:
                                    pnl_usd = -margin_base
                                    net_pct = -100.0
                                else:
                                    price_change_pct = ((futures_exit_price - entry_p) / entry_p)
                                    gross_pnl = price_change_pct * notional_usd
                                    pnl_usd = gross_pnl - tot_fees - est_funding_usd
                                    net_pct = (pnl_usd / margin_base) * 100.0

                                futures_trade_id = f"FUT_TRADE_{pname}_{spot_signal_id}"

                                closed_trade = {
                                    "futures_trade_id": futures_trade_id,
                                    "spot_signal_id": pos["spot_signal_id"],
                                    "symbol": symbol,
                                    "leverage": lev,
                                    "entry_date": pos["entry_date"],
                                    "exit_date": curr_date,
                                    "spot_entry_reference": pos["spot_entry_reference"],
                                    "futures_entry_price": entry_p,
                                    "futures_exit_price": round(futures_exit_price, 2),
                                    "stop_loss": sl_p,
                                    "liquidation_price": liq_p,
                                    "is_liquidated": is_liquidated,
                                    "days_in_trade": days,
                                    "margin_usd": margin_base,
                                    "notional_usd": notional_usd,
                                    "mfe_d1_pct": pos["mfe_d1_pct"],
                                    "mfe_d2_pct": pos["mfe_d2_pct"],
                                    "mfe_d3_pct": pos["mfe_d3_pct"],
                                    "exit_reason": exit_reason,
                                    "fees_usd": round(tot_fees, 2),
                                    "funding_usd": round(-est_funding_usd, 2),
                                    "net_pnl_usd": round(pnl_usd, 2),
                                    "net_pnl_pct": round(net_pct, 2)
                                }

                                port_state["closed_trades"].insert(0, closed_trade)
                                port_state["open_positions"] = [p for p in port_state["open_positions"] if p["id"] != pos["id"]]
                                port_state["capital_current"] = round(port_state["capital_current"] + pnl_usd, 2)
                                self.add_log("INFO", f"⚡ [{pname}] Trade Encerrado em {symbol} ({futures_trade_id}): {exit_reason} | PnL: {net_pct:+.2f}% (${pnl_usd:+.2f})")

                            elif days == 3 and pos["mfe_d3_pct"] < FROZEN_FUTURES_PARAMS["mfe_threshold_pct"]:
                                pos["early_prune_pending"] = True

                        # B) Checar novas entradas com Idempotência Estrita por SPOT_SIGNAL_ID
                        has_open = any(p["symbol"] == symbol for p in port_state["open_positions"])
                        signal_already_processed_for_port = any(t.get("spot_signal_id") == spot_signal_id for t in port_state["closed_trades"]) or any(p.get("spot_signal_id") == spot_signal_id for p in port_state["open_positions"])

                        if not has_open and spot_signal_triggered and not signal_already_processed_for_port:
                            spot_entry_ref = row['open']
                            futures_entry_price = spot_entry_ref * (1.0 + slippage_rate)
                            qty = notional_usd / futures_entry_price
                            atr_spot = prev['atr_14']
                            spot_sl_price = spot_entry_ref - (FROZEN_FUTURES_PARAMS["stop_loss_atr_mult"] * atr_spot)

                            liq_price = calculate_bybit_long_liquidation_price(
                                entry_price_futures=futures_entry_price,
                                leverage=lev,
                                mmr=mmr,
                                mmd=mmd,
                                taker_fee=taker_fee_rate,
                                qty=qty
                            )
                            entry_fee = notional_usd * taker_fee_rate
                            futures_trade_id = f"FUT_POS_{pname}_{spot_signal_id}"

                            new_pos = {
                                "id": futures_trade_id,
                                "futures_trade_id": futures_trade_id,
                                "spot_signal_id": spot_signal_id,
                                "symbol": symbol,
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
                                "mfe_d1_pct": round(((row['high'] - spot_entry_ref) / spot_entry_ref) * 100.0, 2),
                                "mfe_d2_pct": 0.0,
                                "mfe_d3_pct": 0.0,
                                "mfe_max_pct": round(((row['high'] - spot_entry_ref) / spot_entry_ref) * 100.0, 2),
                                "early_prune_pending": False
                            }

                            port_state["open_positions"].append(new_pos)
                            processed_ids.add(spot_signal_id)
                            self.add_log("INFO", f"🚀 [{pname}] VÍNCULO SPOT->FUTURES: {spot_signal_id} -> {futures_trade_id} | Ativo: {symbol} ({lev}x) Entry: ${futures_entry_price:,.2f}")

            self.state["processed_spot_signal_ids"] = sorted(list(processed_ids))
            
            # Atualizar métricas e série diária para cada carteira
            for p_config in portfolios_list:
                pname = p_config["name"]
                self._update_portfolio_metrics(pname, latest_closed_date)

            self.state["last_processed_date_1d"] = latest_closed_date
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

        tot_funding = sum(t.get("funding_usd", 0.0) for t in closed)
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
    print("Estado do Forward OOS Futuros:", json.dumps(engine.get_full_state(), indent=2))
