"""
Motor de Execução Cloud Idempotente do Forward OOS - EARLY_PRUNE_V1 (Candidata Congelada).
Roda 24/7 no Render com persistência em data/forward_oos_state.json.
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
STATE_FILE = os.path.join(DATA_DIR, 'forward_oos_state.json')

# CONFIGURAÇÕES DA ESTRATÉGIA CONGELADA (EARLY_PRUNE_V1)
FROZEN_PARAMS = {
    "strategy_name": "EARLY_PRUNE_V1",
    "status_label": "CONGELADA (FORWARD OOS PROSPECTIVO)",
    "mfe_threshold_pct": 1.0,
    "observation_days": 3,
    "execution_timing": "Day 4 Open",
    "sizing_per_trade_usd": 333.33,
    "fee_rate_pct": 0.15,
    "fee_rate_decimal": 0.0015,
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

class ForwardOOSCloudEngine:
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
                print(f"[ForwardOOS] Aviso ao carregar estado: {e}. Criando estado limpo.")
                self.state = self._get_initial_state()
                self._save_to_disk()

    def _get_initial_state(self) -> dict:
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return {
            "status": "FORWARD OOS ATIVO EM NUVEM",
            "candidate_id": "EARLY_PRUNE_V1",
            "freeze_confirmation": "100% INALTERADA (+1.0% MFE 3d, Day 4 Open exit, 2x ATR SL)",
            "start_date_oos": FROZEN_PARAMS["oos_start_date"],
            "start_timestamp_utc": FROZEN_PARAMS["oos_start_timestamp_utc"],
            "last_processed_date_1d": FROZEN_PARAMS["in_sample_end_date"],
            "last_heartbeat": now_utc,
            "capital_initial": 3000.0,
            "capital_current": 3000.0,
            "open_positions": [],
            "closed_trades": [],
            "daily_equity_series": [
                {
                    "date": FROZEN_PARAMS["in_sample_end_date"],
                    "equity_usd": 3000.0,
                    "pnl_pct": 0.0
                }
            ],
            "execution_logs": [
                {
                    "timestamp": now_utc,
                    "level": "INFO",
                    "message": "🟢 Forward OOS (Paper Trading Prospectivo) iniciado oficialmente para EARLY_PRUNE_V1 congelada."
                }
            ],
            "frozen_params": FROZEN_PARAMS,
            "metrics": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 1.0,
                "total_pnl_usd": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "losses_saved_usd": 0.0,
                "profit_sacrificed_usd": 0.0
            }
        }

    def _save_to_disk(self):
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ForwardOOS] Erro ao salvar estado: {e}")

    def add_log(self, level: str, message: str):
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        log_entry = {
            "timestamp": now_utc,
            "level": level,
            "message": message
        }
        self.state.setdefault("execution_logs", []).insert(0, log_entry)
        # Manter no máximo 100 logs
        self.state["execution_logs"] = self.state["execution_logs"][:100]

    def _fetch_indicator_data(self, symbol: str) -> pd.DataFrame:
        """
        Baixa candles diários e calcula indicadores com shift(1) estrito sem look-ahead.
        """
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
            
            # Shift(1) para evitar qualquer futuro/look-ahead
            df['donchian_high_30'] = df['high'].shift(1).rolling(window=30).max()
            df['donchian_low_10'] = df['low'].shift(1).rolling(window=10).min()
            df['ema_200'] = df['close'].shift(1).ewm(span=200, adjust=False).mean()
            df['ema_20'] = df['close'].shift(1).ewm(span=20, adjust=False).mean()
            
            # ATR 14
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = tr.shift(1).rolling(window=14).mean()
            
            # ADX 14
            up_move = df['high'] - df['high'].shift(1)
            down_move = df['low'].shift(1) - df['low']
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / df['atr_14'])
            minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / df['atr_14'])
            dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
            df['adx_14'] = dx.shift(1).rolling(14).mean()
            
            # Vol SMA 20
            df['vol_sma_20'] = df['volume'].shift(1).rolling(window=20).mean()
            
            return df
        except Exception as e:
            print(f"[ForwardOOS] Erro ao baixar dados de {symbol}: {e}")
            return pd.DataFrame()

    def process_daily_update(self):
        """
        Rotina idempotente de atualização diária.
        Verifica se há novos candles diários fechados após last_processed_date_1d.
        """
        with self.lock:
            now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            self.state["last_heartbeat"] = now_utc
            
            # 1. Obter dados mais recentes de BTC-USD para checar a data do último candle
            btc_df = self._fetch_indicator_data("BTC-USD")
            if btc_df.empty:
                self._save_to_disk()
                return

            latest_available_date = btc_df['date_str'].iloc[-1]
            last_processed = self.state.get("last_processed_date_1d", FROZEN_PARAMS["in_sample_end_date"])
            
            # Garantir idempotência: Se já processamos o último candle disponível, não fazemos nada
            if latest_available_date <= last_processed:
                self._save_to_disk()
                return

            self.add_log("INFO", f"🔄 Novo candle diário detectado: {latest_available_date}. Executando validação prospectiva...")

            # 2. Processar posições abertas e novas entradas para todos os ativos
            sizing = FROZEN_PARAMS["sizing_per_trade_usd"]
            fee_decimal = FROZEN_PARAMS["fee_rate_decimal"]
            
            for symbol in self.symbols:
                df = self._fetch_indicator_data(symbol)
                if df.empty or len(df) < 200:
                    continue

                # Localizar linhas a serem processadas desde last_processed até latest_available_date
                unprocessed_indices = df[df['date_str'] > last_processed].index.tolist()
                
                for idx in unprocessed_indices:
                    row = df.iloc[idx]
                    prev = df.iloc[idx-1]
                    curr_date = row['date_str']
                    
                    # A. ATUALIZAR POSIÇÕES EM ABERTO PARA ESTE ATIVO
                    open_pos_list = [p for p in self.state["open_positions"] if p["symbol"] == symbol]
                    
                    for pos in open_pos_list:
                        pos["days_in_trade"] += 1
                        days = pos["days_in_trade"]
                        entry_p = pos["entry_price"]
                        sl_p = pos["stop_loss"]
                        
                        # Atualizar MFE diário do trade
                        curr_high_pnl = ((row['high'] - entry_p) / entry_p) * 100.0
                        if curr_high_pnl > pos["mfe_max_pct"]:
                            pos["mfe_max_pct"] = round(curr_high_pnl, 2)

                        if days == 1:
                            pos["mfe_d1_pct"] = round(curr_high_pnl, 2)
                        elif days == 2:
                            pos["mfe_d2_pct"] = round(max(pos["mfe_d1_pct"], curr_high_pnl), 2)
                        elif days == 3:
                            pos["mfe_d3_pct"] = round(max(pos["mfe_d2_pct"], curr_high_pnl), 2)

                        # Checagens de Saída
                        hit_sl = row['low'] <= sl_p
                        hit_donchian_exit = row['close'] < prev['donchian_low_10']
                        
                        # EARLY PRUNE CHECK (No Fechamento do Dia 3 -> Saída na Abertura do Dia 4)
                        is_day4_open_prune = False
                        if days == 4 and pos.get("early_prune_pending", False):
                            is_day4_open_prune = True

                        if hit_sl or hit_donchian_exit or is_day4_open_prune:
                            if hit_sl:
                                exit_price = sl_p
                                exit_reason = "Stop Loss (2.0x ATR)"
                            elif is_day4_open_prune:
                                exit_price = row['open']
                                exit_reason = "EARLY_PRUNE_V1 (Day 4 Open)"
                            else:
                                exit_price = row['close']
                                exit_reason = "Donchian Exit (10d Low)"

                            raw_pct = ((exit_price - entry_p) / entry_p) * 100.0
                            net_pct = raw_pct - (fee_decimal * 2 * 100.0)
                            pnl_usd = (net_pct / 100.0) * sizing
                            
                            closed_trade = {
                                "trade_id": len(self.state["closed_trades"]) + 1,
                                "symbol": symbol,
                                "entry_date": pos["entry_date"],
                                "exit_date": curr_date,
                                "entry_price": entry_p,
                                "exit_price": round(exit_price, 2),
                                "stop_loss": sl_p,
                                "days_in_trade": days,
                                "mfe_d1_pct": pos["mfe_d1_pct"],
                                "mfe_d2_pct": pos["mfe_d2_pct"],
                                "mfe_d3_pct": pos["mfe_d3_pct"],
                                "early_prune_triggered": pos.get("early_prune_triggered", False) or is_day4_open_prune,
                                "exit_reason": exit_reason,
                                "pnl_pct": round(net_pct, 2),
                                "pnl_usd": round(pnl_usd, 2)
                            }
                            
                            self.state["closed_trades"].insert(0, closed_trade)
                            self.state["open_positions"] = [p for p in self.state["open_positions"] if p["id"] != pos["id"]]
                            self.state["capital_current"] = round(self.state["capital_current"] + pnl_usd, 2)
                            
                            log_msg = f"🚪 [FORWARD OOS] Posição Encerrada em {symbol}: {exit_reason} | Preço Saída: ${exit_price:,.2f} | PnL: {net_pct:+.2f}% (${pnl_usd:+.2f})"
                            self.add_log("SUCCESS" if pnl_usd > 0 else "WARNING", log_msg)

                        # Se chegou ao final do Dia 3 e MFE < 1.0%, marcar pendência de Early Prune para abertura do Dia 4
                        elif days == 3 and pos["mfe_d3_pct"] < FROZEN_PARAMS["mfe_threshold_pct"]:
                            pos["early_prune_pending"] = True
                            pos["early_prune_triggered"] = True
                            self.add_log("INFO", f"⚡ [EARLY PRUNE DETECTADO] {symbol} MFE D3 ({pos['mfe_d3_pct']}%) < +1.0%. Agendada saída na Abertura do Dia 4.")

                    # B. CHECAR NOVAS ENTRADAS SE NÃO HOUVER POSIÇÃO EM ABERTO NO ATIVO
                    has_open = any(p["symbol"] == symbol for p in self.state["open_positions"])
                    if not has_open:
                        c_trend = prev['close'] > prev['ema_200']
                        c_donchian = prev['close'] >= prev['donchian_high_30']
                        c_adx = prev['adx_14'] >= FROZEN_PARAMS["adx_threshold"]
                        c_vol = prev['volume'] >= prev['vol_sma_20']
                        
                        if c_trend and c_donchian and c_adx and c_vol:
                            entry_price = row['open']
                            atr_val = prev['atr_14']
                            sl_price = entry_price - (FROZEN_PARAMS["stop_loss_atr_mult"] * atr_val)
                            
                            new_pos = {
                                "id": f"{symbol}_{curr_date}",
                                "symbol": symbol,
                                "entry_date": curr_date,
                                "entry_price": round(entry_price, 2),
                                "stop_loss": round(sl_price, 2),
                                "sizing_usd": sizing,
                                "days_in_trade": 1,
                                "mfe_d1_pct": round(((row['high'] - entry_price) / entry_price) * 100.0, 2),
                                "mfe_d2_pct": 0.0,
                                "mfe_d3_pct": 0.0,
                                "mfe_max_pct": round(((row['high'] - entry_price) / entry_price) * 100.0, 2),
                                "early_prune_pending": False,
                                "early_prune_triggered": False
                            }
                            
                            self.state["open_positions"].append(new_pos)
                            self.add_log("INFO", f"🚀 [FORWARD OOS ENTRADA PROSPECTIVA] Nova ordem aberta em {symbol} a ${entry_price:,.2f} (SL: ${sl_price:,.2f})")

            # 3. Atualizar métricas gerais e histórico de patrimônio diário
            self._update_metrics()
            self.state["last_processed_date_1d"] = latest_available_date
            
            # Registrar patrimônio diário
            tot_pnl = sum(t["pnl_usd"] for t in self.state["closed_trades"])
            tot_equity = round(self.state["capital_initial"] + tot_pnl, 2)
            tot_ret = round((tot_pnl / self.state["capital_initial"]) * 100.0, 2)
            
            # Evitar entradas duplicadas de data na série de equity
            eq_series = self.state.get("daily_equity_series", [])
            if not eq_series or eq_series[-1]["date"] != latest_available_date:
                eq_series.append({
                    "date": latest_available_date,
                    "equity_usd": tot_equity,
                    "pnl_pct": tot_ret
                })
                self.state["daily_equity_series"] = eq_series

            self._save_to_disk()

    def _update_metrics(self):
        closed = self.state.get("closed_trades", [])
        tot = len(closed)
        wins = [t for t in closed if t["pnl_usd"] > 0]
        losses = [t for t in closed if t["pnl_usd"] <= 0]
        
        win_rate = (len(wins) / tot * 100.0) if tot > 0 else 0.0
        gross_profit = sum(t["pnl_usd"] for t in wins)
        gross_loss = abs(sum(t["pnl_usd"] for t in losses))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
        
        tot_pnl = sum(t["pnl_usd"] for t in closed)
        tot_ret = (tot_pnl / self.state["capital_initial"]) * 100.0
        
        # Drawdown max
        cap_init = self.state["capital_initial"]
        equity = cap_init
        peak = cap_init
        max_dd = 0.0
        for t in reversed(closed): # Da mais antiga à mais recente
            equity += t["pnl_usd"]
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

        self.state["metrics"] = {
            "total_trades": tot,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(pf, 2),
            "total_pnl_usd": round(tot_pnl, 2),
            "total_return_pct": round(tot_ret, 2),
            "max_drawdown_pct": round(max_dd, 2)
        }

    def get_full_state(self) -> dict:
        with self.lock:
            # Forçar atualização de heartbeat
            self.state["last_heartbeat"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            return self.state

    def start_background_loop(self, interval_seconds=900):
        """
        Inicia a thread 24/7 em segundo plano (Varredura a cada 15 min / 900s).
        """
        if self.is_running:
            return
        self.is_running = True
        
        def loop():
            print(f"[ForwardOOS Engine] Motor 24/7 de Validação Prospectiva iniciado (Varredura a cada {interval_seconds}s)...")
            while self.is_running:
                try:
                    self.process_daily_update()
                except Exception as e:
                    print(f"[ForwardOOS Engine] Erro no loop: {e}")
                time.sleep(interval_seconds)

        t = threading.Thread(target=loop, daemon=True)
        t.start()

# Instância Singleton Global
forward_oos_cloud_engine = ForwardOOSCloudEngine()

if __name__ == '__main__':
    engine = ForwardOOSCloudEngine()
    engine.process_daily_update()
    print("Estado do Forward OOS:", json.dumps(engine.get_full_state(), indent=2))
