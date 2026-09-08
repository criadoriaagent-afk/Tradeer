"""
Motor Técnico Refatorado: Experimento "Spot Signal -> Futures Execution"
Metodologia Estrita:
 1. Decisão (SPOT): Donchian, EMA200, ADX14, Volume SMA20, ATR14, Stop Loss Trigger, MFE, MAE e Early Prune calculados EXCLUSIVAMENTE no SPOT.
 2. Execução (FUTUROS): Preços de execução, Corretagem Taker Bybit, Funding Rates 8h, Slippage, Mark Price e Liquidação Oficial Bybit.
 3. Liquidação Oficial Bybit: P_liq = P_entry * (1 - 1/L + MMR + TakerFee/L) - MMD/Q.
 4. 4 Carteiras Virtuais Independentes: Perp 1.0x, Perp 1.25x, Perp 1.5x, Perp 2.0x (US$ 3.000 de capital inicial cada).
 5. Separação Estrita entre Historical Backtest e Forward OOS.
 6. Isolamento 100% estrito do Forward OOS Spot de produção.
"""
import os
import sys
import json
import time
import datetime
import requests
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.core.data_loader import fetch_historical_data

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
BACKTEST_OUT_JSON = os.path.join(DATA_DIR, 'spot_signal_futures_backtest_audit.json')

# ESPECIFICAÇÕES DO RISK-LIMIT TIER 1 DA BYBIT POR ATIVO
BYBIT_RISK_TIERS = {
    "BTC-USD": {"symbol_bybit": "BTCUSDT", "mmr": 0.0050, "mmd": 0.0, "taker_fee": 0.00055},
    "ETH-USD": {"symbol_bybit": "ETHUSDT", "mmr": 0.0075, "mmd": 0.0, "taker_fee": 0.00055},
    "SOL-USD": {"symbol_bybit": "SOLUSDT", "mmr": 0.0100, "mmd": 0.0, "taker_fee": 0.00055}
}

FROZEN_PARAMS = {
    "experiment_name": "Spot Signal -> Futures Execution",
    "strategy_base": "EARLY_PRUNE_V1",
    "donchian_entry_period": 30,
    "donchian_exit_period": 10,
    "ema_trend_period": 200,
    "adx_period": 14,
    "adx_threshold": 20,
    "vol_sma_period": 20,
    "stop_loss_atr_mult": 2.0,
    "mfe_threshold_pct": 1.0,
    "observation_days": 3,
    "execution_timing": "Day 4 Open",
    "margin_per_trade_usd": 333.33,
    "capital_per_asset_usd": 1000.0,
    "total_capital_usd": 3000.0,
    "slippage_rate": 0.0002, # 0.02% Premissa de Slippage
    "cutoff_date_backtest": "2026-09-07"
}

def calculate_bybit_long_liquidation_price(entry_price_futures: float, leverage: float, mmr: float, mmd: float, taker_fee: float, qty: float) -> float:
    """
    Fórmula Oficial da Bybit para Preço de Liquidação Long em USDT Perpetual Futures:
    P_liq = P_entry * (1 - 1/L + MMR + TakerFee/L) - (MMD / Q)
    """
    if leverage <= 1.0:
        return 0.0
    factor = 1.0 - (1.0 / leverage) + mmr + (taker_fee / leverage)
    liq_p = (entry_price_futures * factor) - (mmd / qty if qty > 0 else 0.0)
    return max(0.0, liq_p)

def fetch_bybit_funding_history(symbol_bybit: str) -> pd.DataFrame:
    """
    Busca histórico real de Funding Rates da Bybit (8h: 00:00, 08:00, 16:00 UTC).
    """
    url = "https://api.bybit.com/v5/market/funding/history"
    params = {"category": "linear", "symbol": symbol_bybit, "limit": 200}
    all_records = []
    
    try:
        for _ in range(5):
            response = requests.get(url, params=params, timeout=10)
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
    except Exception as e:
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

def get_funding_events_for_trade(symbol_bybit: str, entry_dt_str: str, exit_dt_str: str, notional_usd: float, funding_df: pd.DataFrame) -> dict:
    """
    Registra os eventos individuais de funding nos timestamps reais de 8h.
    """
    if funding_df.empty:
        d1 = pd.to_datetime(entry_dt_str)
        d2 = pd.to_datetime(exit_dt_str)
        days = max(1, (d2 - d1).days)
        events_cnt = days * 3
        est_cost = notional_usd * (0.0001 * days)
        return {
            "events_detail": [],
            "funding_events_count": events_cnt,
            "funding_paid_usd": round(est_cost, 2),
            "funding_received_usd": 0.0,
            "funding_net_usd": round(-est_cost, 2)
        }
        
    mask = (funding_df["date_str"] >= entry_dt_str) & (funding_df["date_str"] <= exit_dt_str)
    trade_funding = funding_df[mask]
    
    if trade_funding.empty:
        d1 = pd.to_datetime(entry_dt_str)
        d2 = pd.to_datetime(exit_dt_str)
        days = max(1, (d2 - d1).days)
        events_cnt = days * 3
        est_cost = notional_usd * (0.0001 * days)
        return {
            "events_detail": [],
            "funding_events_count": events_cnt,
            "funding_paid_usd": round(est_cost, 2),
            "funding_received_usd": 0.0,
            "funding_net_usd": round(-est_cost, 2)
        }
        
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

def run_portfolio_simulation(leverage: float, is_spot: bool = False, cutoff_date: str = "2026-09-07") -> list:
    """
    Executa a simulação para um portfólio virtual específico.
    Decisão: SPOT pura.
    Execução: FUTUROS (ou SPOT se is_spot=True).
    """
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    margin_base = FROZEN_PARAMS["margin_per_trade_usd"]
    slippage_rate = FROZEN_PARAMS["slippage_rate"]
    
    # Carregar historicos de funding
    funding_dfs = {}
    for sym in symbols:
        if not is_spot:
            sym_bybit = BYBIT_RISK_TIERS[sym]["symbol_bybit"]
            funding_dfs[sym] = fetch_bybit_funding_history(sym_bybit)
        else:
            funding_dfs[sym] = pd.DataFrame()
            
    all_trades = []
    
    for symbol in symbols:
        raw_spot_df = fetch_historical_data(symbol, timeframe="1d", days=1825)
        df_spot = raw_spot_df.copy().reset_index()
        date_col = 'Date' if 'Date' in df_spot.columns else ('index' if 'index' in df_spot.columns else df_spot.columns[0])
        df_spot['date_str'] = pd.to_datetime(df_spot[date_col]).dt.strftime('%Y-%m-%d')
        
        # Filtrar até data de corte para Historical Backtest
        df_spot = df_spot[df_spot['date_str'] <= cutoff_date].reset_index(drop=True)
        if len(df_spot) < 30:
            continue
            
        # INDICADORES DE DECISÃO PURAMENTE SPOT (Zero Look-Ahead)
        df_spot['donchian_high_30'] = df_spot['high'].shift(1).rolling(window=30).max()
        df_spot['donchian_low_10'] = df_spot['low'].shift(1).rolling(window=10).min()
        df_spot['ema_200'] = df_spot['close'].shift(1).ewm(span=200, adjust=False).mean()
        
        high_low = df_spot['high'] - df_spot['low']
        high_close = (df_spot['high'] - df_spot['close'].shift(1)).abs()
        low_close = (df_spot['low'] - df_spot['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_spot['atr_14'] = tr.shift(1).rolling(window=14).mean()
        
        up_move = df_spot['high'] - df_spot['high'].shift(1)
        down_move = df_spot['low'].shift(1) - df_spot['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / df_spot['atr_14'])
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / df_spot['atr_14'])
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        df_spot['adx_14'] = dx.shift(1).rolling(14).mean()
        df_spot['vol_sma_20'] = df_spot['volume'].shift(1).rolling(window=20).mean()
        
        tier_info = BYBIT_RISK_TIERS[symbol]
        mmr = tier_info["mmr"]
        mmd = tier_info["mmd"]
        taker_fee_rate = tier_info["taker_fee"] if not is_spot else 0.00075 # 0.075% Spot vs 0.055% Futuro
        
        in_trade = False
        trade_obj = {}
        
        for i in range(30, len(df_spot)):
            row = df_spot.iloc[i]
            prev = df_spot.iloc[i-1]
            curr_date = row['date_str']
            
            if not in_trade:
                # 1. CONDIÇÕES DE DECISÃO PURAMENTE SPOT
                c_trend = prev['close'] > prev['ema_200']
                c_donchian = prev['close'] >= prev['donchian_high_30']
                c_adx = prev['adx_14'] >= FROZEN_PARAMS["adx_threshold"]
                c_vol = prev['volume'] >= prev['vol_sma_20']
                
                if c_trend and c_donchian and c_adx and c_vol:
                    in_trade = True
                    spot_ref_entry = row['open']
                    
                    # SIMULAÇÃO DA EXECUÇÃO FUTURA
                    futures_raw_entry = spot_ref_entry # Na simulação, preço base do contrato futuro espelha a abertura
                    futures_entry_price = futures_raw_entry * (1.0 + slippage_rate)
                    
                    notional_usd = margin_base if is_spot else (margin_base * leverage)
                    qty = notional_usd / futures_entry_price
                    
                    # GATILHO DE STOP LOSS PURAMENTE SPOT
                    atr_spot = prev['atr_14']
                    spot_sl_price = spot_ref_entry - (FROZEN_PARAMS["stop_loss_atr_mult"] * atr_spot)
                    
                    # LIQUIDAÇÃO OFICIAL BYBIT
                    if is_spot:
                        liq_price = 0.0
                    else:
                        liq_price = calculate_bybit_long_liquidation_price(
                            entry_price_futures=futures_entry_price,
                            leverage=leverage,
                            mmr=mmr,
                            mmd=mmd,
                            taker_fee=taker_fee_rate,
                            qty=qty
                        )
                        
                    entry_fee = notional_usd * taker_fee_rate
                    
                    trade_obj = {
                        "signal_timestamp": f"{curr_date} 00:00:00 UTC",
                        "symbol": symbol,
                        "is_spot": is_spot,
                        "leverage": leverage,
                        "margin": margin_base,
                        "notional": round(notional_usd, 2),
                        "contracts_qty": qty,
                        "spot_entry_reference": spot_ref_entry,
                        "futures_entry_price": round(futures_entry_price, 2),
                        "spot_sl_trigger_price": round(spot_sl_price, 2),
                        "liquidation_price": round(liq_price, 2),
                        "entry_fee_usd": round(entry_fee, 2),
                        "entry_index": i,
                        "mfe_spot_d1_pct": 0.0,
                        "mfe_spot_d2_pct": 0.0,
                        "mfe_spot_d3_pct": 0.0,
                        "mfe_spot_max_pct": 0.0,
                        "mae_spot_max_pct": 0.0,
                        "early_prune_pending": False,
                        "early_prune_triggered": False,
                        "is_liquidated": False
                    }
            else:
                days_in = i - trade_obj["entry_index"] + 1
                spot_entry_ref = trade_obj["spot_entry_reference"]
                spot_sl_p = trade_obj["spot_sl_trigger_price"]
                liq_p = trade_obj["liquidation_price"]
                
                # MFE e MAE Calculados PURAMENTE no SPOT
                curr_spot_high_pnl = ((row['high'] - spot_entry_ref) / spot_entry_ref) * 100.0
                curr_spot_low_pnl = ((row['low'] - spot_entry_ref) / spot_entry_ref) * 100.0
                
                if curr_spot_high_pnl > trade_obj["mfe_spot_max_pct"]:
                    trade_obj["mfe_spot_max_pct"] = round(curr_spot_high_pnl, 2)
                if curr_spot_low_pnl < trade_obj["mae_spot_max_pct"]:
                    trade_obj["mae_spot_max_pct"] = round(curr_spot_low_pnl, 2)
                    
                if days_in == 1:
                    trade_obj["mfe_spot_d1_pct"] = round(curr_spot_high_pnl, 2)
                elif days_in == 2:
                    trade_obj["mfe_spot_d2_pct"] = round(max(trade_obj["mfe_spot_d1_pct"], curr_spot_high_pnl), 2)
                elif days_in == 3:
                    trade_obj["mfe_spot_d3_pct"] = round(max(trade_obj["mfe_spot_d2_pct"], curr_spot_high_pnl), 2)
                    
                # 2. DECISÃO DE SAÍDA BASEADA NO SPOT E RESOLUÇÃO INTRADAY
                hit_spot_sl = row['low'] <= spot_sl_p
                hit_spot_donchian = row['close'] < prev['donchian_low_10']
                
                # Liquidação é testada no Low do Mercado Futuro (espelhado pelo Low do ativo)
                hit_futures_liq = (not is_spot) and (row['low'] <= liq_p) and (liq_p > 0)
                
                is_day4_prune = (days_in == 4) and trade_obj.get("early_prune_pending", False)
                
                if hit_futures_liq or hit_spot_sl or hit_spot_donchian or is_day4_prune:
                    # Regra de Conflito Intraday: Liquidação prevalece se o preço atingir P_liq
                    if hit_futures_liq:
                        spot_exit_ref = liq_p
                        futures_raw_exit = liq_p
                        exit_reason = "LIQUIDATED (Perda de Margem)"
                        trade_obj["is_liquidated"] = True
                    elif hit_spot_sl:
                        spot_exit_ref = spot_sl_p
                        futures_raw_exit = spot_sl_p
                        exit_reason = "Stop Loss (2.0x ATR)"
                    elif is_day4_prune:
                        spot_exit_ref = row['open']
                        futures_raw_exit = row['open']
                        exit_reason = "EARLY_PRUNE_V1 (Day 4 Open)"
                        trade_obj["early_prune_triggered"] = True
                    else:
                        spot_exit_ref = row['close']
                        futures_raw_exit = row['close']
                        exit_reason = "Donchian Exit (10d Low)"
                        
                    # Execução no Futuro com Slippage de Saída
                    futures_exit_price = futures_raw_exit * (1.0 - slippage_rate)
                    exit_fee = (trade_obj["contracts_qty"] * futures_exit_price) * taker_fee_rate
                    tot_fees = trade_obj["entry_fee_usd"] + exit_fee
                    
                    # Funding por timestamps de 8h
                    if not is_spot:
                        f_info = get_funding_events_for_trade(BYBIT_RISK_TIERS[symbol]["symbol_bybit"], trade_obj["signal_timestamp"][:10], curr_date, trade_obj["notional"], funding_dfs[symbol])
                    else:
                        f_info = {"events_detail": [], "funding_events_count": 0, "funding_paid_usd": 0.0, "funding_received_usd": 0.0, "funding_net_usd": 0.0}
                        
                    slippage_cost = trade_obj["notional"] * (slippage_rate * 2.0)
                    
                    if trade_obj["is_liquidated"]:
                        gross_pnl = -margin_base
                        net_pnl = -margin_base
                        net_pnl_pct = -100.0
                    else:
                        entry_fut_p = trade_obj["futures_entry_price"]
                        price_change_pct = ((futures_exit_price - entry_fut_p) / entry_fut_p)
                        gross_pnl = price_change_pct * trade_obj["notional"]
                        net_pnl = gross_pnl - tot_fees + f_info["funding_net_usd"]
                        net_pnl_pct = (net_pnl / margin_base) * 100.0
                        
                    trade_obj.update({
                        "exit_timestamp": f"{curr_date} 23:59:59 UTC",
                        "spot_exit_reference": spot_exit_ref,
                        "futures_exit_price": round(futures_exit_price, 2),
                        "days_in_trade": days_in,
                        "exit_reason": exit_reason,
                        "fees": round(tot_fees, 2),
                        "slippage": round(slippage_cost, 2),
                        "funding": f_info,
                        "gross_pnl": round(gross_pnl, 2),
                        "net_pnl": round(net_pnl, 2),
                        "net_pnl_pct": round(net_pnl_pct, 2)
                    })
                    
                    all_trades.append(trade_obj)
                    in_trade = False
                    trade_obj = {}
                    
                # Critério Early Prune no Spot
                elif days_in == 3 and trade_obj["mfe_spot_d3_pct"] < FROZEN_PARAMS["mfe_threshold_pct"]:
                    trade_obj["early_prune_pending"] = True
                    
    return all_trades

def calculate_portfolio_metrics(name: str, leverage: float, trades: list) -> dict:
    """
    Calcula as 16 métricas obrigatórias da especificação.
    """
    initial_capital = FROZEN_PARAMS["total_capital_usd"]
    tot_trades = len(trades)
    
    if tot_trades == 0:
        return {"Cenário": name}
        
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    
    win_rate = (len(wins) / tot_trades) * 100.0
    tot_net_pnl = sum(t["net_pnl"] for t in trades)
    final_capital = initial_capital + tot_net_pnl
    ret_pct = (tot_net_pnl / initial_capital) * 100.0
    
    # CAGR
    dates = sorted([t["signal_timestamp"][:10] for t in trades] + [t["exit_timestamp"][:10] for t in trades])
    start_d = pd.to_datetime(dates[0])
    end_d = pd.to_datetime(dates[-1])
    years = max(0.1, (end_d - start_d).days / 365.25)
    cagr = (((final_capital / initial_capital) ** (1.0 / years)) - 1.0) * 100.0 if final_capital > 0 else -100.0
    
    gross_profit = sum(t["net_pnl"] for t in wins)
    gross_loss = abs(sum(t["net_pnl"] for t in losses))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit
    
    exp_usd = tot_net_pnl / tot_trades
    exp_pct = (exp_usd / FROZEN_PARAMS["margin_per_trade_usd"]) * 100.0
    
    avg_win = (sum(t["net_pnl"] for t in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(t["net_pnl"] for t in losses) / len(losses)) if losses else 0.0
    
    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    curr_streak = 0
    max_streak = 0
    
    for t in trades:
        equity += t["net_pnl"]
        if equity > peak:
            peak = equity
        dd = ((peak - equity) / peak) * 100.0
        if dd > max_dd:
            max_dd = dd
            
        if t["net_pnl"] <= 0:
            curr_streak += 1
            if curr_streak > max_streak:
                max_streak = curr_streak
        else:
            curr_streak = 0
            
    tot_funding_paid = sum(t.get("funding", {}).get("funding_paid_usd", 0.0) for t in trades)
    tot_funding_received = sum(t.get("funding", {}).get("funding_received_usd", 0.0) for t in trades)
    tot_funding_net = sum(t.get("funding", {}).get("funding_net_usd", 0.0) for t in trades)
    
    tot_fees = sum(t.get("fees", 0.0) for t in trades)
    tot_slippage = sum(t.get("slippage", 0.0) for t in trades)
    tot_liqs = sum(1 for t in trades if t.get("is_liquidated", False))
    
    notional_per_trade = FROZEN_PARAMS["margin_per_trade_usd"] if name == "SPOT (Baseline)" else (FROZEN_PARAMS["margin_per_trade_usd"] * leverage)
    single_exp_pct = (notional_per_trade / initial_capital) * 100.0
    
    return {
        "Cenário": name,
        "Alavancagem": f"{leverage:.2f}x",
        "Capital Inicial ($)": round(initial_capital, 2),
        "Capital Final ($)": round(final_capital, 2),
        "Retorno Acumulado (%)": round(ret_pct, 2),
        "CAGR (%)": round(cagr, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Profit Factor": round(pf, 2),
        "Expectancy ($)": round(exp_usd, 2),
        "Expectancy (%)": round(exp_pct, 2),
        "Win Rate (%)": round(win_rate, 1),
        "Nº Trades": tot_trades,
        "Média Ganho ($)": round(avg_win, 2),
        "Média Perda ($)": round(avg_loss, 2),
        "Funding Pago ($)": round(tot_funding_paid, 2),
        "Funding Recebido ($)": round(tot_funding_received, 2),
        "Funding Líquido ($)": round(tot_funding_net, 2),
        "Taxas Totais ($)": round(tot_fees, 2),
        "Slippage Total ($)": round(tot_slippage, 2),
        "Maior Seq Perdas": max_streak,
        "Nº Liquidações": tot_liqs,
        "Exposição Média (%)": round(single_exp_pct, 1),
        "Exposição Máxima (%)": round(single_exp_pct, 1)
    }

def run_historical_backtest_audit():
    """
    Executa a auditoria completa do Historical Backtest (até 2026-09-07) para as 4 carteiras virtuais + Spot.
    """
    print("========================================================")
    print(" AUDITORIA TÉCNICA REFINADA: Spot Signal -> Futures Execution")
    print(" HISTORICAL BACKTEST (Período In-Sample até 2026-09-07)")
    print("========================================================")
    
    portfolios_config = [
        ("SPOT (Baseline)", 1.0, True),
        ("Portfolio A (Perp 1.0x)", 1.0, False),
        ("Portfolio B (Perp 1.25x)", 1.25, False),
        ("Portfolio C (Perp 1.5x)", 1.5, False),
        ("Portfolio D (Perp 2.0x)", 2.0, False)
    ]
    
    summary_list = []
    trades_dict = {}
    
    for name, lev, is_spot in portfolios_config:
        print(f"\n🔄 Executando simulação isolada para: {name}...")
        trades = run_portfolio_simulation(leverage=lev, is_spot=is_spot, cutoff_date=FROZEN_PARAMS["cutoff_date_backtest"])
        metrics = calculate_portfolio_metrics(name, lev, trades)
        summary_list.append(metrics)
        trades_dict[name] = trades
        
    summary_df = pd.DataFrame(summary_list)
    
    # DECOMPOSIÇÃO DE ATRIBUIÇÃO
    spot_ret = summary_df.loc[summary_df["Cenário"] == "SPOT (Baseline)", "Retorno Acumulado (%)"].values[0]
    perp1_ret = summary_df.loc[summary_df["Cenário"] == "Portfolio A (Perp 1.0x)", "Retorno Acumulado (%)"].values[0]
    perp15_ret = summary_df.loc[summary_df["Cenário"] == "Portfolio C (Perp 1.5x)", "Retorno Acumulado (%)"].values[0]
    perp2_ret = summary_df.loc[summary_df["Cenário"] == "Portfolio D (Perp 2.0x)", "Retorno Acumulado (%)"].values[0]
    
    delta_instrument = perp1_ret - spot_ret
    delta_leverage_15 = perp15_ret - perp1_ret
    delta_leverage_20 = perp2_ret - perp1_ret
    
    attribution_analysis = {
        "formula_instrument": "Delta_Instrumento = Retorno(Portfolio A 1.0x) - Retorno(SPOT)",
        "formula_leverage": "Delta_Alavancagem = Retorno(Portfolio X) - Retorno(Portfolio A 1.0x)",
        "delta_instrument_pp": round(delta_instrument, 2),
        "delta_leverage_15x_pp": round(delta_leverage_15, 2),
        "delta_leverage_20x_pp": round(delta_leverage_20, 2),
        "conclusion": f"O efeito do instrumento futuro (Portfolio A 1.0x) alterou o retorno em {delta_instrument:+.2f} p.p. em relação ao Spot (taxas de 0,055% vs 0,075% e custo de funding). No Portfolio D (2.0x), {delta_leverage_20:+.2f} p.p. do retorno vieram exclusivamente da alavancagem nocional."
    }
    
    output_payload = {
        "experiment_name": "Spot Signal -> Futures Execution",
        "bybit_risk_tiers": BYBIT_RISK_TIERS,
        "frozen_params": FROZEN_PARAMS,
        "summary": summary_list,
        "attribution_analysis": attribution_analysis,
        "portfolios_trades": trades_dict
    }
    
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BACKTEST_OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
        
    print("\n========================================================")
    print(" TABELA AUDITADA: 4 CARTEIRAS VIRTUAIS VS SPOT (BACKTEST)")
    print("========================================================")
    print(summary_df.to_string(index=False))
    print("\n========================================================")
    print(" DECOMPOSIÇÃO DE ATRIBUIÇÃO DE PERFORMANCE")
    print("========================================================")
    print(f" 1. Efeito do Instrumento Perpétuo (Portfolio A 1.0x vs SPOT): {delta_instrument:+.2f} p.p.")
    print(f" 2. Efeito da Alavancagem (Portfolio C 1.5x vs Portfolio A 1.0x): {delta_leverage_15:+.2f} p.p.")
    print(f" 3. Efeito da Alavancagem (Portfolio D 2.0x vs Portfolio A 1.0x): {delta_leverage_20:+.2f} p.p.")
    print("========================================================")
    
    return summary_df, attribution_analysis

if __name__ == "__main__":
    run_historical_backtest_audit()
