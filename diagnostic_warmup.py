import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

sys.path.append(r"c:\Users\fdfm1\OneDrive\Documentos\N2")
from src.core.data_loader import fetch_historical_data
from src.engines.spot_signal_futures_cloud_engine import calculate_bybit_long_liquidation_price, calculate_portfolio_metrics, FROZEN_PARAMS, BYBIT_RISK_TIERS

def run_simulation_with_warmup(start_index: int, pre_warmup_days: int = 0, cutoff_date: str = "2026-09-07"):
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    margin_base = FROZEN_PARAMS["margin_per_trade_usd"]
    slippage_rate = FROZEN_PARAMS["slippage_rate"]
    
    all_trades = []
    
    for symbol in symbols:
        days_to_fetch = 1825 + pre_warmup_days
        raw_df = fetch_historical_data(symbol, timeframe="1d", days=days_to_fetch)
        df = raw_df.copy().reset_index()
        date_col = 'Date' if 'Date' in df.columns else ('index' if 'index' in df.columns else df.columns[0])
        df['date_str'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
        
        # Calcular indicadores SOBRE TODA A SÉRIE CARREGADA
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
        
        # Se temos pré-aquecimento, a janela de avaliação OFICIAL começa em 2021-09-09
        official_eval_df = df[df['date_str'] >= "2021-09-09"].reset_index(drop=True)
        official_eval_df = official_eval_df[official_eval_df['date_str'] <= cutoff_date].reset_index(drop=True)
        
        in_trade = False
        trade_obj = {}
        taker_fee_rate = 0.00075 # Spot Taker Fee
        
        for i in range(start_index, len(official_eval_df)):
            row = official_eval_df.iloc[i]
            prev = official_eval_df.iloc[i-1]
            curr_date = row['date_str']
            
            if not in_trade:
                c_trend = prev['close'] > prev['ema_200']
                c_donchian = prev['close'] >= prev['donchian_high_30']
                c_adx = prev['adx_14'] >= FROZEN_PARAMS["adx_threshold"]
                c_vol = prev['volume'] >= prev['vol_sma_20']
                
                if c_trend and c_donchian and c_adx and c_vol:
                    in_trade = True
                    spot_ref_entry = row['open']
                    futures_entry_price = spot_ref_entry * (1.0 + slippage_rate)
                    notional_usd = margin_base
                    qty = notional_usd / futures_entry_price
                    atr_spot = prev['atr_14']
                    spot_sl_price = spot_ref_entry - (FROZEN_PARAMS["stop_loss_atr_mult"] * atr_spot)
                    entry_fee = notional_usd * taker_fee_rate
                    
                    trade_obj = {
                        "signal_timestamp": f"{curr_date} 00:00:00 UTC",
                        "symbol": symbol,
                        "spot_entry_reference": spot_ref_entry,
                        "futures_entry_price": round(futures_entry_price, 2),
                        "spot_sl_trigger_price": round(spot_sl_price, 2),
                        "contracts_qty": qty,
                        "entry_fee_usd": round(entry_fee, 2),
                        "entry_index": i,
                        "candles_warmup_available": i if pre_warmup_days == 0 else (i + pre_warmup_days),
                        "ema_200_at_signal": round(prev['ema_200'], 2),
                        "close_at_signal": round(prev['close'], 2),
                        "donchian_high_30_at_signal": round(prev['donchian_high_30'], 2),
                        "adx_14_at_signal": round(prev['adx_14'], 2),
                        "vol_sma_20_at_signal": round(prev['vol_sma_20'], 2),
                        "mfe_spot_d1_pct": 0.0,
                        "mfe_spot_d2_pct": 0.0,
                        "mfe_spot_d3_pct": 0.0,
                        "mfe_spot_max_pct": 0.0,
                        "mae_spot_max_pct": 0.0,
                        "early_prune_pending": False,
                        "early_prune_triggered": False
                    }
            else:
                days_in = i - trade_obj["entry_index"] + 1
                spot_entry_ref = trade_obj["spot_entry_reference"]
                spot_sl_p = trade_obj["spot_sl_trigger_price"]
                
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
                    
                hit_spot_sl = row['low'] <= spot_sl_p
                hit_spot_donchian = row['close'] < prev['donchian_low_10']
                is_day4_prune = (days_in == 4) and trade_obj.get("early_prune_pending", False)
                
                if hit_spot_sl or hit_spot_donchian or is_day4_prune:
                    if hit_spot_sl:
                        spot_exit_ref = spot_sl_p
                        exit_reason = "Stop Loss (2.0x ATR)"
                    elif is_day4_prune:
                        spot_exit_ref = row['open']
                        exit_reason = "EARLY_PRUNE_V1 (Day 4 Open)"
                        trade_obj["early_prune_triggered"] = True
                    else:
                        spot_exit_ref = row['close']
                        exit_reason = "Donchian Exit (10d Low)"
                        
                    futures_exit_price = spot_exit_ref * (1.0 - slippage_rate)
                    exit_fee = (trade_obj["contracts_qty"] * futures_exit_price) * taker_fee_rate
                    tot_fees = trade_obj["entry_fee_usd"] + exit_fee
                    
                    entry_fut_p = trade_obj["futures_entry_price"]
                    price_change_pct = ((futures_exit_price - entry_fut_p) / entry_fut_p)
                    gross_pnl = price_change_pct * margin_base
                    net_pnl = gross_pnl - tot_fees
                    net_pnl_pct = (net_pnl / margin_base) * 100.0
                    
                    trade_obj.update({
                        "exit_timestamp": f"{curr_date} 23:59:59 UTC",
                        "spot_exit_reference": spot_exit_ref,
                        "futures_exit_price": round(futures_exit_price, 2),
                        "days_in_trade": days_in,
                        "exit_reason": exit_reason,
                        "fees": round(tot_fees, 2),
                        "slippage": round(margin_base * slippage_rate * 2.0, 2),
                        "gross_pnl": round(gross_pnl, 2),
                        "net_pnl": round(net_pnl, 2),
                        "net_pnl_pct": round(net_pnl_pct, 2)
                    })
                    all_trades.append(trade_obj)
                    in_trade = False
                    trade_obj = {}
                elif days_in == 3 and trade_obj["mfe_spot_d3_pct"] < FROZEN_PARAMS["mfe_threshold_pct"]:
                    trade_obj["early_prune_pending"] = True
                    
    return all_trades

def main():
    print("==========================================================================")
    print(" DIAGNÓSTICO METODOLÓGICO: AQUECIMENTO DA EMA200 (INDEX=30 VS INDEX=200)")
    print("==========================================================================")
    
    # 1. Simulação com Loop iniciando em index=30 (sem pré-aquecimento extra)
    trades_idx30 = run_simulation_with_warmup(start_index=30, pre_warmup_days=0)
    metrics_idx30 = calculate_portfolio_metrics("Loop Index = 30", 1.0, trades_idx30)
    
    # 2. Simulação com Loop iniciando em index=200 (sem pré-aquecimento extra)
    trades_idx200 = run_simulation_with_warmup(start_index=200, pre_warmup_days=0)
    metrics_idx200 = calculate_portfolio_metrics("Loop Index = 200", 1.0, trades_idx200)

    # 3. Simulação com PRÉ-AQUECIMENTO REAL (250 dias antes de 2021-09-09, loop em index=30)
    trades_prewarmed = run_simulation_with_warmup(start_index=30, pre_warmup_days=250)
    metrics_prewarmed = calculate_portfolio_metrics("Com Pré-Aquecimento Real (250d prévios)", 1.0, trades_prewarmed)

    # PARTE 1: TABELA COMPARATIVA LADO A LADO
    df_comp = pd.DataFrame([metrics_idx30, metrics_idx200, metrics_prewarmed])
    cols_show = ["Cenário", "Nº Trades", "Capital Final ($)", "Retorno Acumulado (%)", "CAGR (%)", "Profit Factor", "Max Drawdown (%)", "Win Rate (%)"]
    print("\n--- TABELA COMPARATIVA DOS CENÁRIOS ---")
    print(df_comp[cols_show].to_string(index=False))

    # PARTE 2: TRADES EXCLUSIVOS DE CADA VERSÃO
    set_30 = {(t["symbol"], t["signal_timestamp"][:10]) for t in trades_idx30}
    set_200 = {(t["symbol"], t["signal_timestamp"][:10]) for t in trades_idx200}
    set_pre = {(t["symbol"], t["signal_timestamp"][:10]) for t in trades_prewarmed}

    exclusive_in_30 = set_30 - set_200
    
    print("\n--- TRADES EXCLUSIVOS DO LOOP INDEX=30 (EM RELAÇÃO AO INDEX=200) ---")
    print(f"Quantidade de trades exclusivos: {len(exclusive_in_30)}")
    for sym, date_str in sorted(list(exclusive_in_30)):
        t_match = [t for t in trades_idx30 if t["symbol"] == sym and t["signal_timestamp"][:10] == date_str][0]
        print(f" Ativo: {sym:<8} | Data Entrada: {date_str} | Saída: {t_match['exit_reason']:<30} | Net PnL: ${t_match['net_pnl']:+.2f}")

    # PARTE 3: ANÁLISE DETALHADA DOS 3 TRADES DE OUTUBRO DE 2021
    oct_2021_trades = [t for t in trades_idx30 if t["signal_timestamp"][:7] == "2021-10"]
    print("\n--- ANÁLISE DETALHADA DOS 3 TRADES DE OUTUBRO DE 2021 (INDEX=30) ---")
    for idx_t, t in enumerate(oct_2021_trades, 1):
        print(f"\n[Trade Outubro/2021 #{idx_t}]")
        print(f" Ativo: {t['symbol']}")
        print(f" Data Sinal / Entrada: {t['signal_timestamp'][:10]}")
        print(f" Preço de Fechamento (Sinal): ${t['close_at_signal']}")
        print(f" Valor da EMA200 no Sinal: ${t['ema_200_at_signal']}")
        print(f" Condição Tendência (Close > EMA200): {t['close_at_signal'] > t['ema_200_at_signal']} (${t['close_at_signal']} > ${t['ema_200_at_signal']})")
        print(f" Candles de Histórico Disponíveis para EMA: {t['candles_warmup_available']} velas")
        print(f" Donchian High 30: ${t['donchian_high_30_at_signal']} | ADX 14: {t['adx_14_at_signal']} | Vol SMA 20: {t['vol_sma_20_at_signal']}")
        print(f" Data Saída: {t['exit_timestamp'][:10]} | Motivo: {t['exit_reason']} | PnL: ${t['net_pnl']:+.2f} ({t['net_pnl_pct']:+.2f}%)")
        
        exists_in_200 = (t["symbol"], t["signal_timestamp"][:10]) in set_200
        print(f" Existiria no Loop Index=200? {'SIM' if exists_in_200 else 'NÃO (Ignorado porque ocorreu antes do candle 200 - Março/2022)'}")

    # PARTE 4: TESTE COM PRÉ-AQUECIMENTO REAL (DADOS ANTERIORES A 09/09/2021)
    print("\n==========================================================================")
    print(" ANÁLISE DO PRÉ-AQUECIMENTO REAL COM DADOS ANTERIORES A 09/09/2021")
    print("==========================================================================")
    print(f"Trades com pré-aquecimento real de 250 dias: {len(trades_prewarmed)} trades")
    
    oct_2021_prewarmed = [t for t in trades_prewarmed if t["signal_timestamp"][:7] == "2021-10"]
    print(f"Trades disparados em Outubro/2021 COM PRÉ-AQUECIMENTO REAL: {len(oct_2021_prewarmed)} trades")
    
    for t_pre in oct_2021_prewarmed:
        print(f" -> {t_pre['symbol']} | Data: {t_pre['signal_timestamp'][:10]} | EMA200 com Pré-aquecimento Real: ${t_pre['ema_200_at_signal']} | Close: ${t_pre['close_at_signal']} | Close > EMA200? {t_pre['close_at_signal'] > t_pre['ema_200_at_signal']} | Net PnL: ${t_pre['net_pnl']:+.2f}")

if __name__ == '__main__':
    main()
