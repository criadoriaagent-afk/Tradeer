import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import fetch_historical_data

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
JSON_FILE = os.path.join(DATA_DIR, 'prospective_paper_trading.json')

def prepare_data(symbol: str, days: int = 1825) -> pd.DataFrame:
    raw_df = fetch_historical_data(symbol, timeframe="1d", days=days)
    df = raw_df.copy().reset_index()
    date_col = 'Date' if 'Date' in df.columns else ('index' if 'index' in df.columns else df.columns[0])
    df['date'] = df[date_col]
    
    # Zero Look-Ahead (.shift(1))
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
    
    # Volume SMA 20
    df['vol_sma_20'] = df['volume'].shift(1).rolling(window=20).mean()
    
    return df

def run_parallel_simulation(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'], days=1825):
    sizing = 333.33
    fee_rate = 0.0015
    initial_capital_per_asset = 1000.0
    
    detailed_trades = []
    
    for symbol in symbols:
        df = prepare_data(symbol, days)
        in_pos_v1 = False
        in_pos_v2 = False
        in_pos_ep = False
        
        # Estruturas independentes para simulação paralela estrita
        # 1. Baseline V1.0
        v1_entry_idx, v1_entry_price, v1_sl = 0, 0.0, 0.0
        # 2. V2.0 Prototype
        v2_entry_idx, v2_entry_price, v2_sl = 0, 0.0, 0.0
        # 3. EARLY_PRUNE_V1
        ep_entry_idx, ep_entry_price, ep_sl = 0, 0.0, 0.0
        
        for i in range(200, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            c_trend = prev['close'] > prev['ema_200']
            c_donchian = prev['close'] >= prev['donchian_high_30']
            c_adx = prev['adx_14'] >= 20
            c_vol = prev['volume'] >= prev['vol_sma_20']
            entry_signal = c_trend and c_donchian and c_adx and c_vol
            
            # ----------------------------------------------------
            # RASTREAMENTO PARALELO DA V1.0 (Donchian 10d Exit)
            # ----------------------------------------------------
            if not in_pos_v1:
                if entry_signal:
                    in_pos_v1 = True
                    v1_entry_price = row['open']
                    v1_sl = v1_entry_price - (2.0 * prev['atr_14'])
                    v1_entry_idx = i
            else:
                hit_sl = row['low'] <= v1_sl
                hit_exit = row['close'] < prev['donchian_low_10']
                if hit_sl or hit_exit:
                    exit_price = v1_sl if hit_sl else row['close']
                    exit_reason = "SL" if hit_sl else "DonchianLow10"
                    
                    raw_pct = ((exit_price - v1_entry_price) / v1_entry_price) * 100.0
                    pnl_sizing_pct = raw_pct - (fee_rate * 2 * 100.0)
                    pnl_usd = (pnl_sizing_pct / 100.0) * sizing
                    
                    w1 = df.iloc[v1_entry_idx]
                    w2 = df.iloc[min(v1_entry_idx+1, len(df)-1)]
                    w3 = df.iloc[min(v1_entry_idx+2, len(df)-1)]
                    
                    mfe_d1 = ((w1['high'] - v1_entry_price) / v1_entry_price) * 100.0
                    mfe_d2 = ((max(w1['high'], w2['high']) - v1_entry_price) / v1_entry_price) * 100.0
                    mfe_d3 = ((max(w1['high'], w2['high'], w3['high']) - v1_entry_price) / v1_entry_price) * 100.0
                    
                    # Calcular resultado correspondente se estivesse com Early Prune no Open do Dia 4
                    early_prune_triggered = False
                    ep_pnl_usd = pnl_usd
                    ep_pnl_pct = pnl_sizing_pct
                    ep_exit_reason = exit_reason
                    ep_exit_price = exit_price
                    
                    if mfe_d3 < 1.0 and (i >= v1_entry_idx + 3):
                        # Verificar se SL ocorreu nos primeiros 3 dias
                        if not (w1['low'] <= v1_sl or w2['low'] <= v1_sl or w3['low'] <= v1_sl):
                            early_prune_triggered = True
                            ep_exit_price = df.iloc[v1_entry_idx + 3]['open']
                            ep_raw_pct = ((ep_exit_price - v1_entry_price) / v1_entry_price) * 100.0
                            ep_pnl_pct = ep_raw_pct - (fee_rate * 2 * 100.0)
                            ep_pnl_usd = (ep_pnl_pct / 100.0) * sizing
                            ep_exit_reason = "EarlyPrune_Day4Open"
                            
                    # Calcular resultado correspondente na V2.0 (EMA 20 exit)
                    # Simular saída V2 na mesma janela
                    v2_pnl_usd = pnl_usd
                    v2_pnl_pct = pnl_sizing_pct
                    v2_exit_reason = exit_reason
                    v2_exit_price = exit_price
                    
                    # Procurar saída real V2
                    for k in range(v1_entry_idx, len(df)):
                        k_row = df.iloc[k]
                        k_prev = df.iloc[k-1]
                        k_sl = v1_sl
                        k_hit_sl = k_row['low'] <= k_sl
                        k_hit_ema20 = k_row['close'] < k_prev['ema_20']
                        if k_hit_sl or k_hit_ema20:
                            v2_exit_price = k_sl if k_hit_sl else k_row['close']
                            v2_exit_reason = "SL" if k_hit_sl else "EMA20"
                            v2_raw = ((v2_exit_price - v1_entry_price) / v1_entry_price) * 100.0
                            v2_pnl_pct = v2_raw - (fee_rate * 2 * 100.0)
                            v2_pnl_usd = (v2_pnl_pct / 100.0) * sizing
                            break
                            
                    detailed_trades.append({
                        'trade_id': len(detailed_trades) + 1,
                        'symbol': symbol,
                        'entry_date': str(df.iloc[v1_entry_idx]['date'])[:10],
                        'exit_date_v1': str(row['date'])[:10],
                        'entry_price': round(v1_entry_price, 2),
                        'stop_loss_initial': round(v1_sl, 2),
                        'mfe_d1_pct': round(mfe_d1, 2),
                        'mfe_d2_pct': round(mfe_d2, 2),
                        'mfe_d3_pct': round(mfe_d3, 2),
                        # V1.0 Metrics
                        'pnl_usd_v1': round(pnl_usd, 2),
                        'pnl_pct_v1': round(pnl_sizing_pct, 2),
                        'exit_reason_v1': exit_reason,
                        # V2.0 Metrics
                        'pnl_usd_v2': round(v2_pnl_usd, 2),
                        'pnl_pct_v2': round(v2_pnl_pct, 2),
                        'exit_reason_v2': v2_exit_reason,
                        # EARLY_PRUNE_V1 Metrics
                        'early_prune_triggered': early_prune_triggered,
                        'pnl_usd_early_prune': round(ep_pnl_usd, 2),
                        'pnl_pct_early_prune': round(ep_pnl_pct, 2),
                        'exit_reason_early_prune': ep_exit_reason,
                        'duracao_dias_v1': i - v1_entry_idx
                    })
                    in_pos_v1 = False
                    
    tdf = pd.DataFrame(detailed_trades)
    return tdf

def compute_metrics_summary(tdf: pd.DataFrame):
    sizing = 333.33
    initial_cap_total = 3000.0 # 3 ativos x $1,000
    
    strategies = [
        ('V1.0 1D Simplificada (Donchian 10d Exit)', 'pnl_usd_v1', 'pnl_pct_v1'),
        ('V2.0 Prototype (EMA 20 Exit)', 'pnl_usd_v2', 'pnl_pct_v2'),
        ('EARLY_PRUNE_V1 (Candidata Congelada)', 'pnl_usd_early_prune', 'pnl_pct_early_prune')
    ]
    
    summary = []
    
    # Calcular baseline PnLs para comparação de perdas/lucro sacrificado
    b_wins = tdf[tdf['pnl_usd_v1'] > 0]
    b_losers = tdf[tdf['pnl_usd_v1'] <= 0]
    
    for name, pnl_col, pct_col in strategies:
        tot_pnl = tdf[pnl_col].sum()
        tot_ret_pct = (tot_pnl / initial_cap_total) * 100.0
        
        # CAGR (5 anos = 5.0)
        final_wallet = initial_cap_total + tot_pnl
        cagr = ((final_wallet / initial_cap_total) ** (1/5.0) - 1) * 100.0
        
        wins = tdf[tdf[pnl_col] > 0]
        losers = tdf[tdf[pnl_col] <= 0]
        
        gross_p = wins[pnl_col].sum() if len(wins) > 0 else 0.0
        gross_l = abs(losers[pnl_col].sum()) if len(losers) > 0 else 1.0
        pf = gross_p / gross_l if gross_l > 0 else np.nan
        
        win_rate = len(wins) / len(tdf) * 100.0 if len(tdf) > 0 else 0.0
        exp_sizing = tdf[pct_col].mean()
        
        # Max Drawdown aproximado da série de trades
        cum_equity = initial_cap_total + tdf[pnl_col].cumsum()
        running_max = cum_equity.cummax()
        drawdowns = (cum_equity - running_max) / running_max * 100.0
        max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
        
        # Métrica Principal: CAGR / Max DD
        cagr_over_max_dd = cagr / max_dd if max_dd > 0 else np.nan
        
        # Perdas Evitadas vs Lucro Sacrificado
        if name.startswith('EARLY_PRUNE'):
            pruned_mask = tdf['early_prune_triggered']
            # Perdas evitadas em perdedores originais
            saved_mask = pruned_mask & (tdf['pnl_usd_v1'] <= 0)
            losses_saved = (tdf[saved_mask]['pnl_usd_early_prune'] - tdf[saved_mask]['pnl_usd_v1']).sum()
            
            # Lucro sacrificado em vencedores originais
            sacrificed_mask = pruned_mask & (tdf['pnl_usd_v1'] > 0)
            profit_sacrificed = (tdf[sacrificed_mask]['pnl_usd_v1'] - tdf[sacrificed_mask]['pnl_usd_early_prune']).sum()
            sacrificed_winners_cnt = (sacrificed_mask & (tdf['pnl_usd_early_prune'] < tdf['pnl_usd_v1'])).sum()
        else:
            losses_saved = 0.0
            profit_sacrificed = 0.0
            sacrificed_winners_cnt = 0
            
        summary.append({
            'Estratégia': name,
            'Nº Trades': len(tdf),
            'Win Rate (%)': round(win_rate, 1),
            'PnL Total ($)': round(tot_pnl, 2),
            'Retorno Carteira (%)': round(tot_ret_pct, 2),
            'CAGR (%)': round(cagr, 2),
            'Max Drawdown (%)': round(max_dd, 2),
            'CAGR / Max DD (Principal)': round(cagr_over_max_dd, 2),
            'Profit Factor': round(pf, 2),
            'Expectancy Sizing (%)': round(exp_sizing, 2),
            'Perdas Evitadas ($)': round(losses_saved, 2),
            'Lucro Sacrificado ($)': round(profit_sacrificed, 2),
            'Vencedores Sacrificados': sacrificed_winners_cnt
        })
        
    return pd.DataFrame(summary)

def execute_prospective_suite():
    print("========================================================")
    print(" SUÍTE DE PAPER TRADING PROSPECTIVO EM PARALELO")
    print("========================================================")
    
    tdf = run_parallel_simulation()
    summary_df = compute_metrics_summary(tdf)
    
    print("\n========================================================")
    print(" 1. PAINEL COMPARATIVO DE MÉTRICAS (V1.0 VS V2.0 VS EARLY_PRUNE_V1)")
    print("========================================================")
    print(summary_df.to_string(index=False))
    
    # Salvar a persistência em JSON no diretório data/
    os.makedirs(DATA_DIR, exist_ok=True)
    json_data = {
        'status': 'HIPÓTESE CONGELADA PENDENTE DE VALIDAÇÃO PROSPECTIVA',
        'candidate_id': 'EARLY_PRUNE_V1',
        'params': {
            'mfe_threshold_pct': 1.0,
            'observation_days': 3,
            'execution_timing': 'Day 4 Open',
            'sizing_per_trade_usd': 333.33,
            'fee_rate_pct': 0.15
        },
        'summary': summary_df.to_dict(orient='records'),
        'trades': tdf.to_dict(orient='records')
    }
    
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
        
    print("\n========================================================")
    print(f" Structure de persistência gravada com sucesso em:")
    print(f" {JSON_FILE}")
    print("========================================================")

if __name__ == "__main__":
    execute_prospective_suite()
