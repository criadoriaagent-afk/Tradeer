import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import fetch_historical_data

def prepare_data(symbol: str, days: int = 1825) -> pd.DataFrame:
    raw_df = fetch_historical_data(symbol, timeframe="1d", days=days)
    df = raw_df.copy().reset_index()
    date_col = 'Date' if 'Date' in df.columns else ('index' if 'index' in df.columns else df.columns[0])
    df['date'] = df[date_col]
    
    # Zero Look-Ahead (.shift(1))
    df['donchian_high_30'] = df['high'].shift(1).rolling(window=30).max()
    df['donchian_low_10'] = df['low'].shift(1).rolling(window=10).min()
    df['ema_200'] = df['close'].shift(1).ewm(span=200, adjust=False).mean()
    
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

# ========================================================
# 1. AUDITORIA COMPLETA DO FUNIL DE SINAIS
# ========================================================
def audit_signal_funnel(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'], days=1825):
    funnel_summary = {
        'total_raw_breakouts': 0,
        'approved_by_filters': 0,
        'rejected_by_ema': 0,
        'rejected_by_adx': 0,
        'rejected_by_vol': 0,
        'executed_trades': 0,
        'blocked_by_open_position': 0
    }
    
    by_symbol_funnel = []
    
    for symbol in symbols:
        df = prepare_data(symbol, days)
        in_position = False
        
        raw_cnt = 0
        app_cnt = 0
        rej_ema = 0
        rej_adx = 0
        rej_vol = 0
        exec_cnt = 0
        blocked_pos = 0
        
        for i in range(200, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            c_donchian = prev['close'] >= prev['donchian_high_30']
            
            if c_donchian:
                raw_cnt += 1
                c_trend = prev['close'] > prev['ema_200']
                c_adx = prev['adx_14'] >= 20
                c_vol = prev['volume'] >= prev['vol_sma_20']
                
                if not c_trend:
                    rej_ema += 1
                if not c_adx:
                    rej_adx += 1
                if not c_vol:
                    rej_vol += 1
                    
                passed_all = c_trend and c_adx and c_vol
                if passed_all:
                    app_cnt += 1
                    if in_position:
                        blocked_pos += 1
                    else:
                        in_position = True
                        exec_cnt += 1
            
            # Gerenciar saída simples para manter rastreamento de in_position
            if in_position:
                stop_loss = row['open'] - (2.0 * prev['atr_14']) # aproximação para atualização de posição
                hit_sl = row['low'] <= stop_loss
                hit_exit = row['close'] < prev['donchian_low_10']
                if hit_sl or hit_exit:
                    in_position = False
                    
        by_symbol_funnel.append({
            'symbol': symbol,
            'raw_breakouts': raw_cnt,
            'approved_filters': app_cnt,
            'rejected_ema': rej_ema,
            'rejected_adx': rej_adx,
            'rejected_vol': rej_vol,
            'executed_trades': exec_cnt,
            'blocked_open_pos': blocked_pos
        })
        
        funnel_summary['total_raw_breakouts'] += raw_cnt
        funnel_summary['approved_by_filters'] += app_cnt
        funnel_summary['rejected_by_ema'] += rej_ema
        funnel_summary['rejected_by_adx'] += rej_adx
        funnel_summary['rejected_by_vol'] += rej_vol
        funnel_summary['executed_trades'] += exec_cnt
        funnel_summary['blocked_by_open_position'] += blocked_pos
        
    return funnel_summary, pd.DataFrame(by_symbol_funnel)

# ========================================================
# 2. SIMULAÇÃO RIGOROSA DO EARLY-PRUNE NO OPEN DO DIA 4
# ========================================================
def run_rigorous_early_prune_sim(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'], days=1825):
    sizing = 333.33
    fee_rate = 0.0015
    
    baseline_trades = []
    pruned_trades = []
    
    for symbol in symbols:
        df = prepare_data(symbol, days)
        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        entry_idx = 0
        
        for i in range(200, len(df) - 4):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            c_trend = prev['close'] > prev['ema_200']
            c_donchian = prev['close'] >= prev['donchian_high_30']
            c_adx = prev['adx_14'] >= 20
            c_vol = prev['volume'] >= prev['vol_sma_20']
            entry_signal = c_trend and c_donchian and c_adx and c_vol
            
            if not in_position:
                if entry_signal:
                    in_position = True
                    entry_price = row['open']
                    atr = prev['atr_14']
                    stop_loss = entry_price - (2.0 * atr)
                    entry_idx = i
            else:
                current_day = (i - entry_idx) + 1
                hit_sl = row['low'] <= stop_loss
                hit_exit = row['close'] < prev['donchian_low_10']
                
                if hit_sl or hit_exit:
                    exit_idx = i
                    exit_price = stop_loss if hit_sl else row['close']
                    exit_reason = "SL" if hit_sl else "DonchianLow10"
                    
                    raw_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    pnl_sizing_pct = raw_pct - (fee_rate * 2 * 100.0)
                    pnl_usd = (pnl_sizing_pct / 100.0) * sizing
                    
                    # Calcular MFE nos dias 1, 2 e 3
                    w1 = df.iloc[entry_idx]
                    w2 = df.iloc[min(entry_idx+1, len(df)-1)]
                    w3 = df.iloc[min(entry_idx+2, len(df)-1)]
                    
                    mfe_d1 = ((w1['high'] - entry_price) / entry_price) * 100.0
                    mfe_d2 = ((max(w1['high'], w2['high']) - entry_price) / entry_price) * 100.0
                    mfe_d3 = ((max(w1['high'], w2['high'], w3['high']) - entry_price) / entry_price) * 100.0
                    
                    # Retornos prospectivos em 5d, 10d, 20d a partir do Open de entrada
                    fwd_5d = ((df.iloc[min(entry_idx+4, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                    fwd_10d = ((df.iloc[min(entry_idx+9, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                    fwd_20d = ((df.iloc[min(entry_idx+19, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                    
                    # Risco Inicial R (em USD)
                    risk_r_usd = (2.0 * atr / entry_price) * sizing
                    
                    trade_obj = {
                        'symbol': symbol,
                        'entry_date': str(df.iloc[entry_idx]['date'])[:10],
                        'exit_date': str(row['date'])[:10],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_usd': pnl_usd,
                        'pnl_pct': pnl_sizing_pct,
                        'risk_r_usd': risk_r_usd,
                        'pnl_r_multiple': pnl_usd / risk_r_usd if risk_r_usd > 0 else 0,
                        'mfe_d1': mfe_d1,
                        'mfe_d2': mfe_d2,
                        'mfe_d3': mfe_d3,
                        'fwd_5d': fwd_5d,
                        'fwd_10d': fwd_10d,
                        'fwd_20d': fwd_20d,
                        'duracao_dias': exit_idx - entry_idx
                    }
                    
                    baseline_trades.append(trade_obj)
                    
                    # Verificar se a regra de Early Prune no Open do Dia 4 seria ativada
                    # Condição: No fechamento do Dia 3 (entry_idx + 2), se mfe_d3 < 1.0%, sai no Open do Dia 4 (entry_idx + 3)
                    if mfe_d3 < 1.0 and (exit_idx >= entry_idx + 3):
                        open_day4 = df.iloc[entry_idx + 3]['open']
                        # Verificar se Stop Loss ocorreu antes do Open do Dia 4
                        low_d1 = w1['low']
                        low_d2 = w2['low']
                        low_d3 = w3['low']
                        
                        if low_d1 <= stop_loss or low_d2 <= stop_loss or low_d3 <= stop_loss:
                            # Stop Loss ocorreu nos primeiros 3 dias, prevalece a saída pelo Stop
                            early_trade = trade_obj.copy()
                        else:
                            # Saída pelo Early Prune no Open do Dia 4
                            early_exit_price = open_day4
                            early_raw_pct = ((early_exit_price - entry_price) / entry_price) * 100.0
                            early_pnl_pct = early_raw_pct - (fee_rate * 2 * 100.0)
                            early_pnl_usd = (early_pnl_pct / 100.0) * sizing
                            
                            early_trade = trade_obj.copy()
                            early_trade['exit_date'] = str(df.iloc[entry_idx + 3]['date'])[:10]
                            early_trade['exit_price'] = early_exit_price
                            early_trade['exit_reason'] = "EarlyPrune_Day4Open"
                            early_trade['pnl_usd'] = early_pnl_usd
                            early_trade['pnl_pct'] = early_pnl_pct
                            early_trade['pnl_r_multiple'] = early_pnl_usd / risk_r_usd if risk_r_usd > 0 else 0
                            early_trade['duracao_dias'] = 3
                            
                        pruned_trades.append(early_trade)
                    else:
                        pruned_trades.append(trade_obj.copy())
                        
                    in_position = False
                    
    return pd.DataFrame(baseline_trades), pd.DataFrame(pruned_trades)

# ========================================================
# 3. MATRIZ DE ABLATION LEAVE-ONE-OUT DE FILTROS
# ========================================================
def run_clean_ablation(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'], days=1825):
    cases = [
        ('Teste A: EMA200 + Donchian30', True, False, False),
        ('Teste B: EMA200 + Donchian30 + ADX', True, True, False),
        ('Teste C: EMA200 + Donchian30 + Volume', True, False, True),
        ('Teste D: V1 Completa (EMA200 + Donchian30 + ADX + Volume)', True, True, True)
    ]
    
    rows = []
    sizing = 333.33
    fee_rate = 0.0015
    
    for label, use_ema, use_adx, use_vol in cases:
        trades = []
        for symbol in symbols:
            df = prepare_data(symbol, days)
            in_position = False
            entry_price = 0.0
            stop_loss = 0.0
            
            for i in range(200, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                
                c_donchian = prev['close'] >= prev['donchian_high_30']
                c_trend = prev['close'] > prev['ema_200'] if use_ema else True
                c_adx = prev['adx_14'] >= 20 if use_adx else True
                c_vol = prev['volume'] >= prev['vol_sma_20'] if use_vol else True
                
                if not in_position:
                    if c_donchian and c_trend and c_adx and c_vol:
                        in_position = True
                        entry_price = row['open']
                        stop_loss = entry_price - (2.0 * prev['atr_14'])
                else:
                    hit_sl = row['low'] <= stop_loss
                    hit_exit = row['close'] < prev['donchian_low_10']
                    if hit_sl or hit_exit:
                        exit_price = stop_loss if hit_sl else row['close']
                        raw_pct = ((exit_price - entry_price) / entry_price) * 100.0
                        pnl_pct = raw_pct - (fee_rate * 2 * 100.0)
                        pnl_usd = (pnl_pct / 100.0) * sizing
                        trades.append({'pnl_usd': pnl_usd, 'pnl_pct': pnl_pct})
                        in_position = False
                        
        tdf = pd.DataFrame(trades)
        wins = tdf[tdf['pnl_usd'] > 0]
        losers = tdf[tdf['pnl_usd'] <= 0]
        
        pf = wins['pnl_usd'].sum() / abs(losers['pnl_usd'].sum()) if len(losers) > 0 else np.nan
        total_pnl = tdf['pnl_usd'].sum()
        win_rate = len(wins) / len(tdf) * 100.0
        exp_sizing = total_pnl / (len(tdf) * sizing) * 100.0
        
        rows.append({
            'hipotese': label,
            'n_trades': len(tdf),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(pf, 2),
            'expectancy_sizing_pct': round(exp_sizing, 2),
            'pnl_usd': round(total_pnl, 2)
        })
        
    return pd.DataFrame(rows)

def execute_validation_suite():
    print("========================================================")
    print(" VALIDATOR RIGOROSO DE HYPOTHESIS & AUDITORIA DE FUNIL")
    print("========================================================")
    
    # 1. Funil de Sinais
    print("\n[1/5] Auditando Funil de Sinais (251 Brutos -> 153 Aprovados -> 41 Executados)...")
    funnel_tot, funnel_df = audit_signal_funnel()
    
    print("\n--------------------------------------------------------")
    print(" 1. TABELA DE DETALHAMENTO DO FUNIL DE SINAIS")
    print("--------------------------------------------------------")
    print(funnel_df.to_string(index=False))
    print(f"\n RESUMO GLOBAL DO FUNIL:")
    print(f"  - Rompimentos Brutos (Close >= DonchianHigh30): {funnel_tot['total_raw_breakouts']}")
    print(f"  - Aprovados por Todos os Filtros: {funnel_tot['approved_by_filters']} ({funnel_tot['approved_by_filters']/funnel_tot['total_raw_breakouts']*100:.1f}%)")
    print(f"  - Rejeitados por EMA200: {funnel_tot['rejected_by_ema']}")
    print(f"  - Rejeitados por ADX < 20: {funnel_tot['rejected_by_adx']}")
    print(f"  - Rejeitados por Vol < SMA20: {funnel_tot['rejected_by_vol']}")
    print(f"  - Trades Realmente Executados: {funnel_tot['executed_trades']}")
    print(f"  - Sinais Aprovados Bloqueados por Posição Já Aberta: {funnel_tot['blocked_by_open_position']}")

    # 2. Simulação Rigorosa de Early-Prune no Open do Dia 4
    print("\n[2/5] Simulação Rigorosa sem Look-Ahead (MFE 3d < 1% -> Saída no Open do Dia 4)...")
    base_df, prune_df = run_rigorous_early_prune_sim()
    
    print("\n--------------------------------------------------------")
    print(" 2. COMPARATIVO BASELINE VS CANDIDATE (OPEN DO DIA 4)")
    print("--------------------------------------------------------")
    b_wins = base_df[base_df['pnl_usd'] > 0]
    b_losers = base_df[base_df['pnl_usd'] <= 0]
    b_pf = b_wins['pnl_usd'].sum() / abs(b_losers['pnl_usd'].sum())
    b_pnl = base_df['pnl_usd'].sum()
    
    p_wins = prune_df[prune_df['pnl_usd'] > 0]
    p_losers = prune_df[prune_df['pnl_usd'] <= 0]
    p_pf = p_wins['pnl_usd'].sum() / abs(p_losers['pnl_usd'].sum())
    p_pnl = prune_df['pnl_usd'].sum()
    
    print(f" BASELINE V1.0: PnL Total = ${b_pnl:+.2f} | PF = {b_pf:.2f} | Win Rate = {len(b_wins)/len(base_df)*100:.1f}% ({len(base_df)} trades)")
    print(f" CANDIDATE MFE3d_Day4Open: PnL Total = ${p_pnl:+.2f} | PF = {p_pf:.2f} | Win Rate = {len(p_wins)/len(prune_df)*100:.1f}% ({len(prune_df)} trades)")

    # 3. Autópsia de TODOS os Trades Afetados pelo Early Exit
    print("\n--------------------------------------------------------")
    print(" 3. DETALHAMENTO DE TODOS OS TRADES AFETADOS PELO EARLY EXIT")
    print("--------------------------------------------------------")
    pruned_only = prune_df[prune_df['exit_reason'] == 'EarlyPrune_Day4Open']
    
    print(f" Total de Trades Encerrados no Open do Dia 4: {len(pruned_only)}")
    hdr_p = f" {'Simbolo':<8} | {'Entrada':<10} | {'MFE 3d(%)':<9} | {'PnL Orig($)':<11} | {'PnL Early($)':<12} | {'Fwd 5d(%)':<9} | {'Fwd 10d(%)':<10} | {'Fwd 20d(%)':<10} | {'Motivo Orig':<13}"
    print(hdr_p)
    print("-" * len(hdr_p))
    
    for idx, r in pruned_only.iterrows():
        # Encontrar original no base_df
        orig = base_df[(base_df['symbol'] == r['symbol']) & (base_df['entry_date'] == r['entry_date'])].iloc[0]
        print(f" {r['symbol']:<8} | {r['entry_date']:<10} | {r['mfe_d3']:>+8.2f}% | ${orig['pnl_usd']:>+10.2f} | ${r['pnl_usd']:>+11.2f} | {r['fwd_5d']:>+8.2f}% | {r['fwd_10d']:>+9.2f}% | {r['fwd_20d']:>+9.2f}% | {orig['exit_reason']:<13}")

    # 4. Autópsia de TODOS os 15 Vencedores da Baseline
    print("\n--------------------------------------------------------")
    print(" 4. AUTÓPSIA MFE DIAS 1, 2, 3 DE TODOS OS 15 VENCEDORES DA BASELINE")
    print("--------------------------------------------------------")
    hdr_w = f" {'#':<2} | {'Simbolo':<8} | {'Entrada':<10} | {'PnL Baseline($)':<15} | {'MFE Dia 1(%)':<12} | {'MFE Dia 2(%)':<12} | {'MFE Dia 3(%)':<12} | {'Foi Cortado?'}"
    print(hdr_w)
    print("-" * len(hdr_w))
    
    cut_wins_count = 0
    for idx, r in b_wins.reset_index().iterrows():
        is_cut = r['mfe_d3'] < 1.0
        if is_cut: cut_wins_count += 1
        cut_str = "SIM (AFETADO)" if is_cut else "NÃO (Mantido)"
        print(f" {idx+1:<2} | {r['symbol']:<8} | {r['entry_date']:<10} | ${r['pnl_usd']:>+14.2f} | {r['mfe_d1']:>+11.2f}% | {r['mfe_d2']:>+11.2f}% | {r['mfe_d3']:>+11.2f}% | {cut_str}")
        
    print(f"\n Total de Vencedores Cortados pela Regra MFE 3d < 1%: {cut_wins_count} de {len(b_wins)}")

    # 5. Formatação Matemática da Expectancy
    print("\n--------------------------------------------------------")
    print(" 5. PADRONIZAÇÃO DAS FÓRMULAS DE EXPECTANCY")
    print("--------------------------------------------------------")
    exp_sizing = b_pnl / (len(base_df) * 333.33) * 100.0
    exp_wallet = b_pnl / (len(base_df) * 1000.0) * 100.0
    mean_r_mult = base_df['pnl_r_multiple'].mean()
    
    print(f" 1. Expectancy sobre Sizing ($333.33): {exp_sizing:+.2f}% por trade")
    print(f" 2. Expectancy sobre Carteira Total ($1,000.00): {exp_wallet:+.2f}% por trade")
    print(f" 3. Expectancy em Múltiplos de Risco R (2x ATR): {mean_r_mult:+.2f}R por trade")

    # 6. Matrix Clean Leave-One-Out
    print("\n--------------------------------------------------------")
    print(" 6. MATRIZ DE ABLATION LEAVE-ONE-OUT ISOLADA")
    print("--------------------------------------------------------")
    ablation_res = run_clean_ablation()
    print(ablation_res.to_string(index=False))

if __name__ == "__main__":
    execute_validation_suite()
