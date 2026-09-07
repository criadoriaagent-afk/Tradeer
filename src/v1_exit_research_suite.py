import sys
import os
import pandas as pd
import numpy as np

# Adicionar src ao PATH
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import fetch_historical_data

def run_v1_exit_audit():
    # Carregar 5 anos (1825 dias) de BTC-USD em timeframe diário (1d)
    df = fetch_historical_data("BTC-USD", timeframe="1d", days=1825)
    if df is None or df.empty:
        print("[ERRO] Falha ao carregar dados do BTC-USD.")
        return

    df = df.reset_index()
    if 'Date' in df.columns:
        df['date'] = df['Date']
    elif 'index' in df.columns:
        df['date'] = df['index']
    else:
        df['date'] = df.index

    # 1. Recalcular indicadores da V1.0 congelada
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['donchian_high_30'] = df['high'].shift(1).rolling(window=30).max()
    df['donchian_low_30'] = df['low'].shift(1).rolling(window=30).min()
    df['donchian_low_20'] = df['low'].shift(1).rolling(window=20).min()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # ATR 14
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()

    # ADX 14 simplificado para filtro V1.0
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['plus_di'] = 100 * (pd.Series(df['plus_dm']).rolling(14).mean() / df['atr_14'])
    df['minus_di'] = 100 * (pd.Series(df['minus_dm']).rolling(14).mean() / df['atr_14'])
    dx = 100 * (np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
    df['adx_14'] = dx.rolling(14).mean()

    # Volume SMA 20
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()

    # 2. Simular Trades da V1.0 Congelada para extrair logs detalhados
    trades = []
    in_position = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_idx = 0

    for i in range(200, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Filtros V1.0 Congelados
        c_trend = prev['close'] > prev['ema_200']
        c_donchian = prev['close'] >= prev['donchian_high_30']
        c_adx = prev['adx_14'] >= 20
        c_vol = prev['volume'] >= prev['vol_sma_20']
        
        if not in_position:
            if c_trend and c_donchian and c_adx and c_vol:
                in_position = True
                entry_price = row['open']
                atr = prev['atr_14']
                stop_loss = entry_price - (2.0 * atr)
                take_profit = entry_price + (2.5 * atr)
                entry_idx = i
        else:
            hit_sl = row['low'] <= stop_loss
            hit_tp = row['high'] >= take_profit
            
            if hit_sl or hit_tp:
                exit_price = stop_loss if hit_sl else take_profit
                exit_reason = "SL" if hit_sl else "TP"
                exit_idx = i
                
                # Janela de vida da posição
                pos_window = df.iloc[entry_idx : exit_idx + 1]
                max_high = pos_window['high'].max()
                min_low = pos_window['low'].min()
                
                mfe_pct = ((max_high - entry_price) / entry_price) * 100.0
                mae_pct = ((min_low - entry_price) / entry_price) * 100.0
                
                duracao = exit_idx - entry_idx
                idx_mfe = pos_window['high'].idxmax() - df.index[entry_idx]
                idx_mae = pos_window['low'].idxmin() - df.index[entry_idx]
                
                pnl_usd = (exit_price - entry_price) * (333.33 / entry_price) - (333.33 * 0.0015)
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0 - 0.15
                
                # Post-Exit Drift (Subsequentes 10 dias pós-saída)
                future_10 = df.iloc[exit_idx + 1 : min(len(df), exit_idx + 11)]
                post_max_10 = future_10['high'].max() if len(future_10) > 0 else exit_price
                post_drift_10_pct = ((post_max_10 - exit_price) / exit_price) * 100.0
                
                # Captura do MFE
                mfe_capture_pct = (pnl_pct / mfe_pct * 100.0) if mfe_pct > 0 else 0.0

                trades.append({
                    'entry_date': df.iloc[entry_idx]['date'],
                    'exit_date': row['date'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_usd': pnl_usd,
                    'pnl_pct': pnl_pct,
                    'mfe_pct': mfe_pct,
                    'mae_pct': mae_pct,
                    'mfe_capture_pct': mfe_capture_pct,
                    'duracao_dias': duracao,
                    'dias_ate_mfe': idx_mfe,
                    'dias_ate_mae': idx_mae,
                    'post_drift_10_pct': post_drift_10_pct,
                })
                in_position = False

    tdf = pd.DataFrame(trades)
    
    print("========================================================")
    print(" 1. AUDITORIA COMPLETA DO POST-EXIT DRIFT (15 VENCEDORES)")
    print("========================================================")
    winners = tdf[tdf['exit_reason'] == 'TP']
    losers = tdf[tdf['exit_reason'] == 'SL']
    
    drift_series = winners['post_drift_10_pct']
    drift_percentiles = {
        'Média': np.round(drift_series.mean(), 2),
        'Mediana (P50)': np.round(drift_series.median(), 2),
        'P25': np.round(drift_series.quantile(0.25), 2),
        'P75': np.round(drift_series.quantile(0.75), 2),
        'P90': np.round(drift_series.quantile(0.90), 2),
        'Máximo': np.round(drift_series.max(), 2)
    }
    print(f" Post-Exit Drift (10 dias pós-saída) nos Vencedores:")
    for k, v in drift_percentiles.items():
        print(f"   - {k}: +{v}%")
        
    num_continued = (winners['post_drift_10_pct'] > 1.0).sum()
    num_reversed = len(winners) - num_continued
    print(f"\n Vencedores que continuaram subindo pós-saída: {num_continued} de {len(winners)} ({num_continued/len(winners)*100:.1f}%)")
    print(f" Vencedores que reverteram/estagnaram pós-saída: {num_reversed} de {len(winners)} ({num_reversed/len(winners)*100:.1f}%)")

    print("\n========================================================")
    print(" 2. AUDITORIA DE MFE/MAE E CAPTURA DO SISTEMA")
    print("========================================================")
    for name, group in [("VENCEDORES (TP)", winners), ("PERDEDORES (SL)", losers), ("TOTAL GERAL", tdf)]:
        print(f"\n --- {name} [{len(group)} trades] ---")
        mfe_stats = {
            'Média': np.round(group['mfe_pct'].mean(), 2),
            'Mediana': np.round(group['mfe_pct'].median(), 2),
            'P25': np.round(group['mfe_pct'].quantile(0.25), 2),
            'P75': np.round(group['mfe_pct'].quantile(0.75), 2),
            'P90': np.round(group['mfe_pct'].quantile(0.90), 2),
            'Máximo': np.round(group['mfe_pct'].max(), 2)
        }
        mae_stats = {
            'Média': np.round(group['mae_pct'].mean(), 2),
            'Mediana': np.round(group['mae_pct'].median(), 2),
            'P25': np.round(group['mae_pct'].quantile(0.25), 2),
            'P75': np.round(group['mae_pct'].quantile(0.75), 2),
            'P90': np.round(group['mae_pct'].quantile(0.90), 2),
            'Máximo': np.round(group['mae_pct'].max(), 2)
        }
        cap_mean = np.round(group['mfe_capture_pct'].mean(), 2)
        print(f"  MFE %: {mfe_stats}")
        print(f"  MAE %: {mae_stats}")
        print(f"  Eficiência Média de Captura do MFE: {cap_mean}%")

    print("\n========================================================")
    print(" 3. INVESTIGAÇÃO DE DURAÇÃO E TEMPO ATÉ EVENTOS")
    print("========================================================")
    print(f" Duração Média Geral: {tdf['duracao_dias'].mean():.1f} dias | Mediana: {tdf['duracao_dias'].median():.1f} dias")
    print(f" Duração Média Vencedores (TP): {winners['duracao_dias'].mean():.1f} dias | Mediana: {winners['duracao_dias'].median():.1f} dias")
    print(f" Duração Média Perdedores (SL): {losers['duracao_dias'].mean():.1f} dias | Mediana: {losers['duracao_dias'].median():.1f} dias")
    print(f" Tempo Médio até Pico de MFE nos Perdedores: {losers['dias_ate_mfe'].mean():.1f} dias")
    print(f" Tempo Médio até Fundo de MAE nos Perdedores: {losers['dias_ate_mae'].mean():.1f} dias")

    print("\n========================================================")
    print(" 4. EXPERIMENTOS DE SAÍDA E TIME STOP (MANTENDO ENTRADAS V1)")
    print("========================================================")
    
    def run_exit_experiment(exit_type, param=None):
        t_list = []
        in_pos = False
        e_price = 0
        sl_price = 0
        tp_price = 0
        e_idx = 0
        highest_high = 0
        
        for i in range(200, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            c_trend = prev['close'] > prev['ema_200']
            c_donchian = prev['close'] >= prev['donchian_high_30']
            c_adx = prev['adx_14'] >= 20
            c_vol = prev['volume'] >= prev['vol_sma_20']
            
            if not in_pos:
                if c_trend and c_donchian and c_adx and c_vol:
                    in_pos = True
                    e_price = row['open']
                    atr = prev['atr_14']
                    sl_price = e_price - (2.0 * atr)
                    tp_price = e_price + (2.5 * atr) if exit_type in ['fixed_tp', 'partial_tp'] else 999999
                    e_idx = i
                    highest_high = row['high']
            else:
                highest_high = max(highest_high, row['high'])
                current_days = i - e_idx
                atr = prev['atr_14']
                
                hit_sl = row['low'] <= sl_price
                hit_tp = (row['high'] >= tp_price) if exit_type in ['fixed_tp', 'partial_tp'] else False
                
                hit_trailing = False
                if exit_type == 'trailing_atr':
                    trail_sl = highest_high - (param * atr)
                    hit_trailing = row['low'] <= trail_sl
                
                hit_reversal = False
                if exit_type == 'reversal_ema20':
                    hit_reversal = prev['close'] < prev['ema_20']
                
                hit_timestop = False
                if exit_type == 'time_stop':
                    if current_days >= param and (row['close'] - e_price)/e_price < 0.01:
                        hit_timestop = True

                if hit_sl or hit_tp or hit_trailing or hit_reversal or hit_timestop:
                    ex_price = sl_price if hit_sl else row['close']
                    if hit_tp: ex_price = tp_price
                    
                    pnl_p = ((ex_price - e_price) / e_price) * 100.0 - 0.15
                    t_list.append(pnl_p)
                    in_pos = False

        if len(t_list) == 0:
            return None
            
        t_arr = np.array(t_list)
        wins = t_arr[t_arr > 0]
        losses = t_arr[t_arr <= 0]
        pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0
        exp = t_arr.mean()
        total_ret = t_arr.sum()
        
        # Sizing e Wallet
        cagr = (pow(max(0.01, 1 + total_ret/100.0), 1/5.0) - 1) * 100.0
        
        return {
            'trades': len(t_arr),
            'win_rate': f"{len(wins)/len(t_arr)*100:.1f}%",
            'profit_factor': round(pf, 2),
            'expectancy_pct': round(exp, 2),
            'total_return_pct': round(total_ret, 2),
            'cagr_pct': round(cagr, 2)
        }

    ex_baseline = run_exit_experiment('fixed_tp')
    ex_trailing = run_exit_experiment('trailing_atr', param=2.5)
    ex_reversal = run_exit_experiment('reversal_ema20')
    ex_timestop = run_exit_experiment('time_stop', param=7)

    print(f" A) TP Fixo Atual (V1 Baseline) : {ex_baseline}")
    print(f" B) Trailing ATR (2.5x ATR)     : {ex_trailing}")
    print(f" C) Saída por Reversão (EMA 20) : {ex_reversal}")
    print(f" D) Time Stop (7 Dias Estagnado): {ex_timestop}")

    print("\n========================================================")
    print(" 5. BENCHMARK SIMPLES (V1 vs BUY & HOLD vs EMA 200 LONG-ONLY)")
    print("========================================================")
    bnh_ret = ((df.iloc[-1]['close'] - df.iloc[200]['open']) / df.iloc[200]['open']) * 100.0
    
    # EMA 200 Long-Only
    df['ema_sig'] = df['close'] > df['ema_200']
    ema_trades = []
    in_ema = False
    ema_entry = 0
    for i in range(200, len(df)):
        r = df.iloc[i]
        p = df.iloc[i-1]
        if not in_ema and p['close'] > p['ema_200']:
            in_ema = True
            ema_entry = r['open']
        elif in_ema and p['close'] < p['ema_200']:
            in_ema = False
            ret = ((r['open'] - ema_entry) / ema_entry) * 100.0 - 0.15
            ema_trades.append(ret)
            
    ema_ret = sum(ema_trades)
    print(f" Benchmark Buy & Hold (5 Anos BTC)    : +{bnh_ret:.2f}%")
    print(f" Benchmark EMA 200 Long-Only Simples : +{ema_ret:.2f}% ({len(ema_trades)} trades)")
    print(f" V1.0 Congelada                       : {ex_baseline['total_return_pct']:.2f}% ({ex_baseline['trades']} trades)")

if __name__ == "__main__":
    run_v1_exit_audit()
