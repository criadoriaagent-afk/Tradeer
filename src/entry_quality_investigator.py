import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import fetch_historical_data

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index()
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
    df['vol_rel'] = df['volume'].shift(1) / df['vol_sma_20']
    
    # Kaufman Efficiency Ratio (ER20)
    change = (df['close'].shift(1) - df['close'].shift(21)).abs()
    volatility = (df['close'].shift(1) - df['close'].shift(2)).abs().rolling(20).sum()
    df['er_20'] = (change / volatility).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Overextension Metrics (No look-ahead)
    df['dist_donchian_pct'] = ((df['open'] - df['donchian_high_30']) / df['donchian_high_30']) * 100.0
    df['dist_ema200_pct'] = ((df['open'] - df['ema_200']) / df['ema_200']) * 100.0
    df['dist_ema200_atrs'] = (df['open'] - df['ema_200']) / df['atr_14']
    
    # Breakout Candle (candle i-1)
    df['candle_body_pct'] = ((df['close'].shift(1) - df['open'].shift(1)) / df['open'].shift(1)) * 100.0
    df['candle_body_atrs'] = (df['close'].shift(1) - df['open'].shift(1)) / df['atr_14']
    df['candle_range_atrs'] = (df['high'].shift(1) - df['low'].shift(1)) / df['atr_14']
    
    return df

# ========================================================
# 1. ESTUDO DE EVENTOS DE ROMPIMENTOS BRUTOS (FILTER-FREE)
# ========================================================
def run_breakout_event_study(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'], days=1825):
    all_events = []
    
    for symbol in symbols:
        raw_df = fetch_historical_data(symbol, timeframe="1d", days=days)
        df = prepare_indicators(raw_df)
        
        for i in range(200, len(df) - 20):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            # Sinal de rompimento bruto (qualquer vela que fecha acima do Donchian30 anterior)
            is_breakout = prev['close'] >= prev['donchian_high_30']
            
            if is_breakout:
                entry_price = curr['open']
                
                # Horizontes de retorno futuro (sem regras de saída)
                fwd_1d = ((df.iloc[i+1]['close'] - entry_price) / entry_price) * 100.0
                fwd_3d = ((df.iloc[min(i+3, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                fwd_5d = ((df.iloc[min(i+5, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                fwd_10d = ((df.iloc[min(i+10, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                fwd_20d = ((df.iloc[min(i+20, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                
                window_20 = df.iloc[i : i+21]
                mfe_20d = ((window_20['high'].max() - entry_price) / entry_price) * 100.0
                mae_20d = ((window_20['low'].min() - entry_price) / entry_price) * 100.0
                max_ret_20d = ((window_20['close'].max() - entry_price) / entry_price) * 100.0
                min_ret_20d = ((window_20['close'].min() - entry_price) / entry_price) * 100.0
                
                # Filtros Ativos na V1
                c_trend = prev['close'] > prev['ema_200']
                c_adx = prev['adx_14'] >= 20
                c_vol = prev['volume'] >= prev['vol_sma_20']
                passed_v1_filters = c_trend and c_adx and c_vol
                
                all_events.append({
                    'symbol': symbol,
                    'date': str(curr['date'])[:10],
                    'entry_price': entry_price,
                    'er_20': prev['er_20'],
                    'dist_donchian_pct': curr['dist_donchian_pct'],
                    'dist_ema200_pct': curr['dist_ema200_pct'],
                    'dist_ema200_atrs': curr['dist_ema200_atrs'],
                    'candle_body_atrs': curr['candle_body_atrs'],
                    'candle_range_atrs': curr['candle_range_atrs'],
                    'vol_rel': curr['vol_rel'],
                    'adx_14': prev['adx_14'],
                    'atr_14': prev['atr_14'],
                    'passed_v1_filters': passed_v1_filters,
                    'fwd_1d': fwd_1d,
                    'fwd_3d': fwd_3d,
                    'fwd_5d': fwd_5d,
                    'fwd_10d': fwd_10d,
                    'fwd_20d': fwd_20d,
                    'mfe_20d': mfe_20d,
                    'mae_20d': mae_20d,
                    'max_ret_20d': max_ret_20d,
                    'min_ret_20d': min_ret_20d
                })
                
    return pd.DataFrame(all_events)

# ========================================================
# 2. MOTOR DE SIMULAÇÃO DE TRADES (V1 E V2) PARA AUTÓPSIA
# ========================================================
def simulate_trades(strategy_type="V1", symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'], days=1825):
    all_trades = []
    
    for symbol in symbols:
        raw_df = fetch_historical_data(symbol, timeframe="1d", days=days)
        df = prepare_indicators(raw_df)
        
        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        entry_idx = 0
        fee_rate = 0.0015
        sizing = 333.33
        
        for i in range(200, len(df)):
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
                current_low = row['low']
                current_close = row['close']
                hit_sl = current_low <= stop_loss
                
                if strategy_type == "V1":
                    hit_exit = current_close < prev['donchian_low_10']
                    exit_reason_name = "DonchianLow10"
                else: # V2
                    hit_exit = current_close < prev['ema_20']
                    exit_reason_name = "EMA20"
                    
                if hit_sl or hit_exit:
                    exit_idx = i
                    if hit_sl:
                        exit_price = stop_loss
                        exit_reason = "SL"
                    else:
                        exit_price = current_close
                        exit_reason = exit_reason_name
                        
                    pos_window = df.iloc[entry_idx : exit_idx + 1]
                    mfe_pct = ((pos_window['high'].max() - entry_price) / entry_price) * 100.0
                    mae_pct = ((pos_window['low'].min() - entry_price) / entry_price) * 100.0
                    duracao = exit_idx - entry_idx
                    
                    raw_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    pnl_sizing_pct = raw_pct - (fee_rate * 2 * 100.0)
                    pnl_usd = (pnl_sizing_pct / 100.0) * sizing
                    
                    entry_row = df.iloc[entry_idx]
                    prev_entry = df.iloc[entry_idx - 1]
                    
                    # Horizontes de retorno pós-entrada (sem interferência da saída)
                    fwd_1d = ((df.iloc[min(entry_idx+1, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                    fwd_3d = ((df.iloc[min(entry_idx+3, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                    fwd_5d = ((df.iloc[min(entry_idx+5, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                    fwd_10d = ((df.iloc[min(entry_idx+10, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                    fwd_20d = ((df.iloc[min(entry_idx+20, len(df)-1)]['close'] - entry_price) / entry_price) * 100.0
                    
                    all_trades.append({
                        'strategy': strategy_type,
                        'symbol': symbol,
                        'entry_date': str(entry_row['date'])[:10],
                        'exit_date': str(row['date'])[:10],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_usd': pnl_usd,
                        'pnl_sizing_pct': pnl_sizing_pct,
                        'mfe_pct': mfe_pct,
                        'mae_pct': mae_pct,
                        'duracao_dias': duracao,
                        # Características no momento da Entrada
                        'er_20': prev_entry['er_20'],
                        'dist_donchian_pct': entry_row['dist_donchian_pct'],
                        'dist_ema200_pct': entry_row['dist_ema200_pct'],
                        'dist_ema200_atrs': entry_row['dist_ema200_atrs'],
                        'candle_body_atrs': entry_row['candle_body_atrs'],
                        'vol_rel': entry_row['vol_rel'],
                        'adx_14': prev_entry['adx_14'],
                        'atr_14': prev_entry['atr_14'],
                        # Retornos prospectivos fixos
                        'fwd_1d': fwd_1d,
                        'fwd_3d': fwd_3d,
                        'fwd_5d': fwd_5d,
                        'fwd_10d': fwd_10d,
                        'fwd_20d': fwd_20d
                    })
                    in_position = False
                    
    return pd.DataFrame(all_trades)

# ========================================================
# 3. ANÁLISE DIAGNÓSTICA POR CAMADAS (ABLATION COMPARISON)
# ========================================================
def run_ablation_layers(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'], days=1825):
    layers_results = []
    
    # Camada 0: Donchian30 Puro
    # Camada 1: EMA200 + Donchian30
    # Camada 2: Donchian30 + EMA200 + ADX >= 20 + Vol >= SMA20 (V1.0 1D Simplificada)
    
    for layer_id, layer_name in [(0, "Camada 0: Donchian 30 Puro"), 
                                  (1, "Camada 1: EMA 200 + Donchian 30"), 
                                  (2, "Camada 2: V1.0 1D Simplificada (Completa)")]:
        all_trades = []
        for symbol in symbols:
            raw_df = fetch_historical_data(symbol, timeframe="1d", days=days)
            df = prepare_indicators(raw_df)
            
            in_position = False
            entry_price = 0.0
            stop_loss = 0.0
            fee_rate = 0.0015
            sizing = 333.33
            
            for i in range(200, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                
                c_donchian = prev['close'] >= prev['donchian_high_30']
                c_trend = prev['close'] > prev['ema_200']
                c_adx = prev['adx_14'] >= 20
                c_vol = prev['volume'] >= prev['vol_sma_20']
                
                if layer_id == 0:
                    entry_signal = c_donchian
                elif layer_id == 1:
                    entry_signal = c_donchian and c_trend
                else: # Layer 2
                    entry_signal = c_donchian and c_trend and c_adx and c_vol
                    
                if not in_position:
                    if entry_signal:
                        in_position = True
                        entry_price = row['open']
                        atr = prev['atr_14']
                        stop_loss = entry_price - (2.0 * atr)
                else:
                    current_low = row['low']
                    current_close = row['close']
                    hit_sl = current_low <= stop_loss
                    hit_exit = current_close < prev['donchian_low_10']
                    
                    if hit_sl or hit_exit:
                        exit_price = stop_loss if hit_sl else current_close
                        raw_pct = ((exit_price - entry_price) / entry_price) * 100.0
                        pnl_sizing_pct = raw_pct - (fee_rate * 2 * 100.0)
                        pnl_usd = (pnl_sizing_pct / 100.0) * sizing
                        
                        all_trades.append({'symbol': symbol, 'pnl_usd': pnl_usd, 'pnl_pct': pnl_sizing_pct})
                        in_position = False
                        
        tdf = pd.DataFrame(all_trades)
        wins = tdf[tdf['pnl_usd'] > 0]
        losers = tdf[tdf['pnl_usd'] <= 0]
        
        gross_profit = wins['pnl_usd'].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losers['pnl_usd'].sum()) if len(losers) > 0 else 1.0
        pf = gross_profit / gross_loss if gross_loss > 0 else np.nan
        total_pnl = tdf['pnl_usd'].sum()
        win_rate = (len(wins) / len(tdf) * 100.0) if len(tdf) > 0 else 0.0
        expectancy = tdf['pnl_pct'].mean() if len(tdf) > 0 else 0.0
        
        layers_results.append({
            'layer_name': layer_name,
            'total_trades': len(tdf),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(pf, 2),
            'expectancy_pct': round(expectancy, 2),
            'pnl_usd': round(total_pnl, 2)
        })
        
    return pd.DataFrame(layers_results)

def execute_full_diagnostic():
    print("========================================================")
    print(" INICIANDO ESTUDO AVANÇADO DE QUALIDADE DAS ENTRADAS")
    print("========================================================")
    
    # 1. Estudo de Eventos de Rompimentos Brutos
    print("\n[1/4] Executando Estudo de Eventos de Rompimentos Brutos (Filter-Free)...")
    events_df = run_breakout_event_study()
    
    print("\n--------------------------------------------------------")
    print(" RESUMO DOS ROMPIMENTOS DONCHIAN-30 BRUTOS (TODOS OS SINAIS)")
    print("--------------------------------------------------------")
    print(f" Total de Sinais de Rompimento Identificados em 5 Anos: {len(events_df)}")
    print(f" Sinais Aprovados pelos Filtros da V1.0: {len(events_df[events_df['passed_v1_filters']])} ({len(events_df[events_df['passed_v1_filters']])/len(events_df)*100:.1f}%)")
    print(f" Sinais Rejeitados pelos Filtros: {len(events_df[~events_df['passed_v1_filters']])} ({len(events_df[~events_df['passed_v1_filters']])/len(events_df)*100:.1f}%)\n")
    
    print(" EXPECTATIVA DE RETORNO PROSPECTIVO DOS ROMPIMENTOS BRUTOS (SEM SAÍDA):")
    for horizon, col in [('1 Dia', 'fwd_1d'), ('3 Dias', 'fwd_3d'), ('5 Dias', 'fwd_5d'), ('10 Dias', 'fwd_10d'), ('20 Dias', 'fwd_20d')]:
        s = events_df[col]
        wr = (s > 0).mean() * 100.0
        print(f"  - Horizon {horizon:<7}: Média {s.mean():>+6.2f}% | Mediana {s.median():>+6.2f}% | Win Rate {wr:>5.1f}% | P25 {s.quantile(0.25):>+6.2f}% | P75 {s.quantile(0.75):>+6.2f}%")
        
    print(f"\n MFE Médio em 20 Dias: +{events_df['mfe_20d'].mean():.2f}% (Mediana: +{events_df['mfe_20d'].median():.2f}%)")
    print(f" MAE Médio em 20 Dias: {events_df['mae_20d'].mean():.2f}% (Mediana: {events_df['mae_20d'].median():.2f}%)")

    # 2. Diagnóstico de Rompimento Esticado vs Desempenho
    print("\n--------------------------------------------------------")
    print(" 2. IMPACTO DA DISTÂNCIA DA EMA200 NO RETORNO DE 20 DIAS")
    print("--------------------------------------------------------")
    events_df['dist_ema_bucket'] = pd.cut(events_df['dist_ema200_atrs'], 
                                          bins=[-np.inf, 2.0, 4.0, 6.0, np.inf],
                                          labels=['< 2.0 ATRs', '2.0-4.0 ATRs', '4.0-6.0 ATRs', '>= 6.0 ATRs (Esticado)'])
    
    ema_grouped = events_df.groupby('dist_ema_bucket', observed=False).agg(
        n_sinais=('fwd_20d', 'count'),
        ret_medio_20d=('fwd_20d', 'mean'),
        ret_mediano_20d=('fwd_20d', 'median'),
        win_rate_20d=('fwd_20d', lambda x: (x > 0).mean() * 100),
        mfe_medio=('mfe_20d', 'mean'),
        mae_medio=('mae_20d', 'mean')
    )
    print(ema_grouped.to_string())

    print("\n--------------------------------------------------------")
    print(" 3. IMPACTO DO TAMANHO DA VELA DE ROMPIMENTO (CANDLE BODY IN ATRs)")
    print("--------------------------------------------------------")
    events_df['candle_bucket'] = pd.cut(events_df['candle_body_atrs'], 
                                        bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
                                        labels=['< 0.5 ATR', '0.5-1.0 ATR', '1.0-2.0 ATRs', '>= 2.0 ATRs (Explosivo)'])
    
    candle_grouped = events_df.groupby('candle_bucket', observed=False).agg(
        n_sinais=('fwd_20d', 'count'),
        ret_medio_20d=('fwd_20d', 'mean'),
        ret_mediano_20d=('fwd_20d', 'median'),
        win_rate_20d=('fwd_20d', lambda x: (x > 0).mean() * 100),
        mfe_medio=('mfe_20d', 'mean')
    )
    print(candle_grouped.to_string())

    # 3. Autópsia Comparativa: Vencedores vs Perdedores Executados (V1 e V2)
    print("\n[2/4] Executando Autópsia dos Trades Executados (V1 vs V2)...")
    v1_trades = simulate_trades("V1")
    v2_trades = simulate_trades("V2")
    all_exec = pd.concat([v1_trades, v2_trades], ignore_index=True)
    
    wins = all_exec[all_exec['pnl_usd'] > 0]
    losers = all_exec[all_exec['pnl_usd'] <= 0]
    
    print("\n========================================================")
    print(" AUTÓPSIA COMPARATIVA: VENCEDORES VS PERDEDORES (V1 + V2 COMBINADOS)")
    print("========================================================")
    
    features = [
        ('Distância EMA200 (%):', 'dist_ema200_pct'),
        ('Distância EMA200 (ATRs):', 'dist_ema200_atrs'),
        ('Distância Donchian (%):', 'dist_donchian_pct'),
        ('Tamanho Vela Rompimento (ATRs):', 'candle_body_atrs'),
        ('Volume Relativo:', 'vol_rel'),
        ('ADX 14:', 'adx_14'),
        ('ER 20 na Entrada:', 'er_20'),
        ('Retorno Fwd 3 Dias (%):', 'fwd_3d'),
        ('Retorno Fwd 10 Dias (%):', 'fwd_10d'),
        ('Retorno Fwd 20 Dias (%):', 'fwd_20d'),
        ('MFE Alcançado (%):', 'mfe_pct'),
        ('MAE Sofrido (%):', 'mae_pct'),
        ('Duração do Trade (Dias):', 'duracao_dias')
    ]
    
    hdr = f" {'Característica no Momento da Entrada':<35} | {'VENCEDORES (Média)':<20} | {'PERDEDORES (Média)':<20} | {'Diferença':<12}"
    print(hdr)
    print("-" * len(hdr))
    for label, col in features:
        w_val = wins[col].mean()
        l_val = losers[col].mean()
        diff = w_val - l_val
        print(f" {label:<35} | {w_val:>+19.2f}  | {l_val:>+19.2f}  | {diff:>+11.2f}")

    # 4. Análise Diagnóstica por Camadas (Ablation Comparison)
    print("\n[3/4] Executando Comparação Diagnóstica por Camadas (Ablation)...")
    ablation_df = run_ablation_layers()
    
    print("\n========================================================")
    print(" COMPARATIVO POR CAMADAS DE FILTRO (BTC + ETH + SOL - 5 ANOS)")
    print("========================================================")
    print(ablation_df.to_string(index=False))

    # 5. Esclarecimento da Metodologia de Portfólio
    print("\n========================================================")
    print(" 5. NOTA METODOLÓGICA SOBRE GESTÃO DE PORTFÓLIO E CAPITAL")
    print("========================================================")
    print(" [REGISTRO FORMAL DE ARQUITETURA]:")
    print(" - Os testes apresentados operam com $1.000,00 alocados INDIVIDUALMENTE para cada ativo.")
    print(" - A soma dos PnLs ($241 BTC + $156 ETH + $754 SOL = $1.151,00) NÃO DEVE ser interpretada")
    print("   como o retorno de uma conta única de $1.000,00 (pois a exposição total combinada seria de $3.000).")
    print(" - Para a avaliação final de produção, um Motor de Portfólio Unificado aplicará:")
    print("    * Banca Única Consolidada = $1.000,00.")
    print("    * Alocação por Operação = 33.3% ($333,33) por posição ativa.")
    print("    * Exposição Máxima Simultânea da Carteira = 100%.")
    print("    * Correlação de Ativos e Risco Agregado Diário.")
    print("========================================================")

if __name__ == "__main__":
    execute_full_diagnostic()
