import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.core.data_loader import fetch_historical_data

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

def run_independent_strategy(strategy_type: str, symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'], days=1825):
    trades = []
    
    for symbol in symbols:
        df = prepare_data(symbol, days)
        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        entry_idx = 0
        
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
                current_day = (i - entry_idx) + 1
                hit_sl = row['low'] <= stop_loss
                
                # Definir política de saída estrita de cada estratégia
                if strategy_type == "V1.0":
                    hit_exit = row['close'] < prev['donchian_low_10']
                elif strategy_type == "V2.0":
                    hit_exit = row['close'] < prev['ema_20']
                elif strategy_type == "EARLY_PRUNE_V1":
                    # MFE nos primeiros 3 dias
                    w1 = df.iloc[entry_idx]
                    w2 = df.iloc[min(entry_idx+1, len(df)-1)]
                    w3 = df.iloc[min(entry_idx+2, len(df)-1)]
                    mfe_d3 = ((max(w1['high'], w2['high'], w3['high']) - entry_price) / entry_price) * 100.0
                    
                    hit_early_prune = (current_day == 3) and (mfe_d3 < 1.0) and not hit_sl
                    hit_donchian = row['close'] < prev['donchian_low_10']
                    hit_exit = hit_early_prune or hit_donchian
                    
                if hit_sl or hit_exit:
                    entry_date_str = str(df.iloc[entry_idx]['date'])[:10]
                    entry_id = f"{symbol}_{entry_date_str}_{entry_price:.2f}"
                    
                    trades.append({
                        'entry_id': entry_id,
                        'strategy': strategy_type,
                        'symbol': symbol,
                        'entry_date': entry_date_str,
                        'entry_price': round(entry_price, 2),
                        'signal_description': "Close > EMA200 AND Close >= DonchianHigh30 AND ADX >= 20 AND Vol >= VolSMA20",
                        'exit_date': str(row['date'])[:10],
                        'exit_price': round(stop_loss if hit_sl else row['close'], 2),
                        'duracao_dias': i - entry_idx
                    })
                    in_position = False
                    
    return pd.DataFrame(trades)

def execute_identity_audit():
    print("========================================================")
    print(" AUDITORIA DE IDENTIDADE DE ENTRADAS TRADE-BY-TRADE")
    print("========================================================")
    
    df_v1 = run_independent_strategy("V1.0")
    df_v2 = run_independent_strategy("V2.0")
    df_ep = run_independent_strategy("EARLY_PRUNE_V1")
    
    set_v1 = set(df_v1['entry_id'].tolist())
    set_v2 = set(df_v2['entry_id'].tolist())
    set_ep = set(df_ep['entry_id'].tolist())
    
    # Todos os IDs únicos observados
    all_unique_ids = sorted(list(set_v1.union(set_v2).union(set_ep)))
    
    print("\n--------------------------------------------------------")
    print(" 1. CONTAGEM E COMPARAÇÃO DE CONJUNTOS DE ENTRY IDs")
    print("--------------------------------------------------------")
    print(f" Total de Entry IDs na V1.0         : {len(set_v1)}")
    print(f" Total de Entry IDs na V2.0         : {len(set_v2)}")
    print(f" Total de Entry IDs na EARLY_PRUNE_V1: {len(set_ep)}")
    
    shared_all = set_v1.intersection(set_v2).intersection(set_ep)
    print(f"\n Quantidade de Entry IDs 100% Compartilhados entre as 3: {len(shared_all)}")
    
    exclusive_v1 = set_v1 - (set_v2.union(set_ep))
    exclusive_v2 = set_v2 - (set_v1.union(set_ep))
    exclusive_ep = set_ep - (set_v1.union(set_v2))
    
    v1_v2_diff = set_v1.symmetric_difference(set_v2)
    v1_ep_diff = set_v1.symmetric_difference(set_ep)
    
    print(f" Entry IDs Exclusivos da V1.0         : {len(exclusive_v1)}")
    print(f" Entry IDs Exclusivos da V2.0         : {len(exclusive_v2)}")
    print(f" Entry IDs Exclusivos da EARLY_PRUNE_V1: {len(exclusive_ep)}")
    
    # Construir tabela mestre trade-by-trade
    master_rows = []
    for eid in all_unique_ids:
        in_v1 = eid in set_v1
        in_v2 = eid in set_v2
        in_ep = eid in set_ep
        
        # Pegar informações cadastrais do trade
        match_row = df_v1[df_v1['entry_id'] == eid]
        if match_row.empty:
            match_row = df_v2[df_v2['entry_id'] == eid]
        if match_row.empty:
            match_row = df_ep[df_ep['entry_id'] == eid]
            
        r = match_row.iloc[0]
        master_rows.append({
            'entry_id': eid,
            'symbol': r['symbol'],
            'entry_date': r['entry_date'],
            'entry_price': r['entry_price'],
            'in_v1': 'SIM' if in_v1 else 'NÃO',
            'in_v2': 'SIM' if in_v2 else 'NÃO',
            'in_ep': 'SIM' if in_ep else 'NÃO',
            'signal': r['signal_description']
        })
        
    master_df = pd.DataFrame(master_rows)
    
    print("\n--------------------------------------------------------")
    print(" 2. TABELA TRADE-BY-TRADE COMPLETA DE TODAS AS ENTRADAS")
    print("--------------------------------------------------------")
    hdr = f" {'#':<2} | {'Entry ID Único':<28} | {'Ativo':<8} | {'Data':<10} | {'Entrada($)':<10} | {'V1.0':<5} | {'V2.0':<5} | {'EARLY_PRUNE':<11}"
    print(hdr)
    print("-" * len(hdr))
    
    for idx, r in master_df.iterrows():
        print(f" {idx+1:<2} | {r['entry_id']:<28} | {r['symbol']:<8} | {r['entry_date']:<10} | ${r['entry_price']:<9.2f} | {r['in_v1']:<5} | {r['in_v2']:<5} | {r['in_ep']:<11}")
        
    print("\n========================================================")
    print(" 3. VEREDITO FORMAL DE IDENTIDADE DE ENTRADAS")
    print("========================================================")
    
    is_100_percent_identical = (len(set_v1) == len(set_v2) == len(set_ep) == len(shared_all))
    
    if is_100_percent_identical:
        print(" [VEREDITO OFICIAL]: 100% IDÊNTICOS!")
        print(' "V1.0, V2.0 e EARLY_PRUNE_V1 possuem exatamente as mesmas entradas históricas; as diferenças de desempenho são exclusivamente causadas pelas regras de saída/gestão."')
    else:
        print(" [VEREDITO OFICIAL]: DIVERGÊNCIA DETECTADA!")
        print(f" - V1.0 possui {len(set_v1)} trades | V2.0 possui {len(set_v2)} trades | EARLY_PRUNE_V1 possui {len(set_ep)} trades.")
        print(" - Explicação técnica: A política de saída da V2.0 (EMA 20) encerra posições em momentos distintos da V1.0 (Donchian 10d). Quando a V2.0 sai mais cedo ou mais tarde, uma vaga fica disponível no motor de execução, permitindo que um sinal subsequente abra uma nova posição que estava bloqueada na V1.0 (ou vice-versa).")
        if len(exclusive_v2) > 0:
            print("\n IDs Presentes Apenas na V2.0:")
            for eid in exclusive_v2:
                print(f"   * {eid}")
        if len(exclusive_v1) > 0:
            print("\n IDs Presentes Apenas na V1.0:")
            for eid in exclusive_v1:
                print(f"   * {eid}")
        if len(exclusive_ep) > 0:
            print("\n IDs Presentes Apenas na EARLY_PRUNE_V1:")
            for eid in exclusive_ep:
                print(f"   * {eid}")

if __name__ == "__main__":
    execute_identity_audit()
