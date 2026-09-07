import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import fetch_historical_data

class V1FinalIntegritySuite:
    def __init__(self, symbol: str = "BTC-USD", days: int = 1825):
        self.symbol = symbol
        self.days = days
        self.initial_capital = 1000.0
        self.fixed_sizing = 333.33
        
    def run_simulation(self, stop_mode: str = "gap_sensitive", round_trip_cost: float = 0.0030):
        df = fetch_historical_data(self.symbol, timeframe="1d", days=self.days)
        df = df.reset_index()
        date_col = 'Date' if 'Date' in df.columns else ('index' if 'index' in df.columns else df.columns[0])
        df['date'] = df[date_col]
        
        # Zero Look-Ahead Bias (.shift(1))
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
        
        trades = []
        in_position = False
        entry_price = 0.0
        stop_loss = 0.0
        entry_idx = 0
        current_wallet = self.initial_capital
        peak_wallet = self.initial_capital
        
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
                current_open = row['open']
                current_low = row['low']
                current_close = row['close']
                prev_donchian_low = prev['donchian_low_10']
                
                hit_sl = current_low <= stop_loss
                hit_donchian_exit = current_close < prev_donchian_low
                
                if hit_sl or hit_donchian_exit:
                    exit_idx = i
                    if hit_sl:
                        exit_reason = "SL"
                        if stop_mode == "gap_sensitive" and current_open <= stop_loss:
                            # Modelo Conservador de Gap Down: Executa na Abertura (Open) se Open <= Stop
                            exit_price = current_open
                            is_gap_execution = True
                        else:
                            # Preenchimento no nível teórico do Stop
                            exit_price = stop_loss
                            is_gap_execution = False
                    else:
                        exit_price = current_close
                        exit_reason = "DonchianLow10"
                        is_gap_execution = False
                        
                    pos_window = df.iloc[entry_idx : exit_idx + 1]
                    max_high = pos_window['high'].max()
                    min_low = pos_window['low'].min()
                    
                    mfe_pct = ((max_high - entry_price) / entry_price) * 100.0
                    mae_pct = ((min_low - entry_price) / entry_price) * 100.0
                    duracao = exit_idx - entry_idx
                    
                    # Duração até MFE e MAE
                    idx_mfe = pos_window['high'].idxmax() - df.index[entry_idx]
                    idx_mae = pos_window['low'].idxmin() - df.index[entry_idx]
                    
                    # Custo de Fricção Aplicado
                    raw_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    pnl_sizing_pct = raw_pct - (round_trip_cost * 100.0)
                    pnl_usd = (pnl_sizing_pct / 100.0) * self.fixed_sizing
                    
                    wallet_before = current_wallet
                    current_wallet += pnl_usd
                    pnl_wallet_pct = (pnl_usd / self.initial_capital) * 100.0
                    
                    peak_wallet = max(peak_wallet, current_wallet)
                    drawdown_pct = ((peak_wallet - current_wallet) / peak_wallet) * 100.0
                    
                    # Post-Exit Drift em 10 dias (Máxima e Fechamento)
                    future_10 = df.iloc[exit_idx + 1 : min(len(df), exit_idx + 11)]
                    post_max_10 = future_10['high'].max() if len(future_10) > 0 else exit_price
                    post_close_10 = future_10['close'].iloc[-1] if len(future_10) > 0 else exit_price
                    
                    drift_max_10_pct = ((post_max_10 - exit_price) / exit_price) * 100.0
                    drift_close_10_pct = ((post_close_10 - exit_price) / exit_price) * 100.0
                    
                    # Post-Exit Drift em 20 dias (Máxima)
                    future_20 = df.iloc[exit_idx + 1 : min(len(df), exit_idx + 21)]
                    post_max_20 = future_20['high'].max() if len(future_20) > 0 else exit_price
                    drift_max_20_pct = ((post_max_20 - exit_price) / exit_price) * 100.0
                    
                    trades.append({
                        'trade_id': len(trades) + 1,
                        'entry_date': str(df.iloc[entry_idx]['date'])[:10],
                        'exit_date': str(row['date'])[:10],
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(exit_price, 2),
                        'stop_loss_price': round(stop_loss, 2),
                        'daily_open_price': round(current_open, 2),
                        'daily_low_price': round(current_low, 2),
                        'is_gap_execution': is_gap_execution,
                        'exit_reason': exit_reason,
                        'wallet_before': round(wallet_before, 2),
                        'sizing_usd': round(self.fixed_sizing, 2),
                        'pnl_usd': round(pnl_usd, 2),
                        'pnl_sizing_pct': round(pnl_sizing_pct, 2),
                        'wallet_after': round(current_wallet, 2),
                        'pnl_wallet_pct': round(pnl_wallet_pct, 2),
                        'drawdown_pct': round(drawdown_pct, 2),
                        'mfe_pct': round(mfe_pct, 2),
                        'mae_pct': round(mae_pct, 2),
                        'duracao_dias': duracao,
                        'dias_ate_mfe': idx_mfe,
                        'dias_ate_mae': idx_mae,
                        'drift_max_10_pct': round(drift_max_10_pct, 2),
                        'drift_close_10_pct': round(drift_close_10_pct, 2),
                        'drift_max_20_pct': round(drift_max_20_pct, 2)
                    })
                    in_position = False
                    
        tdf = pd.DataFrame(trades)
        return df, tdf

def execute_final_audit():
    suite = V1FinalIntegritySuite(days=1825)
    df, tdf = suite.run_simulation(stop_mode="gap_sensitive", round_trip_cost=0.0030)
    
    wins = tdf[tdf['pnl_usd'] > 0]
    losers = tdf[tdf['pnl_usd'] <= 0]
    
    print("========================================================")
    print(" RELATÓRIO FINAL DE INTEGRIDADE E CONCILIAÇÃO V1.0 1D")
    print("========================================================")
    print(f" Total de Trades Oficiais: {len(tdf)}")
    print(f" Vencedores Oficiais: {len(wins)} trades (IDs: {wins['trade_id'].tolist()})")
    print(f" Perdedores Oficiais : {len(losers)} trades (IDs: {losers['trade_id'].tolist()})\n")
    
    # 1. Auditoria de Execução de Stop em Gaps de Abertura
    print("========================================================")
    print(" 1. AUDITORIA DE STOP COM EXECUÇÃO DE GAP DOWN EM ABERTURA")
    print("========================================================")
    gap_trades = tdf[tdf['is_gap_execution']]
    print(f" Trades com Execução de Gap Down em Open (Open <= Stop): {len(gap_trades)} trades")
    for idx, r in gap_trades.iterrows():
        print(f"   - Trade #{r['trade_id']} na data {r['exit_date']}: Stop Teórico ${r['stop_loss_price']:,.2f} | Open ${r['daily_open_price']:,.2f} -> Preço Executado em Open: ${r['exit_price']:,.2f}")

    # Comparação Modelo Ideal vs Modelo Conservador Gap
    _, tdf_ideal = suite.run_simulation(stop_mode="ideal_stop", round_trip_cost=0.0030)
    pnl_ideal = tdf_ideal['pnl_usd'].sum()
    pnl_gap = tdf['pnl_usd'].sum()
    
    print(f"\n   [IMPACTO DA AUDITORIA DE GAP]:")
    print(f"   - PnL Líquido Modelo Ideal Stop ($114k no Trade #15)       : ${pnl_ideal:+.2f} (+{(pnl_ideal/1000)*100:.2f}% na carteira)")
    print(f"   - PnL Líquido Modelo Conservador Gap ($109k em Open no #15): ${pnl_gap:+.2f} (+{(pnl_gap/1000)*100:.2f}% na carteira)")
    print(f"   - Diferença Total de Gap em Dólar: ${pnl_gap - pnl_ideal:+.2f} USD")

    # 2. Desagregação de Custos e Estresse
    print("\n========================================================")
    print(" 2. DESAGREGAÇÃO DE CUSTOS E TESTE DE ESTRESSE")
    print("========================================================")
    print(" Desmembramento da Fricção Padrão (0.30% Total Round-Trip):")
    print("   - Taxa de Entrada: 0.05%  | Slippage de Entrada: 0.10% -> Subtotal: 0.15%")
    print("   - Taxa de Saída  : 0.05%  | Slippage de Saída  : 0.10% -> Subtotal: 0.15%")
    print("\n Teste de Estresse de Fricção (Modelo Conservador Gap):")
    for cost in [0.0030, 0.0040, 0.0050, 0.0060]:
        _, tdf_c = suite.run_simulation(stop_mode="gap_sensitive", round_trip_cost=cost)
        p_sum = tdf_c['pnl_usd'].sum()
        wins_c = tdf_c[tdf_c['pnl_usd'] > 0]
        losses_c = tdf_c[tdf_c['pnl_usd'] <= 0]
        pf_c = abs(wins_c['pnl_usd'].sum() / losses_c['pnl_usd'].sum()) if len(losses_c) > 0 else 0
        print(f"   - Fricção {cost*100:.2f}% Round-Trip -> PnL: ${p_sum:+.2f} (+{(p_sum/1000)*100:.2f}%) | Profit Factor: {pf_c:.2f}")

    # 3. Concentração de Lucros dos Vencedores
    print("\n========================================================")
    print(" 3. CONCENTRAÇÃO DE LUCROS DOS VENCEDORES (MODELO CONSERVADOR GAP)")
    print("========================================================")
    gross_p = wins['pnl_usd'].sum()
    gross_l = abs(losers['pnl_usd'].sum())
    total_net = tdf['pnl_usd'].sum()
    
    top_w = wins.sort_values(by='pnl_usd', ascending=False)
    w1 = top_w.iloc[0]['pnl_usd']
    w2 = top_w.iloc[1]['pnl_usd']
    w3 = top_w.iloc[2]['pnl_usd']
    
    print(f" PnL Líquido Total Carteira: ${total_net:+.2f} (+{(total_net/1000)*100:.2f}%)")
    print(f"   - Sem o #1 Maior Vencedor (Trade #{top_w.iloc[0]['trade_id']} +${w1:.2f}) : ${total_net - w1:+.2f} (+{((total_net - w1)/1000)*100:.2f}%)")
    print(f"   - Sem os Top 2 Vencedores (+${w1+w2:.2f})                         : ${total_net - (w1+w2):+.2f} ({((total_net - (w1+w2))/1000)*100:+.2f}%)")
    print(f"   - Sem os Top 3 Vencedores (+${w1+w2+w3:.2f})                      : ${total_net - (w1+w2+w3):+.2f} ({((total_net - (w1+w2+w3))/1000)*100:+.2f}%)")
    print(f"   - Contribuição dos Top 3 Vencedores para o Lucro Bruto: {((w1+w2+w3)/gross_p)*100:.1f}%")

    # 4. Post-Exit Drift Completo dos 5 Vencedores
    print("\n========================================================")
    print(" 4. TABELA DE POST-EXIT DRIFT DOS 5 VENCEDORES (10d & 20d / Máxima vs Fechamento)")
    print("========================================================")
    hdr_drift = f" {'#':<2} | {'Saída':<10} | {'Preço Saída':<12} | {'PnL Siz(%)':<10} | {'Max 10d(%)':<10} | {'Close 10d(%)':<12} | {'Max 20d(%)':<10}"
    print(hdr_drift)
    print("-" * len(hdr_drift))
    for idx, r in wins.iterrows():
        print(f" {r['trade_id']:<2} | {r['exit_date']:<10} | ${r['exit_price']:<11.2f} | {r['pnl_sizing_pct']:<+9.2f}% | {r['drift_max_10_pct']:<+9.2f}% | {r['drift_close_10_pct']:<+11.2f}% | {r['drift_max_20_pct']:<+9.2f}%")

    # 5. Auditoria de MFE / MAE Completa
    print("\n========================================================")
    print(" 5. AUDITORIA COMPLETA DE MFE / MAE E DURAÇÃO TEMPORAL")
    print("========================================================")
    print(f" VENCEDORES ({len(wins)} trades):")
    print(f"   - MFE Médio: +{wins['mfe_pct'].mean():.2f}% | Mediana: +{wins['mfe_pct'].median():.2f}%")
    print(f"   - PnL Realizado Sizing Médio: +{wins['pnl_sizing_pct'].mean():.2f}% | Mediana: +{wins['pnl_sizing_pct'].median():.2f}%")
    print(f"   - MFE Realized Ratio Médio (PnL/MFE): {wins['pnl_sizing_pct'].mean()/wins['mfe_pct'].mean():.3f}")
    
    print(f"\n PERDEDORES ({len(losers)} trades):")
    print(f"   - MFE Médio Antes do Stop: +{losers['mfe_pct'].mean():.2f}% | Mediana: +{losers['mfe_pct'].median():.2f}%")
    print(f"   - MAE Médio nos Perdedores: {losers['mae_pct'].mean():.2f}% | Mediana: {losers['mae_pct'].median():.2f}%")
    print(f"   - Duração Média nos Perdedores: {losers['duracao_dias'].mean():.1f} dias")
    print(f"   - Tempo Médio até Pico de MFE nos Perdedores: {losers['dias_ate_mfe'].mean():.1f} dias")
    print(f"   - Tempo Médio até Fundo de MAE nos Perdedores: {losers['dias_ate_mae'].mean():.1f} dias")

    # 6. Benchmark Comparável Unificado
    print("\n========================================================")
    print(" 6. BENCHMARK COMPARÁVEL UNIFICADO (MESMO PERÍODO / MESMO $1.000)")
    print("========================================================")
    bnh_ret = ((df.iloc[-1]['close'] - df.iloc[200]['open']) / df.iloc[200]['open']) * 100.0
    
    # EMA 200 Long-Only Simples
    df['ema_sig'] = df['close'].shift(1) > df['ema_200']
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
            ret = ((r['open'] - ema_entry) / ema_entry) * 100.0 - 0.30
            pnl_u = (ret / 100.0) * 333.33
            ema_trades.append(pnl_u)
            
    ema_pnl_usd = sum(ema_trades)
    ema_ret_wallet = (ema_pnl_usd / 1000.0) * 100.0
    
    print(f" A) V1.0 1D Simplificada (Modelo Gap Conservador) : ${pnl_gap:+.2f} USD (+{(pnl_gap/1000)*100:.2f}% na carteira)")
    print(f" B) Buy & Hold BTC-USD (5 Anos Passivo)          : +{bnh_ret:.2f}%")
    print(f" C) EMA 200 Long-Only Simples                    : ${ema_pnl_usd:+.2f} USD (+{ema_ret_wallet:.2f}% na carteira com {len(ema_trades)} trades)")

if __name__ == "__main__":
    execute_final_audit()
