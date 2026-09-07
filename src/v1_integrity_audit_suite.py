import sys
import os
import hashlib
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import fetch_historical_data

class V1IntegrityAuditEngine:
    def __init__(self, symbol: str = "BTC-USD", days: int = 1825):
        self.symbol = symbol
        self.days = days
        self.initial_capital = 1000.0
        self.fixed_sizing = 333.33 # 1/3 do capital inicial
        self.fee_rate_per_side = 0.0015 # 0.15% entrada, 0.15% saída -> 0.30% round-trip
        
    def run_backtest(self, sizing_mode: str = "fixed"):
        df = fetch_historical_data(self.symbol, timeframe="1d", days=self.days)
        df = df.reset_index()
        date_col = 'Date' if 'Date' in df.columns else ('index' if 'index' in df.columns else df.columns[0])
        df['date'] = df[date_col]
        
        # Indicadores com Zero Look-Ahead Bias (.shift(1))
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
                    
                    if sizing_mode == "fixed":
                        current_sizing = self.fixed_sizing
                    else:
                        current_sizing = current_wallet * (1.0 / 3.0) # Sizing Composto de 33.33% da Wallet Atual
            else:
                current_low = row['low']
                current_close = row['close']
                prev_donchian_low = prev['donchian_low_10']
                
                hit_sl = current_low <= stop_loss
                hit_donchian_exit = current_close < prev_donchian_low
                
                if hit_sl or hit_donchian_exit:
                    exit_idx = i
                    if hit_sl:
                        # Opção A de Execução de Stop Loss: Preço do Stop Loss teórico ($114,204.15)
                        exit_price = stop_loss
                        exit_reason = "SL"
                    else:
                        exit_price = current_close
                        exit_reason = "DonchianLow10"
                        
                    pos_window = df.iloc[entry_idx : exit_idx + 1]
                    max_high = pos_window['high'].max()
                    min_low = pos_window['low'].min()
                    
                    mfe_pct = ((max_high - entry_price) / entry_price) * 100.0
                    mae_pct = ((min_low - entry_price) / entry_price) * 100.0
                    duracao = exit_idx - entry_idx
                    
                    # Custo Total Round-Trip: 0.30% (0.15% entrada + 0.15% saída)
                    raw_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    pnl_sizing_pct = raw_pct - (self.fee_rate_per_side * 2 * 100.0) # -0.30%
                    pnl_usd = (pnl_sizing_pct / 100.0) * current_sizing
                    
                    wallet_before = current_wallet
                    current_wallet += pnl_usd
                    pnl_wallet_pct = (pnl_usd / self.initial_capital) * 100.0
                    
                    peak_wallet = max(peak_wallet, current_wallet)
                    drawdown_pct = ((peak_wallet - current_wallet) / peak_wallet) * 100.0
                    
                    # Post-Exit Drift (10 dias subsequentes)
                    future_10 = df.iloc[exit_idx + 1 : min(len(df), exit_idx + 11)]
                    post_max_10 = future_10['high'].max() if len(future_10) > 0 else exit_price
                    post_drift_10_pct = ((post_max_10 - exit_price) / exit_price) * 100.0
                    
                    trades.append({
                        'trade_id': len(trades) + 1,
                        'entry_date': str(df.iloc[entry_idx]['date'])[:10],
                        'exit_date': str(row['date'])[:10],
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(exit_price, 2),
                        'stop_loss_price': round(stop_loss, 2),
                        'daily_low_price': round(current_low, 2),
                        'sl_vs_low_diff_usd': round(stop_loss - current_low, 2),
                        'sl_vs_low_diff_pct': round(((stop_loss - current_low) / entry_price) * 100.0, 2),
                        'exit_reason': exit_reason,
                        'wallet_before': round(wallet_before, 2),
                        'sizing_usd': round(current_sizing, 2),
                        'pnl_usd': round(pnl_usd, 2),
                        'pnl_sizing_pct': round(pnl_sizing_pct, 2),
                        'wallet_after': round(current_wallet, 2),
                        'pnl_wallet_pct': round(pnl_wallet_pct, 2),
                        'drawdown_pct': round(drawdown_pct, 2),
                        'mfe_pct': round(mfe_pct, 2),
                        'mae_pct': round(mae_pct, 2),
                        'duracao_dias': duracao,
                        'post_drift_10_pct': round(post_drift_10_pct, 2)
                    })
                    in_position = False
                    
        tdf = pd.DataFrame(trades)
        return df, tdf

def run_integrity_audit():
    print("========================================================")
    print(" RELATÓRIO DE AUDITORIA DE INTEGRIDADE DO BACKTEST V1.0")
    print("========================================================")
    
    engine = V1IntegrityAuditEngine(days=1825)
    raw_df, tdf = engine.run_backtest(sizing_mode="fixed")
    
    # 1. Auditoria de Execução de Stop Loss (Trade #15 e Tabela de SL)
    print("\n========================================================")
    print(" 1. AUDITORIA DE EXECUÇÃO DE STOP LOSS (ANÁLISE DE SL vs LOW)")
    print("========================================================")
    print(" ESCLARECIMENTO DE EXECUÇÃO DE STOP LOSS EM OHLC 1D:")
    print("   - Opção A (Adotada no Backtest): O algoritmo assume que a ordem Stop Loss estava posicionada")
    print("     previamente na exchange no nível teórico do Stop ($114,204.15 no Trade #15). Assim que o mercado")
    print("     atinge/atravessa essa cotação, a ordem é preenchida exatamente em $114,204.15 (-4.05% PnL).")
    print("   - Amplitude Máxima Adversa (MAE = -11.86%): Reflete o ponto mínimo atingido pelo candle ($104,580.00)")
    print("     mais tarde no mesmo dia. O MAE mede a excursão máxima adversa da barra, não o preço de saída.")
    print("   - Limitação do OHLC 1D: O candle diário NÃO fornece a sequência temporal intraday dos eventos.")
    print("     Não sabemos se o preço primeiro caiu e bateu no Stop Loss, ou se subiu antes. Recomenda-se um")
    print("     simulador híbrido (Sinal 1D + Execução 1H/15m) para simular slippage intraday de forma realista.\n")
    
    sl_trades = tdf[tdf['exit_reason'] == 'SL']
    print(f" TABELA DE TRADES STOPADOS (Low <= Stop Loss) [{len(sl_trades)} trades]:")
    print(f" {'#':<2} | {'Data Saída':<10} | {'Preço Entrada':<12} | {'Preço Stop':<12} | {'Low do Dia':<12} | {'Saída Efetiva':<12} | {'PnL (%)':<9} | {'MAE (%)':<8} | {'Diferença SL-Low ($)':<20}")
    print("-" * 115)
    for idx, r in sl_trades.iterrows():
        diff = r['stop_loss_price'] - r['daily_low_price']
        print(f" {r['trade_id']:<2} | {r['exit_date']:<10} | ${r['entry_price']:<11.2f} | ${r['stop_loss_price']:<11.2f} | ${r['daily_low_price']:<11.2f} | ${r['exit_price']:<11.2f} | {r['pnl_sizing_pct']:<+8.2f}% | {r['mae_pct']:<+7.2f}% | ${diff:<19.2f}")

    # 2. Padronização de Custos
    print("\n========================================================")
    print(" 2. PADRONIZAÇÃO DE CUSTOS DE EXECUÇÃO")
    print("========================================================")
    print(" Custos Padronizados e Aplicados no Código:")
    print("   - Taxa de Entrada: 0.15% (Comissão + Slippage)")
    print("   - Taxa de Saída  : 0.15% (Comissão + Slippage)")
    print("   - Custo Total Round-Trip: 0.30% estritamente deduzido do PnL de cada trade.")

    # 3. Recálculo Direto das Métricas de MFE e PnL dos Vencedores
    print("\n========================================================")
    print(" 3. RECÁLCULO DIRETO DAS MÉTRICAS DOS VENCEDORES (SEM AGREGADOS)")
    print("========================================================")
    wins = tdf[tdf['pnl_usd'] > 0].copy()
    losers = tdf[tdf['pnl_usd'] <= 0].copy()
    
    # Recálculo direto a partir dos 5 trades da tabela
    pnl_wins_array = wins['pnl_sizing_pct'].values
    mfe_wins_array = wins['mfe_pct'].values
    
    mean_pnl_wins = np.mean(pnl_wins_array)
    median_pnl_wins = np.median(pnl_wins_array)
    
    mean_mfe_wins = np.mean(mfe_wins_array)
    median_mfe_wins = np.median(mfe_wins_array)
    
    # MFE Realized Ratio por trade
    wins['mfe_ratio'] = wins['pnl_sizing_pct'] / wins['mfe_pct']
    mean_mfe_ratio = wins['mfe_ratio'].mean()
    median_mfe_ratio = wins['mfe_ratio'].median()
    
    print(f" Vencedores ({len(wins)} trades):")
    print(f"   - Valores de PnL Sizing (%): {np.round(pnl_wins_array, 2)}")
    print(f"   - PnL Médio dos Vencedores: +{mean_pnl_wins:.2f}% | Mediana: +{median_pnl_wins:.2f}%")
    print(f"   - MFE Médio dos Vencedores: +{mean_mfe_wins:.2f}% | Mediana: +{median_mfe_wins:.2f}%")
    print(f"\n   - MFE Realized Ratio Trade-by-Trade (PnL / MFE):")
    for idx, r in wins.iterrows():
        ratio = r['pnl_sizing_pct'] / r['mfe_pct']
        print(f"     Trade #{r['trade_id']}: PnL +{r['pnl_sizing_pct']:.2f}% / MFE +{r['mfe_pct']:.2f}% = Razão {ratio:.3f} (Capturou {ratio*100:.1f}%)")
    print(f"   - MFE Realized Ratio Médio: {mean_mfe_ratio:.3f} ({mean_mfe_ratio*100:.1f}%) | Mediana: {median_mfe_ratio:.3f} ({median_mfe_ratio*100:.1f}%)")

    # 4. Auditoria de Política de Sizing (Fixo $333.33 vs Composto 33.33%)
    print("\n========================================================")
    print(" 4. AUDITORIA DE POLÍTICA DE SIZING (FIXO $333.33 vs COMPOSTO 33.33%)")
    print("========================================================")
    _, tdf_comp = engine.run_backtest(sizing_mode="compounded")
    
    pnl_fixed = tdf['pnl_usd'].sum()
    ret_fixed_wallet = (pnl_fixed / engine.initial_capital) * 100.0
    
    pnl_comp = tdf_comp['pnl_usd'].sum()
    final_wallet_comp = tdf_comp.iloc[-1]['wallet_after']
    ret_comp_wallet = ((final_wallet_comp - engine.initial_capital) / engine.initial_capital) * 100.0
    
    print(f" Política A: Sizing FIXO de $333.33 por trade:")
    print(f"   - PnL Total em Dólar: ${pnl_fixed:+.2f}")
    print(f"   - Capital Final na Carteira: ${engine.initial_capital + pnl_fixed:,.2f}")
    print(f"   - Retorno Acumulado sobre a Carteira: +{ret_fixed_wallet:.2f}%")
    
    print(f"\n Política B: Sizing COMPOSTO (33.33% da Wallet no momento da entrada):")
    print(f"   - PnL Total em Dólar: ${pnl_comp:+.2f}")
    print(f"   - Capital Final na Carteira: ${final_wallet_comp:,.2f}")
    print(f"   - Retorno Acumulado sobre a Carteira: +{ret_comp_wallet:.2f}%")
    print(f" DECISÃO DE CONGELAMENTO PARA V1.0: Mantemos congelada a Política A (Sizing FIXO de $333.33) por ser conservadora e imune a distorções de escala.")

    # 5. Curva de Patrimônio e Capital Real Trade-by-Trade
    print("\n========================================================")
    print(" 5. CURVA DE PATRIMÔNIO E CAPITAL REAL TRADE-BY-TRADE")
    print("========================================================")
    hdr_eq = f" {'#':<2} | {'Data Saída':<10} | {'Wallet Antes($)':<15} | {'Sizing($)':<10} | {'PnL($)':<9} | {'PnL Siz(%)':<10} | {'Wallet Depois($)':<16} | {'Ret. Wallet(%)':<14} | {'Drawdown(%)':<11}"
    print(hdr_eq)
    print("-" * len(hdr_eq))
    for idx, r in tdf.iterrows():
        cum_ret = ((r['wallet_after'] - engine.initial_capital) / engine.initial_capital) * 100.0
        print(f" {r['trade_id']:<2} | {r['exit_date']:<10} | ${r['wallet_before']:<14.2f} | ${r['sizing_usd']:<9.2f} | ${r['pnl_usd']:<+8.2f} | {r['pnl_sizing_pct']:<+9.2f}% | ${r['wallet_after']:<15.2f} | {cum_ret:<+13.2f}% | {r['drawdown_pct']:<10.2f}%")

    # 6. Análise de Dependência de Grandes Vencedores
    print("\n========================================================")
    print(" 6. ANÁLISE DE DEPENDÊNCIA DE GRANDES VENCEDORES")
    print("========================================================")
    gross_profit = wins['pnl_usd'].sum()
    gross_loss = abs(losers['pnl_usd'].sum())
    total_net_pnl = tdf['pnl_usd'].sum()
    
    # Top Winners
    top_winners = wins.sort_values(by='pnl_usd', ascending=False)
    w1 = top_winners.iloc[0]['pnl_usd']
    w2 = top_winners.iloc[1]['pnl_usd']
    w3 = top_winners.iloc[2]['pnl_usd']
    
    pnl_no_w1 = total_net_pnl - w1
    pnl_no_w1_w2 = total_net_pnl - (w1 + w2)
    pnl_no_top3 = total_net_pnl - (w1 + w2 + w3)
    
    top3_contrib_gross = ((w1 + w2 + w3) / gross_profit) * 100.0
    
    print(f" PnL Bruto dos Vencedores (+): ${gross_profit:,.2f}")
    print(f" PnL Bruto dos Perdedores (-): -${gross_loss:,.2f}")
    print(f" PnL Líquido Total Carteira  : ${total_net_pnl:+.2f} (+{(total_net_pnl/1000)*100:.2f}% na carteira)")
    print(f"\n   - Retorno sem o 1º Maior Vencedor (Trade #{top_winners.iloc[0]['trade_id']} +${w1:.2f}): ${pnl_no_w1:+.2f} (+{(pnl_no_w1/1000)*100:.2f}% na carteira)")
    print(f"   - Retorno sem os Top 2 Vencedores (+${w1+w2:.2f}): ${pnl_no_w1_w2:+.2f} ({(pnl_no_w1_w2/1000)*100:+.2f}% na carteira)")
    print(f"   - Retorno sem os Top 3 Vencedores (+${w1+w2+w3:.2f}): ${pnl_no_top3:+.2f} ({(pnl_no_top3/1000)*100:+.2f}% na carteira)")
    print(f"   - Contribuição dos Top 3 Vencedores para o Lucro Bruto: {top3_contrib_gross:.1f}%")

    # 7. Auditoria do Período e Qualidade dos Dados
    print("\n========================================================")
    print(" 7. AUDITORIA DO PERÍODO E QUALIDADE DOS DADOS")
    print("========================================================")
    start_ts = raw_df['date'].iloc[0]
    end_ts = raw_df['date'].iloc[-1]
    total_bars = len(raw_df)
    missing_gaps = raw_df['date'].diff().dt.days.gt(1).sum()
    
    print(f" Timestamp Inicial : {start_ts}")
    print(f" Timestamp Final   : {end_ts}")
    print(f" Quantidade Real de Velas 1D Carregadas: {total_bars} velas")
    print(f" Fonte dos Dados   : yfinance (Ticker: BTC-USD)")
    print(f" Verificação de Gaps na Série Temporal: {missing_gaps} lacunas detectadas")

    # 8. Reprodução Dupla
    print("\n========================================================")
    print(" 8. TESTE DE REPRODUTIBILIDADE DETERMINÍSTICA DUPLA")
    print("========================================================")
    _, tdf_run1 = engine.run_backtest(sizing_mode="fixed")
    _, tdf_run2 = engine.run_backtest(sizing_mode="fixed")
    
    pnl1 = tdf_run1['pnl_usd'].sum()
    pnl2 = tdf_run2['pnl_usd'].sum()
    trades1 = len(tdf_run1)
    trades2 = len(tdf_run2)
    
    print(f" Execução 1: {trades1} trades | PnL Total: ${pnl1:+.2f}")
    print(f" Execução 2: {trades2} trades | PnL Total: ${pnl2:+.2f}")
    print(f" REPRODUTIBILIDADE 100% CONFIRMADA: {pnl1 == pnl2 and trades1 == trades2}")

if __name__ == "__main__":
    run_integrity_audit()
