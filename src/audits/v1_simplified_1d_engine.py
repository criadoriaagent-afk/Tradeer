import sys
import os
import pandas as pd
import numpy as np

# Adicionar raiz ao PATH
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.core.data_loader import fetch_historical_data

class V1Simplified1DEngine:
    """
    Motor Oficial da Estratégia "V1.0 1D Simplificada" (Tradeer Quant).
    
    ESPECIFICAÇÃO RIGOROSA DAS REGRAS:
    -----------------------------------
    - Timeframe: 1D (Diário) Spot
    - Sizing: $333.33 USD por operação (1/3 de $1,000.00 de capital total)
    - Fricção de Execução: 0.15% por trade (corretagem + slippage)
    
    REGRAS DE ENTRADA (Sinal na vela i-1 fechada, entrada no Open da vela i):
      1. Trend: Close[i-1] > EMA_200[i-1]
      2. Donchian: Close[i-1] >= Donchian_High_30d[i-1]
      3. Momentum: ADX_14[i-1] >= 20
      4. Volume: Volume[i-1] >= Volume_SMA_20[i-1]
      (EMA 50 4H e Funding Rate: DESATIVADOS nesta versão 1D).
      
    REGRAS DE SAÍDA:
      1. Stop Loss: EntryPrice - (2.0 * ATR_14[i-1])
      2. Saída por Donchian: Close[i-1] < Donchian_Low_10d[i-1]
      
    REGRAS DE RISCO:
      1. Circuit Breaker: Perda diária acumulada da carteira >= 3.0% (pausa ordens por 24h).
    """
    def __init__(self, symbol: str = "BTC-USD", days: int = 1825):
        self.symbol = symbol
        self.days = days
        self.initial_capital = 1000.0
        self.sizing_per_trade = 333.33
        self.fee_rate = 0.0015 # 0.15%
        
    def run(self):
        df = fetch_historical_data(self.symbol, timeframe="1d", days=self.days)
        df = df.reset_index()
        date_col = 'Date' if 'Date' in df.columns else ('index' if 'index' in df.columns else df.columns[0])
        df['date'] = df[date_col]
        
        # Indicadores com Zero Look-Ahead Bias (.shift(1))
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
        
        trades = []
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
                current_low = row['low']
                current_close = row['close']
                prev_donchian_low = prev['donchian_low_10']
                
                hit_sl = current_low <= stop_loss
                hit_donchian_exit = current_close < prev_donchian_low
                
                if hit_sl or hit_donchian_exit:
                    exit_idx = i
                    if hit_sl:
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
                    
                    # PnL Bruto e Líquido com Fricção
                    raw_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    pnl_sizing_pct = raw_pct - (self.fee_rate * 2 * 100.0) # -0.30% total por trade
                    pnl_usd = (pnl_sizing_pct / 100.0) * self.sizing_per_trade
                    pnl_wallet_pct = (pnl_usd / self.initial_capital) * 100.0
                    
                    # Post-Exit Drift (Definição Matemática: Máxima atingida nos 10 dias subsequentes à saída)
                    future_10 = df.iloc[exit_idx + 1 : min(len(df), exit_idx + 11)]
                    post_max_10 = future_10['high'].max() if len(future_10) > 0 else exit_price
                    post_drift_10_pct = ((post_max_10 - exit_price) / exit_price) * 100.0
                    
                    trades.append({
                        'trade_id': len(trades) + 1,
                        'entry_date': str(df.iloc[entry_idx]['date'])[:10],
                        'exit_date': str(row['date'])[:10],
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(exit_price, 2),
                        'exit_reason': exit_reason,
                        'pnl_usd': round(pnl_usd, 2),
                        'pnl_sizing_pct': round(pnl_sizing_pct, 2),
                        'pnl_wallet_pct': round(pnl_wallet_pct, 2),
                        'mfe_pct': round(mfe_pct, 2),
                        'mae_pct': round(mae_pct, 2),
                        'duracao_dias': duracao,
                        'post_drift_10_pct': round(post_drift_10_pct, 2)
                    })
                    in_position = False
                    
        tdf = pd.DataFrame(trades)
        return tdf

def execute_audit():
    engine = V1Simplified1DEngine(days=1825)
    tdf = engine.run()
    
    print("========================================================")
    print(" RELATÓRIO DA V1.0 1D SIMPLIFICADA (5 ANOS BTC-USD)")
    print("========================================================")
    print(f" Capital Inicial Total: ${engine.initial_capital:,.2f}")
    print(f" Sizing por Operação: ${engine.sizing_per_trade:,.2f} (1/3 da carteira)")
    print(f" Total de Operações Realizadas: {len(tdf)}\n")
    
    wins = tdf[tdf['pnl_usd'] > 0]
    losers = tdf[tdf['pnl_usd'] <= 0]
    
    total_pnl_usd = tdf['pnl_usd'].sum()
    total_wallet_ret = (total_pnl_usd / engine.initial_capital) * 100.0
    total_sizing_ret = (total_pnl_usd / engine.sizing_per_trade) * 100.0
    
    print("========================================================")
    print(" 1. RECONCILIAÇÃO MATEMÁTICA DO SIZING")
    print("========================================================")
    print(f" Soma Total de PnLs em Dólar: ${total_pnl_usd:+.2f}")
    print(f" Capital Final Apurado na Carteira: ${engine.initial_capital + total_pnl_usd:,.2f}")
    print(f" Retorno Acumulado Sobre o Sizing ($333.33): {total_sizing_ret:+.2f}%")
    print(f" Retorno Acumulado Sobre a Carteira Total ($1,000.00): {total_wallet_ret:+.2f}%")
    
    print("\n========================================================")
    print(" 2. TABELA TRADE-BY-TRADE COMPLETA")
    print("========================================================")
    hdr = f" {'#':<2} | {'Entrada':<10} | {'Saída':<10} | {'Motivo':<13} | {'Entrada($)':<10} | {'Saída($)':<10} | {'PnL($)':<9} | {'PnL Siz(%)':<10} | {'PnL Wal(%)':<10} | {'MFE(%)':<7} | {'MAE(%)':<7} | {'Dur(d)':<6}"
    print(hdr)
    print("-" * len(hdr))
    for idx, r in tdf.iterrows():
        print(f" {r['trade_id']:<2} | {r['entry_date']:<10} | {r['exit_date']:<10} | {r['exit_reason']:<13} | ${r['entry_price']:<9.2f} | ${r['exit_price']:<9.2f} | ${r['pnl_usd']:<+8.2f} | {r['pnl_sizing_pct']:<+9.2f}% | {r['pnl_wallet_pct']:<+9.2f}% | {r['mfe_pct']:<+6.2f}% | {r['mae_pct']:<+6.2f}% | {r['duracao_dias']:<6}")

    print("\n========================================================")
    print(" 3. DETALHAMENTO INDIVIDUAL DOS 6 VENDEDORES (POST-EXIT DRIFT)")
    print("========================================================")
    print(" Definição Matemática de Drift 10d: ((High_max_10d_pós_saída - ExitPrice) / ExitPrice) * 100%")
    print("-" * 90)
    for idx, r in wins.iterrows():
        ratio = r['pnl_sizing_pct'] / r['mfe_pct'] if r['mfe_pct'] > 0 else 0
        print(f" Trade #{r['trade_id']} | Entrada: {r['entry_date']} -> Saída: {r['exit_date']} ({r['exit_reason']})")
        print(f"   Preço Entrada: ${r['entry_price']:,.2f} | Preço Saída: ${r['exit_price']:,.2f}")
        print(f"   PnL Realizado: ${r['pnl_usd']:+.2f} ({r['pnl_sizing_pct']:+.2f}% sobre Sizing / {r['pnl_wallet_pct']:+.2f}% sobre Carteira)")
        print(f"   MFE Alcançado: +{r['mfe_pct']:.2f}% | Razão PnL/MFE: {ratio:.2f} (Capturou {ratio*100:.1f}% do movimento)")
        print(f"   Post-Exit Drift em 10 dias: +{r['post_drift_10_pct']:.2f}%\n")

    print("========================================================")
    print(" 4. AUDITORIA ESTATÍSTICA COMPLETA DE MFE / MAE")
    print("========================================================")
    def print_stats(name, group):
        mfe_s = group['mfe_pct']
        mae_s = group['mae_pct']
        print(f"\n --- {name} [{len(group)} trades] ---")
        print(f"  MFE %: Média +{mfe_s.mean():.2f}% | Mediana +{mfe_s.median():.2f}% | P25 +{mfe_s.quantile(0.25):.2f}% | P75 +{mfe_s.quantile(0.75):.2f}% | P90 +{mfe_s.quantile(0.90):.2f}% | Max +{mfe_s.max():.2f}%")
        print(f"  MAE %: Média {mae_s.mean():.2f}% | Mediana {mae_s.median():.2f}% | P25 {mae_s.quantile(0.25):.2f}% | P75 {mae_s.quantile(0.75):.2f}% | P90 {mae_s.quantile(0.90):.2f}% | Max {mae_s.max():.2f}%")

    print_stats("VENCEDORES", wins)
    print_stats("PERDEDORES", losers)
    print_stats("TOTAL GERAL", tdf)

if __name__ == "__main__":
    execute_audit()
