import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import fetch_historical_data

class StrategyEngine:
    def __init__(self, symbol: str = "BTC-USD", days: int = 1825, exit_rule: str = "donchian_10d"):
        self.symbol = symbol
        self.days = days
        self.exit_rule = exit_rule # 'donchian_10d' (V1.0) vs 'ema_20_reversal' (V2.0)
        self.initial_capital = 1000.0
        self.fixed_sizing = 333.33
        self.fee_rate_round_trip = 0.0030 # 0.30%
        
    def run(self):
        df = fetch_historical_data(self.symbol, timeframe="1d", days=self.days)
        df = df.reset_index()
        date_col = 'Date' if 'Date' in df.columns else ('index' if 'index' in df.columns else df.columns[0])
        df['date'] = df[date_col]
        
        # Indicadores com Zero Look-Ahead Bias (.shift(1))
        df['donchian_high_30'] = df['high'].shift(1).rolling(window=30).max()
        df['donchian_low_10'] = df['low'].shift(1).rolling(window=10).min()
        df['ema_200'] = df['close'].shift(1).ewm(span=200, adjust=False).mean()
        df['ema_20'] = df['close'].shift(1).ewm(span=20, adjust=False).mean()
        
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
                
                hit_sl = current_low <= stop_loss
                
                if self.exit_rule == "donchian_10d":
                    hit_exit_signal = current_close < prev['donchian_low_10']
                    exit_reason_name = "DonchianLow10"
                else: # 'ema_20_reversal'
                    hit_exit_signal = current_close < prev['ema_20']
                    exit_reason_name = "EMA20_Reversal"
                    
                if hit_sl or hit_exit_signal:
                    exit_idx = i
                    if hit_sl:
                        exit_price = current_open if current_open <= stop_loss else stop_loss
                        exit_reason = "SL"
                    else:
                        exit_price = current_close
                        exit_reason = exit_reason_name
                        
                    pos_window = df.iloc[entry_idx : exit_idx + 1]
                    max_high = pos_window['high'].max()
                    min_low = pos_window['low'].min()
                    
                    mfe_pct = ((max_high - entry_price) / entry_price) * 100.0
                    mae_pct = ((min_low - entry_price) / entry_price) * 100.0
                    duracao = exit_idx - entry_idx
                    
                    raw_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    pnl_sizing_pct = raw_pct - (self.fee_rate_round_trip * 100.0)
                    pnl_usd = (pnl_sizing_pct / 100.0) * self.fixed_sizing
                    
                    wallet_before = current_wallet
                    current_wallet += pnl_usd
                    pnl_wallet_pct = (pnl_usd / self.initial_capital) * 100.0
                    
                    peak_wallet = max(peak_wallet, current_wallet)
                    drawdown_pct = ((peak_wallet - current_wallet) / peak_wallet) * 100.0
                    
                    # Post-Exit Drift em 10 dias
                    future_10 = df.iloc[exit_idx + 1 : min(len(df), exit_idx + 11)]
                    post_max_10 = future_10['high'].max() if len(future_10) > 0 else exit_price
                    post_close_10 = future_10['close'].iloc[-1] if len(future_10) > 0 else exit_price
                    drift_max_10_pct = ((post_max_10 - exit_price) / exit_price) * 100.0
                    drift_close_10_pct = ((post_close_10 - exit_price) / exit_price) * 100.0
                    
                    # Post-Exit Drift em 20 dias
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
                        'exit_reason': exit_reason,
                        'wallet_before': round(wallet_before, 2),
                        'pnl_usd': round(pnl_usd, 2),
                        'pnl_sizing_pct': round(pnl_sizing_pct, 2),
                        'wallet_after': round(current_wallet, 2),
                        'pnl_wallet_pct': round(pnl_wallet_pct, 2),
                        'drawdown_pct': round(drawdown_pct, 2),
                        'mfe_pct': round(mfe_pct, 2),
                        'mae_pct': round(mae_pct, 2),
                        'duracao_dias': duracao,
                        'drift_max_10_pct': round(drift_max_10_pct, 2),
                        'drift_close_10_pct': round(drift_close_10_pct, 2),
                        'drift_max_20_pct': round(drift_max_20_pct, 2)
                    })
                    in_position = False
                    
        tdf = pd.DataFrame(trades)
        return df, tdf

def calc_metrics(tdf):
    if tdf.empty:
        return {}
    wins = tdf[tdf['pnl_usd'] > 0]
    losers = tdf[tdf['pnl_usd'] <= 0]
    
    total_trades = len(tdf)
    win_rate = (len(wins) / total_trades) * 100.0
    
    gross_p = wins['pnl_usd'].sum() if len(wins) > 0 else 0.0
    gross_l = abs(losers['pnl_usd'].sum()) if len(losers) > 0 else 0.0
    profit_factor = gross_p / gross_l if gross_l > 0 else 999.0
    
    total_pnl_usd = tdf['pnl_usd'].sum()
    total_ret_wallet = (total_pnl_usd / 1000.0) * 100.0
    cagr = (pow(max(0.01, 1 + total_ret_wallet/100.0), 1/5.0) - 1) * 100.0
    
    exp_usd = tdf['pnl_usd'].mean()
    exp_pct = tdf['pnl_sizing_pct'].mean()
    
    max_dd = tdf['drawdown_pct'].max()
    avg_dur = tdf['duracao_dias'].mean()
    
    # Sharpe & Sortino (simplificado anualizado sobre retornos de trades)
    returns = tdf['pnl_sizing_pct'] / 100.0
    mean_ret = returns.mean()
    std_ret = returns.std() if len(returns) > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(total_trades / 5.0) if std_ret > 0 else 0.0
    
    downside_returns = returns[returns < 0]
    std_down = downside_returns.std() if len(downside_returns) > 1 else 1.0
    sortino = (mean_ret / std_down) * np.sqrt(total_trades / 5.0) if std_down > 0 else 0.0
    
    # Maior sequência de perdas
    tdf['is_loss'] = tdf['pnl_usd'] <= 0
    loss_streaks = tdf['is_loss'].groupby((~tdf['is_loss']).cumsum()).sum()
    max_loss_streak = loss_streaks.max() if len(loss_streaks) > 0 else 0
    
    # MFE Realized Ratio nos Vencedores
    mfe_cap_mean = (wins['pnl_sizing_pct'] / wins['mfe_pct']).mean() * 100.0 if len(wins) > 0 else 0.0
    
    # Dependência de Top Winners
    top_w = wins.sort_values(by='pnl_usd', ascending=False)
    w1 = top_w.iloc[0]['pnl_usd'] if len(top_w) > 0 else 0.0
    w2 = top_w.iloc[1]['pnl_usd'] if len(top_w) > 1 else 0.0
    w3 = top_w.iloc[2]['pnl_usd'] if len(top_w) > 2 else 0.0
    
    ret_no_w1 = ((total_pnl_usd - w1) / 1000.0) * 100.0
    ret_no_top3 = ((total_pnl_usd - (w1+w2+w3)) / 1000.0) * 100.0
    contrib_top3 = ((w1+w2+w3) / gross_p * 100.0) if gross_p > 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'win_rate_pct': round(win_rate, 2),
        'total_pnl_usd': round(total_pnl_usd, 2),
        'total_ret_wallet_pct': round(total_ret_wallet, 2),
        'cagr_pct': round(cagr, 2),
        'profit_factor': round(profit_factor, 2),
        'expectancy_usd': round(exp_usd, 2),
        'expectancy_sizing_pct': round(exp_pct, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'avg_duration_days': round(avg_dur, 1),
        'max_loss_streak': max_loss_streak,
        'mfe_cap_mean_pct': round(mfe_cap_mean, 2),
        'ret_no_w1_pct': round(ret_no_w1, 2),
        'ret_no_top3_pct': round(ret_no_top3, 2),
        'contrib_top3_gross_pct': round(contrib_top3, 2)
    }

def execute_v1_vs_v2_comparison():
    print("========================================================")
    print(" AVALIAÇÃO CIENTÍFICA: V1.0 1D SIMPLIFICADA vs V2.0 (EMA 20)")
    print("========================================================")
    
    # 1. Simular V1.0 (Saída Donchian 10d)
    e_v1 = StrategyEngine(days=1825, exit_rule="donchian_10d")
    df_v1, tdf_v1 = e_v1.run()
    m_v1 = calc_metrics(tdf_v1)
    
    # 2. Simular V2.0 (Saída Reversão EMA 20)
    e_v2 = StrategyEngine(days=1825, exit_rule="ema_20_reversal")
    df_v2, tdf_v2 = e_v2.run()
    m_v2 = calc_metrics(tdf_v2)
    
    print("\n========================================================")
    print(" 1. MATRIZ COMPARATIVA GERAL: V1.0 vs V2.0 (IN-SAMPLE 5 ANOS)")
    print("========================================================")
    hdr = f" {'Métrica':<35} | {'V1.0 1D (Donchian 10d)':<22} | {'V2.0 (EMA 20 Reversal)':<22}"
    print(hdr)
    print("-" * len(hdr))
    
    metrics_map = [
        ("Total de Trades", "total_trades"),
        ("Taxa de Acerto (%)", "win_rate_pct"),
        ("PnL Líquido Total ($)", "total_pnl_usd"),
        ("Retorno Acumulado Carteira (%)", "total_ret_wallet_pct"),
        ("CAGR (%/ano)", "cagr_pct"),
        ("Profit Factor", "profit_factor"),
        ("Expectativa por Trade ($)", "expectancy_usd"),
        ("Expectativa por Trade (%)", "expectancy_sizing_pct"),
        ("Max Drawdown (%)", "max_drawdown_pct"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Sortino Ratio", "sortino_ratio"),
        ("Duração Média dos Trades (dias)", "avg_duration_days"),
        ("Maior Sequência de Perdas (trades)", "max_loss_streak"),
        ("Captura Média do MFE (%)", "mfe_cap_mean_pct"),
        ("Retorno sem o #1 Vencedor (%)", "ret_no_w1_pct"),
        ("Retorno sem os Top 3 Vencedores (%)", "ret_no_top3_pct"),
        ("Contribuição Top 3 ao Lucro Bruto (%)", "contrib_top3_gross_pct")
    ]
    
    for label, key in metrics_map:
        val1 = m_v1[key]
        val2 = m_v2[key]
        print(f" {label:<35} | {str(val1):<22} | {str(val2):<22}")

    print("\n========================================================")
    print(" 2. ANÁLISE DE LUCROS CAPTURADOS NOS VENCEDORES (V2.0)")
    print("========================================================")
    wins_v2 = tdf_v2[tdf_v2['pnl_usd'] > 0]
    print(f" Vencedores na V2.0 [{len(wins_v2)} trades]:")
    print(f" {'#':<2} | {'Entrada':<10} | {'Saída':<10} | {'PnL Sizing(%)':<12} | {'MFE(%)':<8} | {'MFE Cap(%)':<10} | {'Max 10d(%)':<10} | {'Max 20d(%)':<10}")
    print("-" * 85)
    for idx, r in wins_v2.iterrows():
        cap = (r['pnl_sizing_pct'] / r['mfe_pct']) * 100.0 if r['mfe_pct'] > 0 else 0
        print(f" {r['trade_id']:<2} | {r['entry_date']:<10} | {r['exit_date']:<10} | {r['pnl_sizing_pct']:<+11.2f}% | {r['mfe_pct']:<+7.2f}% | {cap:<9.1f}% | {r['drift_max_10_pct']:<+9.2f}% | {r['drift_max_20_pct']:<+9.2f}%")

    print("\n========================================================")
    print(" 3. ANÁLISE DO 'CUSTO DE DEIXAR CORRER' (DEVOLUÇÃO DE LUCROS DA V2)")
    print("========================================================")
    print(" Comparando os mesmos pontos de entrada para verificar onde a V2 devoliu margem durante pullbacks:")
    print(" - Na V1.0 (Donchian 10d), a saída ocorreu mais cedo na perda da mínima de 10 dias.")
    print(" - Na V2.0 (EMA 20), o robô aguarda o fechamento abaixo da média rápida de 20 períodos.")
    print(" - Em tendências parabólicas, a EMA 20 fica distante do topo. Quando a reversão ocorre, o preço devolve")
    print("   uma fração do MFE acumulado antes de autorizar a saída.")
    print(f"   * MFE Médio dos Vencedores na V2.0 : +{wins_v2['mfe_pct'].mean():.2f}%")
    print(f"   * PnL Médio Realizado na V2.0      : +{wins_v2['pnl_sizing_pct'].mean():.2f}%")
    print(f"   * Margem Média Devolvida no Topo    : {wins_v2['mfe_pct'].mean() - wins_v2['pnl_sizing_pct'].mean():.2f}%")

    print("\n========================================================")
    print(" 4. BENCHMARK COMPARÁVEL UNIFICADO (V1.0 vs V2.0 vs BUY & HOLD vs EMA 200)")
    print("========================================================")
    bnh_ret = ((df_v1.iloc[-1]['close'] - df_v1.iloc[200]['open']) / df_v1.iloc[200]['open']) * 100.0
    
    # EMA 200 Long-Only Simples
    df_v1['ema_sig'] = df_v1['close'].shift(1) > df_v1['ema_200']
    ema_trades = []
    in_ema = False
    ema_entry = 0
    for i in range(200, len(df_v1)):
        r = df_v1.iloc[i]
        p = df_v1.iloc[i-1]
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
    
    print(f" A) V1.0 1D Simplificada (Donchian 10d Exit) : ${m_v1['total_pnl_usd']:+.2f} USD (+{m_v1['total_ret_wallet_pct']:.2f}% carteira) | PF: {m_v1['profit_factor']}")
    print(f" B) V2.0 Prototype (EMA 20 Reversal Exit)    : ${m_v2['total_pnl_usd']:+.2f} USD (+{m_v2['total_ret_wallet_pct']:.2f}% carteira) | PF: {m_v2['profit_factor']}")
    print(f" C) Buy & Hold BTC-USD (5 Anos Passivo)      : +{bnh_ret:.2f}%")
    print(f" D) EMA 200 Long-Only Simples                : ${ema_pnl_usd:+.2f} USD (+{ema_ret_wallet:.2f}% carteira em {len(ema_trades)} trades)")

    print("\n========================================================")
    print(" 5. PROTOCOLO DE VALIDAÇÃO OUT-OF-SAMPLE (OOS)")
    print("========================================================")
    print(" 1. O período In-Sample de 5 anos (BTC-USD 2021-2026) serviu para formular e congelar a V2.0.")
    print(" 2. A V2.0 está formalmente CONGELADA na seguinte configuração:")
    print("    - Entradas: 100% idênticas à V1.0 (Close > EMA200, DonchianHigh30, ADX>=20, Vol>=SMA20)")
    print("    - Saída: Close < EMA20 (Reversão de Média Rápida) OR Stop Loss 2.0x ATR")
    print("    - Sizing: $333.33 USD fixo | Custos: 0.30% round-trip")
    print(" 3. PRÓXIMO PASSO CIENTÍFICO: Testar a V2.0 Congelada no Período Out-of-Sample (OOS) isolado")
    print("    (ex: histórico independente em ETH-USD e SOL-USD sem nenhuma alteração de parâmetros).")

if __name__ == "__main__":
    execute_v1_vs_v2_comparison()
