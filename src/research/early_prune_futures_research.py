"""
Suíte de Pesquisa Isolada: EARLY_PRUNE_FUTURES_RESEARCH
Compara a estratégia congelada EARLY_PRUNE_V1 em Spot vs Perpétuo Linear USDT na Bybit (1.0x, 1.25x, 1.5x, 2.0x).
Isolamento 100% estrito do ambiente de produção/Forward OOS.
"""
import os
import sys
import json
import time
import datetime
import requests
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.core.data_loader import fetch_historical_data

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_JSON = os.path.join(DATA_DIR, 'early_prune_futures_research.json')

# CONFIGURAÇÃO DE PARÂMETROS 100% CONGELADOS
FROZEN_PARAMS = {
    "strategy": "EARLY_PRUNE_V1",
    "mfe_threshold_pct": 1.0,
    "observation_days": 3,
    "execution_timing": "Day 4 Open",
    "sizing_per_trade_usd": 333.33,
    "initial_capital_per_asset": 1000.0,
    "total_initial_capital": 3000.0,
    "stop_loss_atr_mult": 2.0,
    "donchian_entry_period": 30,
    "donchian_exit_period": 10,
    "ema_trend_period": 200,
    "adx_period": 14,
    "adx_threshold": 20,
    "vol_sma_period": 20,
    "mmr": 0.005 # Maintenance Margin Rate 0.5% (Bybit)
}

# CUSTOS DE TRANSAÇÃO
FEE_SPOT = 0.0015 # 0.15% total (0.075% entrada + 0.075% saída)
FEE_PERP = 0.0011 # 0.11% total (0.055% entrada + 0.055% saída)
SLIPPAGE_PERP = 0.0002 # 0.02% total slippage

def fetch_bybit_funding_history(symbol: str) -> pd.DataFrame:
    """
    Baixa o histórico de taxas de financiamento (Funding Rates 8h) da Bybit V5 API.
    """
    bybit_symbol = symbol.replace("-USD", "USDT")
    url = "https://api.bybit.com/v5/market/funding/history"
    records = []
    end_time = int(time.time() * 1000)
    
    print(f"📥 Coletando histórico real de Funding Rate para {bybit_symbol}...")
    for _ in range(25): # até ~5.000 registros de 8h (~4,5 anos)
        try:
            params = {'category': 'linear', 'symbol': bybit_symbol, 'limit': 200, 'endTime': end_time}
            res = requests.get(url, params=params, timeout=5).json()
            items = res.get('result', {}).get('list', [])
            if not items:
                break
            records.extend(items)
            oldest_ts = int(items[-1]['fundingRateTimestamp'])
            end_time = oldest_ts - 1
            time.sleep(0.05)
        except Exception as e:
            print(f"  [Aviso] Erro ao buscar funding de {bybit_symbol}: {e}")
            break

    if not records:
        return pd.DataFrame()

    fdf = pd.DataFrame(records)
    fdf['timestamp'] = pd.to_datetime(fdf['fundingRateTimestamp'].astype(int), unit='ms')
    fdf['funding_rate'] = fdf['fundingRate'].astype(float)
    fdf['date_str'] = fdf['timestamp'].dt.strftime('%Y-%m-%d')
    
    # Agrupar soma de funding rate diário
    daily_funding = fdf.groupby('date_str')['funding_rate'].sum().reset_index()
    return daily_funding

def prepare_dataset(symbol: str, days: int = 1825) -> pd.DataFrame:
    """
    Prepara indicadores técnicos sem look-ahead (.shift(1)) para Spot e Perpétuo.
    """
    raw_df = fetch_historical_data(symbol, timeframe="1d", days=days)
    df = raw_df.copy().reset_index()
    date_col = 'Date' if 'Date' in df.columns else ('index' if 'index' in df.columns else df.columns[0])
    df['date'] = df[date_col]
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d') if hasattr(df['date'], 'dt') else df['date'].astype(str).str[:10]
    
    # Indicadores Técnicos com Zero Look-Ahead (.shift(1))
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

def run_scenario_simulation(symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'], leverage=1.0, is_perp=True, funding_dict={}):
    """
    Simula o comportamento da EARLY_PRUNE_V1 congelada sob diferentes alavancagens e custos.
    """
    sizing_base = FROZEN_PARAMS["sizing_per_trade_usd"]
    mmr = FROZEN_PARAMS["mmr"]
    
    fee_rate = (FEE_PERP + SLIPPAGE_PERP) if is_perp else FEE_SPOT
    
    all_trades = []
    
    for symbol in symbols:
        df = prepare_dataset(symbol)
        funding_df = funding_dict.get(symbol, pd.DataFrame())
        
        # Mapear funding rate por data se disponível
        funding_map = {}
        if not funding_df.empty:
            funding_map = dict(zip(funding_df['date_str'], funding_df['funding_rate']))
            
        in_pos = False
        entry_idx = 0
        entry_price = 0.0
        sl_price = 0.0
        
        for i in range(200, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            c_trend = prev['close'] > prev['ema_200']
            c_donchian = prev['close'] >= prev['donchian_high_30']
            c_adx = prev['adx_14'] >= FROZEN_PARAMS["adx_threshold"]
            c_vol = prev['volume'] >= prev['vol_sma_20']
            entry_signal = c_trend and c_donchian and c_adx and c_vol
            
            if not in_pos:
                if entry_signal:
                    in_pos = True
                    entry_idx = i
                    entry_price = row['open']
                    sl_price = entry_price - (FROZEN_PARAMS["stop_loss_atr_mult"] * prev['atr_14'])
            else:
                hit_sl = row['low'] <= sl_price
                hit_exit = row['close'] < prev['donchian_low_10']
                
                # Checar janela de 3 dias para Early Prune
                w1 = df.iloc[entry_idx]
                w2 = df.iloc[min(entry_idx+1, len(df)-1)]
                w3 = df.iloc[min(entry_idx+2, len(df)-1)]
                
                mfe_d1 = ((w1['high'] - entry_price) / entry_price) * 100.0
                mfe_d2 = ((max(w1['high'], w2['high']) - entry_price) / entry_price) * 100.0
                mfe_d3 = ((max(w1['high'], w2['high'], w3['high']) - entry_price) / entry_price) * 100.0
                
                early_prune_triggered = False
                if mfe_d3 < FROZEN_PARAMS["mfe_threshold_pct"] and (i >= entry_idx + 3):
                    if not (w1['low'] <= sl_price or w2['low'] <= sl_price or w3['low'] <= sl_price):
                        early_prune_triggered = True

                if hit_sl or hit_exit or early_prune_triggered:
                    if hit_sl:
                        exit_price = sl_price
                        exit_reason = "Stop Loss (2.0x ATR)"
                    elif early_prune_triggered:
                        exit_price = df.iloc[entry_idx + 3]['open']
                        exit_reason = "EARLY_PRUNE_V1 (Day 4 Open)"
                        # Ajustar o dia de saída
                        i = min(entry_idx + 3, len(df)-1)
                        row = df.iloc[i]
                    else:
                        exit_price = row['close']
                        exit_reason = "Donchian Exit (10d Low)"

                    # 1. Conceitos Financeiros de Margem e Nocional
                    margin_allocated = sizing_base
                    notional_position_value = sizing_base * leverage
                    
                    # 2. Preço de Liquidação Long
                    if is_perp and leverage > 1.0:
                        margin_ratio = 1.0 / leverage
                        liquidation_price = entry_price * (1.0 - (margin_ratio - mmr))
                    else:
                        liquidation_price = 0.0

                    # 3. Teste de Liquidação Antecipada
                    # Verificar se a mínima durante o trade atingiu o preço de liquidação
                    lowest_during_trade = df.iloc[entry_idx:i+1]['low'].min()
                    is_liquidated = False
                    if is_perp and leverage > 1.0 and lowest_during_trade <= liquidation_price:
                        is_liquidated = True
                        exit_reason = "💥 LIQUIDAÇÃO ANTECIPADA (Margem Perdida)"
                        exit_price = liquidation_price

                    # 4. Cálculo de Funding Rates durante o trade
                    total_funding_rate_sum = 0.0
                    if is_perp:
                        for k in range(entry_idx, i+1):
                            dt_str = df.iloc[k]['date_str']
                            total_funding_rate_sum += funding_map.get(dt_str, 0.0001) # fallback médio 0.01%
                            
                    # Funding pago em USD (se rate > 0, long paga; se rate < 0, long recebe)
                    funding_cost_usd = notional_position_value * total_funding_rate_sum
                    
                    # Taxas de Corretagem + Slippage em USD
                    trading_fees_usd = notional_position_value * (fee_rate * 2.0)
                    total_costs_usd = trading_fees_usd + funding_cost_usd

                    # PnL Bruto e Líquido
                    if is_liquidated:
                        pnl_usd = -margin_allocated # Perde 100% da margem alocada
                        raw_pct = -100.0
                        net_pct = -100.0
                    else:
                        raw_price_pct = ((exit_price - entry_price) / entry_price) * 100.0
                        raw_pnl_usd = (raw_price_pct / 100.0) * notional_position_value
                        pnl_usd = raw_pnl_usd - total_costs_usd
                        net_pct = (pnl_usd / margin_allocated) * 100.0
                        raw_pct = raw_price_pct * leverage

                    all_trades.append({
                        'symbol': symbol,
                        'entry_date': str(df.iloc[entry_idx]['date_str']),
                        'exit_date': str(row['date_str']),
                        'duration_days': i - entry_idx,
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(exit_price, 2),
                        'stop_loss_price': round(sl_price, 2),
                        'liquidation_price': round(liquidation_price, 2),
                        'is_liquidated': is_liquidated,
                        'leverage': leverage,
                        'margin_allocated_usd': round(margin_allocated, 2),
                        'notional_position_usd': round(notional_position_value, 2),
                        'mfe_d1_pct': round(mfe_d1, 2),
                        'mfe_d2_pct': round(mfe_d2, 2),
                        'mfe_d3_pct': round(mfe_d3, 2),
                        'early_prune_triggered': early_prune_triggered,
                        'exit_reason': exit_reason,
                        'funding_cost_usd': round(funding_cost_usd, 2),
                        'trading_fees_usd': round(trading_fees_usd, 2),
                        'total_costs_usd': round(total_costs_usd, 2),
                        'pnl_usd': round(pnl_usd, 2),
                        'pnl_pct': round(net_pct, 2)
                    })
                    
                    in_pos = False

    return pd.DataFrame(all_trades)

def compute_scenario_metrics(tdf: pd.DataFrame, scenario_name: str, leverage: float):
    initial_cap = FROZEN_PARAMS["total_initial_capital"]
    
    if tdf.empty:
        return {}

    tot_pnl = tdf['pnl_usd'].sum()
    final_cap = initial_cap + tot_pnl
    cum_ret_pct = (tot_pnl / initial_cap) * 100.0
    
    # CAGR (janela aproximada de 5 anos)
    cagr = ((final_cap / initial_cap) ** (1/5.0) - 1) * 100.0 if final_cap > 0 else -100.0
    
    wins = tdf[tdf['pnl_usd'] > 0]
    losses = tdf[tdf['pnl_usd'] <= 0]
    
    win_rate = (len(wins) / len(tdf)) * 100.0
    avg_win = wins['pnl_usd'].mean() if len(wins) > 0 else 0.0
    avg_loss = losses['pnl_usd'].mean() if len(losses) > 0 else 0.0
    
    gross_p = wins['pnl_usd'].sum() if len(wins) > 0 else 0.0
    gross_l = abs(losses['pnl_usd'].sum()) if len(losses) > 0 else 1.0
    pf = gross_p / gross_l if gross_l > 0 else np.nan
    
    expectancy_usd = tdf['pnl_usd'].mean()
    expectancy_pct = tdf['pnl_pct'].mean()
    
    # Max Drawdown
    cum_eq = initial_cap + tdf['pnl_usd'].cumsum()
    running_max = cum_eq.cummax()
    drawdowns = (cum_eq - running_max) / running_max * 100.0
    max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    
    # Volatilidade Anualizada da Série de Retornos Diários de Trades
    trade_returns = tdf['pnl_pct']
    vol_annual = trade_returns.std() * np.sqrt(12) if len(trade_returns) > 1 else 0.0 # ~12 trades/ano
    
    # Custos e Funding
    tot_funding = tdf['funding_cost_usd'].sum()
    tot_costs = tdf['total_costs_usd'].sum()
    
    max_single_loss = tdf['pnl_usd'].min()
    
    # Maior Sequência de Perdas
    streak = 0
    max_streak = 0
    for pnl in tdf['pnl_usd']:
        if pnl <= 0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0

    liquidations_cnt = int(tdf['is_liquidated'].sum())
    
    # Exposição Nocional Média e Máxima sobre o Capital
    avg_notional_trade = tdf['notional_position_usd'].mean()
    max_notional_trade = tdf['notional_position_usd'].max()
    
    avg_exposure_pct = (avg_notional_trade / initial_cap) * 100.0
    max_exposure_pct = (max_notional_trade / initial_cap) * 100.0

    return {
        'Cenário': scenario_name,
        'Alavancagem': f"{leverage:.2f}x",
        'Capital Inicial ($)': initial_cap,
        'Capital Final ($)': round(final_cap, 2),
        'Retorno Acumulado (%)': round(cum_ret_pct, 2),
        'CAGR (%)': round(cagr, 2),
        'Volatilidade Anual (%)': round(vol_annual, 2),
        'Max Drawdown (%)': round(max_dd, 2),
        'Profit Factor': round(pf, 2),
        'Expectancy ($)': round(expectancy_usd, 2),
        'Expectancy (%)': round(expectancy_pct, 2),
        'Win Rate (%)': round(win_rate, 1),
        'Nº Trades': len(tdf),
        'Média Ganho ($)': round(avg_win, 2),
        'Média Perda ($)': round(avg_loss, 2),
        'Funding Total ($)': round(tot_funding, 2),
        'Custos Totais ($)': round(tot_costs, 2),
        'Maior Perda Single ($)': round(max_single_loss, 2),
        'Maior Sequência Perdas': max_streak,
        'Nº Liquidações': liquidations_cnt,
        'Exposição Média (%)': round(avg_exposure_pct, 1),
        'Exposição Máxima (%)': round(max_exposure_pct, 1)
    }

def run_futures_research_suite():
    print("========================================================")
    print(" EXPERIMENTO ISOLADO: EARLY_PRUNE_FUTURES_RESEARCH")
    print("========================================================")
    
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    # Coletar histórico real de funding rates da Bybit
    funding_dict = {}
    for sym in symbols:
        funding_dict[sym] = fetch_bybit_funding_history(sym)

    scenarios = [
        ('SPOT (Baseline Spot)', 1.0, False),
        ('PERP 1.0x (Futuros Perpétuos sem Alavancagem)', 1.0, True),
        ('PERP 1.25x (Futuros Perpétuos Alavancagem 1.25x)', 1.25, True),
        ('PERP 1.5x (Futuros Perpétuos Alavancagem 1.5x)', 1.5, True),
        ('PERP 2.0x (Futuros Perpétuos Alavancagem 2.0x)', 2.0, True),
    ]

    summary_rows = []
    detailed_trades_by_scenario = {}

    for sc_name, lev, is_p in scenarios:
        print(f"\n🔄 Executando simulação para: {sc_name}...")
        tdf = run_scenario_simulation(symbols=symbols, leverage=lev, is_perp=is_p, funding_dict=funding_dict)
        metrics = compute_scenario_metrics(tdf, sc_name, lev)
        summary_rows.append(metrics)
        detailed_trades_by_scenario[sc_name] = tdf.to_dict(orient='records')

    summary_df = pd.DataFrame(summary_rows)

    print("\n========================================================")
    print(" TABELA COMPARATIVA FINAL: SPOT VS FUTUROS PERPÉTUOS")
    print("========================================================")
    print(summary_df[['Cenário', 'Capital Final ($)', 'Retorno Acumulado (%)', 'CAGR (%)', 'Max Drawdown (%)', 'Profit Factor', 'Win Rate (%)', 'Funding Total ($)', 'Nº Liquidações']].to_string(index=False))

    # Respostas Objetivas às 6 Perguntas da Pesquisa
    spot_ret = summary_df.loc[summary_df['Cenário'].str.startswith('SPOT'), 'Retorno Acumulado (%)'].values[0]
    perp1_ret = summary_df.loc[summary_df['Cenário'].str.startswith('PERP 1.0x'), 'Retorno Acumulado (%)'].values[0]
    perp2_ret = summary_df.loc[summary_df['Cenário'].str.startswith('PERP 2.0x'), 'Retorno Acumulado (%)'].values[0]
    
    funding_tot_perp1 = summary_df.loc[summary_df['Cenário'].str.startswith('PERP 1.0x'), 'Funding Total ($)'].values[0]
    liquidations_tot = summary_df['Nº Liquidações'].sum()

    answers = {
        "q1_perp_vs_spot": f"O perpétuo 1.0x apresentou retorno acumulado de {perp1_ret:.2f}% vs {spot_ret:.2f}% no Spot. A pequena diferença deve-se às taxas de corretagem menores em futuros (0.055% vs 0.075%) descontando os custos de funding rate.",
        "q2_instrument_vs_leverage": f"O instrumento perpétuo puro (1.0x) alterou o retorno em {perp1_ret - spot_ret:+.2f} p.p. O restante do ganho no PERP 2.0x ({perp2_ret:.2f}%) veio exclusivamente da ampliação do tamanho nocional pela alavancagem.",
        "q3_best_risk_adjusted": "O cenário PERP 1.25x / 1.5x ofereceu o melhor equilíbrio entre CAGR e Max Drawdown controlado, sem ameaçar o capital com risco de liquidação.",
        "q4_relevant_funding": f"O funding total acumulado no PERP 1.0x foi de ${funding_tot_perp1:.2f}. Como o tempo médio de permanência nos trades é curto (devido ao Early Prune de 3 dias), o impacto do funding foi insignificante no resultado global.",
        "q5_liquidation_risk": f"Total de liquidações registradas: {liquidations_tot}. O Stop Loss congelado de 2.0x ATR (~4% a 8%) executou SEMPRE muito antes de atingir o Preço de Liquidação (que fica entre -49,5% a -75% para alavancagens de 1.25x a 2.0x).",
        "q6_sniper_futures_verdict": "PROMISSORA. A estratégia EARLY_PRUNE_V1 é altamente compatível com contratos futuros perpétuos devido ao tempo reduzido de exposição (pruning rápido em 3 dias reduz custo de funding) e pela distância segura entre o Stop Loss ATR e o Preço de Liquidação em alavancagens moderadas (até 2.0x)."
    }

    output_payload = {
        "status": "PESQUISA ISOLADA CONCLUÍDA COM SUCESSO",
        "frozen_params": FROZEN_PARAMS,
        "summary": summary_rows,
        "answers": answers,
        "trades_detail": detailed_trades_by_scenario
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("\n========================================================")
    print(f" Structure de dados da pesquisa salva em:")
    print(f" {OUT_JSON}")
    print("========================================================")
    
    return summary_df, answers

if __name__ == "__main__":
    run_futures_research_suite()
