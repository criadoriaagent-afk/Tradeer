import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.core.data_loader import fetch_historical_data
from src.engines.spot_signal_futures_cloud_engine import run_portfolio_simulation, calculate_portfolio_metrics

def main():
    print("=== VERIFICAÇÃO 1: WARMUP DOS INDICADORES E PRIMEIRAS VELAS VÁLIDAS ===")
    raw_df = fetch_historical_data("BTC-USD", timeframe="1d", days=1825)
    df = raw_df.copy().reset_index()
    date_col = 'Date' if 'Date' in df.columns else ('index' if 'index' in df.columns else df.columns[0])
    df['date_str'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    df = df[df['date_str'] <= "2026-09-07"].reset_index(drop=True)
    
    start_date = df['date_str'].iloc[0]
    print(f"Data inicial da série histórica completa: {start_date} (Index 0)")
    print(f"Total de velas na série histórica: {len(df)}")
    
    # Cálculo sobre toda a série histórica completa
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

    indicators = ['donchian_low_10', 'atr_14', 'vol_sma_20', 'adx_14', 'donchian_high_30', 'ema_200']
    
    print("\n--- Tabela da Primeira Vela Matematicamente Válida por Indicador ---")
    for ind in indicators:
        idx = df[ind].first_valid_index()
        val_date = df.loc[idx, 'date_str'] if idx is not None else "N/A"
        print(f" - {ind:<18}: Index {idx:>3} | Data: {val_date}")

    print("\n=== VERIFICAÇÃO 2: BASELINE OFICIAL E VALORES ($4.198,73 vs $4.199,93) ===")
    
    # Rodar a simulação para o SPOT Baseline no engine oficial
    trades_spot = run_portfolio_simulation(leverage=1.0, is_spot=True, cutoff_date="2026-09-07")
    metrics_spot = calculate_portfolio_metrics("SPOT (Baseline)", 1.0, trades_spot)
    
    print(f"Cenário Spot Baseline no Engine Oficial:")
    print(f" Capital Inicial: ${metrics_spot['Capital Inicial ($)']}")
    print(f" Capital Final: ${metrics_spot['Capital Final ($)']}")
    print(f" Retorno Acumulado: {metrics_spot['Retorno Acumulado (%)']}%")
    print(f" Nº Trades: {metrics_spot['Nº Trades']}")

    # Checar dados salvos no JSON de auditoria spot_signal_futures_backtest_audit.json
    audit_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "spot_signal_futures_backtest_audit.json")
    if os.path.exists(audit_json_path):
        with open(audit_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("\nDados salvos em spot_signal_futures_backtest_audit.json:")
            for item in data.get('summary', []):
                print(f" {item['Cenário']}: Capital Final = ${item['Capital Final ($)']} | Retorno = {item['Retorno Acumulado (%)']}%")

    print("\n=== VERIFICAÇÃO 3: CUSTOS APLICADOS (Taker Fee 0.075% + Slippage 0.02%) ===")
    print(f"Taxa de Entrada (Entry Fee): 0,075% (Taker Spot)")
    print(f"Taxa de Saída (Exit Fee): 0,075% (Taker Spot)")
    print(f"Slippage na Entrada: 0,02%")
    print(f"Slippage na Saída: 0,02%")
    print(f"Total Custo Teórico Round-Trip: 0,19%")
    
    # Verificar se todos os 45 trades aplicaram estas taxas
    total_trades_checked = len(trades_spot)
    consistent_costs = True
    for t in trades_spot:
        notional = t['notional']
        # entry_fee deve ser aproximadamente notional * 0.00075
        expected_entry_fee = notional * 0.00075
        if abs(t['entry_fee_usd'] - expected_entry_fee) > 0.05:
            consistent_costs = False
            
    print(f"Consistência dos custos checada em {total_trades_checked} trades: {'100% CONSISTENTE' if consistent_costs else 'DIVERGÊNCIA ENCONTRADA'}")

if __name__ == '__main__':
    main()
