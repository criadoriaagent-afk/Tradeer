import numpy as np
import pandas as pd

def calculate_official_sharpe_sortino(daily_equity_series: pd.Series, risk_free_rate: float = 0.0):
    """
    Cálculo Metodológico Oficial e Padronizado de Sharpe e Sortino Anualizados.
    
    METODOLOGIA OFICIAL CONGELADA:
    - Frequência: Diária (252 dias úteis / pregões por ano)
    - Retorno Diário (r_d): (Wallet_t - Wallet_{t-1}) / Wallet_{t-1}
    - Taxa Livre de Risco (r_f): 0.0% a.a.
    - Sharpe Anualizado: (Média(r_d) / Desvio_Padrão_Total(r_d)) * sqrt(252)
    - Sortino Anualizado (Target Downside Deviation com threshold 0.0 sobre todos os N dias):
      - Downside Risk (r_down): sqrt( (1 / N_total) * sum( min(0, r_d)^2 ) )
      - Sortino: (Média(r_d) / Downside Risk) * sqrt(252)
    """
    daily_returns = daily_equity_series.pct_change().dropna()
    if len(daily_returns) == 0:
        return 0.0, 0.0, 0.0
        
    mean_daily = daily_returns.mean()
    std_total = daily_returns.std()
    
    sharpe = (mean_daily / std_total * np.sqrt(252)) if std_total > 0 else 0.0
    
    # Target Downside Deviation oficial (soleira zero sobre N_total)
    neg_returns = np.minimum(0.0, daily_returns)
    target_downside_std = np.sqrt(np.mean(neg_returns**2))
    
    sortino_target = (mean_daily / target_downside_std * np.sqrt(252)) if target_downside_std > 0 else 0.0
    
    # Conditional Downside Deviation (apenas para diagnóstico sobre dias de perda N_losses)
    only_losses = daily_returns[daily_returns < 0]
    cond_downside_std = only_losses.std() if len(only_losses) > 1 else 0.0
    sortino_cond = (mean_daily / cond_downside_std * np.sqrt(252)) if cond_downside_std > 0 else 0.0
    
    return round(sharpe, 4), round(sortino_target, 4), round(sortino_cond, 4)

def test_unit_metrics_suite():
    print("========================================================")
    print(" TESTE UNITÁRIO DE MÉTRICAS: PADRONIZAÇÃO DE SHARPE E SORTINO")
    print("========================================================")
    
    # Caso 1: Série Sintética com Retorno Diário Positivo
    np.random.seed(123)
    rets = np.random.normal(loc=0.001, scale=0.01, size=1000)
    eq_series = pd.Series(1000.0 * np.cumprod(1 + rets))
    
    sh, st_target, st_cond = calculate_official_sharpe_sortino(eq_series)
    
    print(f" Caso 1 (Curva Sintética 1.000 dias - Retorno Médio +0.10%/dia):")
    print(f"   - Sharpe Anualizado Oficial           : {sh}")
    print(f"   - Sortino Anualizado Oficial (Target 0): {st_target}")
    print(f"   - Sortino Condicional (Losses apenas) : {st_cond}")
    
    # Validação do teste unitário
    assert st_target >= sh, f"ERRO: Na métrica oficial Target 0, Sortino ({st_target}) deve ser >= Sharpe ({sh}) para retornos médios positivos!"
    print("   - STATUS TESTE UNITÁRIO: APROVADO (Sortino Target >= Sharpe confirmado)!\n")

if __name__ == "__main__":
    test_unit_metrics_suite()
