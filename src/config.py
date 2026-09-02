"""
Configurações Globais do Sistema Quantitativo de Trading.
"""

# Capital e Gerenciamento de Risco
INITIAL_CAPITAL = 1000.0        # Capital Inicial em USD (ou BRL)
RISK_PER_TRADE = 0.01           # Risco Máximo por Operação (1% do capital)
MAX_DAILY_LOSS = 0.03           # Trava de Perda Diária Máxima (3% do capital)
RISK_REWARD_RATIO = 2.0         # Relação Risco/Retorno (Alvo = 2x o Stop Loss)

# Custos Operacionais Simulados
FEE_RATE = 0.001                # Taxa da corretagem por ordem (0.1% ex: Binance Spot)
SLIPPAGE = 0.0005               # Slippage estimado (0.05%)

# Parâmetros de Mercado
SYMBOL = "BTC-USD"              # Ativo padrão
TIMEFRAME = "1d"                # Período gráfico (1 DIA - Menos ruído, maior confiabilidade)
DAYS_BACK = 730                 # Histórico de 2 Anos (730 dias)

# Parâmetros da Estratégia (Retorno à Média: Bollinger + RSI + EMA200)
BOLLINGER_PERIOD = 20           # Período da Média Móvel Simples
BOLLINGER_STD = 2.0             # Desvios Padrão para as Bandas
RSI_PERIOD = 14                 # Período do IFR / RSI
RSI_OVERSOLD = 30.0             # Nível de Sobrevenda mais estrito
RSI_OVERBOUGHT = 70.0           # Nível de Sobrecompra
RISK_REWARD_RATIO = 2.5         # Relação Risco/Retorno (1:2.5)
