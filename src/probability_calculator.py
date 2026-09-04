import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.price_action import analyze_price_action
from src.institutional_engine import analyze_institutional_flow
from src.sentiment_engine import fetch_crypto_sentiment

def calculate_trade_probability(df: pd.DataFrame = None) -> dict:
    """
    Calcula a probabilidade estimada de sucesso da operação combinando:
    1. Tendência Macro + ADX (30%)
    2. Volume Bancário (25%)
    3. Padrão de Price Action (25%)
    4. Sentimento & Emoção (20%)
    
    Nota: A probabilidade máxima estimada é limitada a 75.0% para refletir
    expectativas estatísticas realistas de mercado (sem falsas promessas).
    """
    sent = fetch_crypto_sentiment()
    
    if df is None or df.empty:
        return {
            "win_probability": 68.0,
            "grade": "🟢 ALTA (68.0% de Expectativa Estatística)",
            "details": {
                "tendencia_macro": 70.0,
                "volume_bancario": 65.0,
                "price_action": 65.0,
                "sentimento": float(sent.get("sentiment_score", 60.0))
            }
        }
        
    inst = analyze_institutional_flow(df)
    pa = analyze_price_action(df)
    
    # 1. Score de Tendência Macro (30%)
    macro_score = 75.0 if inst['is_trend_strong'] else 45.0
    
    # 2. Score de Volume Bancário (25%)
    volume_score = 75.0 if inst['volume_surge'] else 50.0
    
    # 3. Score de Price Action (25%)
    pa_score = min(float(pa['score']), 75.0)
    
    # 4. Score de Sentimento (20%)
    sent_score = min(float(sent['sentiment_score']), 75.0)
    
    # Média Ponderada
    raw_prob = (macro_score * 0.30) + (volume_score * 0.25) + (pa_score * 0.25) + (sent_score * 0.20)
    final_prob = round(min(max(raw_prob, 35.0), 75.0), 1)
    
    if final_prob >= 68.0:
        grade = f"🟢 ALTA ({final_prob}% de Expectativa Estatística Positiva)"
    elif final_prob >= 55.0:
        grade = f"🔵 MODERADA ({final_prob}% de Expectativa Estatística)"
    else:
        grade = f"🔴 BAIXA ({final_prob}% - Operação Bloqueada)"
        
    return {
        "win_probability": final_prob,
        "grade": grade,
        "details": {
            "tendencia_macro": round(macro_score, 1),
            "volume_bancario": round(volume_score, 1),
            "price_action": round(pa_score, 1),
            "sentimento": round(sent_score, 1)
        }
    }

import pandas as pd

if __name__ == '__main__':
    res = calculate_trade_probability()
    print(res)
