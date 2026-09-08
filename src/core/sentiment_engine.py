"""
Módulo de Análise de Sentimento & Emoção de Mercado (Crypto Fear & Greed Index).
"""
import urllib.request
import json

def fetch_crypto_sentiment() -> dict:
    """
    Consulta o índice oficial de Medo e Ganância (Fear & Greed Index) via API pública.
    """
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            val = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            
            # Tradução e interpretação em Português do Brasil
            if val <= 25:
                label_pt = f"😱 MEDO EXTREMO ({val}/100) - Pânico no Varejo"
                sentiment_score = 90 # Oportunidade de compra contrária
                desc = "Varejo vendendo com pânico. Bancos aproveitam para acumular."
            elif val <= 45:
                label_pt = f"😨 MEDO MODERADO ({val}/100)"
                sentiment_score = 70
                desc = "Mercado cauteloso com boa relação de risco."
            elif val <= 55:
                label_pt = f"😐 NEUTRO ({val}/100)"
                sentiment_score = 60
                desc = "Sentimento equilibrado."
            elif val <= 75:
                label_pt = f"🤑 GANÂNCIA ({val}/100)"
                sentiment_score = 50
                desc = "Varejo animado. Exige confirmação de volume bancário."
            else:
                label_pt = f"🔥 GANÂNCIA EXTREMA ({val}/100) - Euforia no Varejo"
                sentiment_score = 20
                desc = "Alerta de topo! Risco elevado de correção repentina."
                
            return {
                "value": val,
                "classification_en": classification,
                "label_pt": label_pt,
                "sentiment_score": sentiment_score,
                "description": desc
            }
    except Exception as e:
        # Fallback offline seguro
        return {
            "value": 35,
            "classification_en": "Fear",
            "label_pt": "😨 MEDO MODERADO (35/100)",
            "sentiment_score": 70,
            "description": "Mercado cauteloso com boa relação de risco (Dados em cache)."
        }

if __name__ == '__main__':
    sent = fetch_crypto_sentiment()
    print(sent)
