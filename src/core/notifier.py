"""
Módulo de Notificações e Alertas Autônomos em Linguagem Simples.
"""
from datetime import datetime

class Notifier:
    def __init__(self):
        self.logs = []
        # Adiciona logs iniciais de boas-vindas
        self.add_log("SUCCESS", "Sistema de Trading Autônomo inicializado com sucesso.")
        self.add_log("INFO", "Gerenciador de risco ativo: Trava de proteção em 3% ativada.")
        
    def add_log(self, type_: str, message: str) -> dict:
        """
        Adiciona uma notificação formatada para o painel do usuário.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {
            "id": len(self.logs) + 1,
            "timestamp": timestamp,
            "type": type_, # 'SUCCESS', 'INFO', 'WARNING', 'BUY', 'SELL'
            "message": message
        }
        self.logs.insert(0, entry) # Insere no topo
        return entry

    def notify_buy(self, asset: str, price: float, stop_loss: float) -> dict:
        msg = f"🟢 OPORTUNIDADE: Robô comprou {asset} a ${price:,.2f} USD. Trava de proteção ajustada em ${stop_loss:,.2f} USD."
        return self.add_log("BUY", msg)

    def notify_sell(self, asset: str, price: float, profit_pct: float) -> dict:
        emoji = "🚀" if profit_pct >= 0 else "🛡️"
        msg = f"{emoji} OPERAÇÃO CONCLUÍDA: Venda realizada em {asset} a ${price:,.2f} USD. Resultado: {profit_pct:+.2f}%."
        return self.add_log("SELL", msg)

    def notify_risk_alert(self, message: str) -> dict:
        msg = f"⚠️ ALERTA DE SEGURANÇA: {message}"
        return self.add_log("WARNING", msg)

    def get_recent_logs(self, limit: int = 15) -> list:
        return self.logs[:limit]

# Instância global do Notificador
notifier = Notifier()
