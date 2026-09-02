"""
Módulo de Notificações no Celular via Telegram Bot.
"""
import requests
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.notifier import notifier

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
    def send_telegram_alert(self, message: str) -> bool:
        """
        Envia uma notificação instantânea para o celular do usuário via Telegram.
        """
        # Adiciona no registro de logs interno da aplicação
        notifier.add_log("INFO", f"[TELEGRAM] {message}")
        
        if not self.bot_token or not self.chat_id:
            print(f"[Telegram] (Simulação) Alerta enviado: {message}")
            return True
            
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            res = requests.post(url, json=payload, timeout=5)
            return res.status_code == 200
        except Exception as e:
            print(f"[Telegram] Erro ao enviar mensagem: {e}")
            return False

telegram_notifier = TelegramNotifier()

if __name__ == '__main__':
    telegram_notifier.send_telegram_alert("🚨 *TRADEER QUANT:* Sistema conectado à corretora Bybit com sucesso!")
