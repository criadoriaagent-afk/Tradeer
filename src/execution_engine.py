"""
Motor de Execução Oficial de Ordens via API Key de Corretora (Binance / Testnet).
"""
import os
import sys
import ccxt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.notifier import notifier

class LiveExecutionEngine:
    def __init__(self, api_key: str = None, api_secret: str = None, use_testnet: bool = True):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self.use_testnet = use_testnet
        self.is_connected = False
        
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        if self.use_testnet:
            self.exchange.set_sandbox_mode(True) # Ativa o ambiente Binance Testnet
            
        self.verify_connection()

    def verify_connection(self) -> bool:
        """
        Verifica se a API Key é válida e se a conexão com a corretora foi estabelecida.
        """
        if not self.api_key or not self.api_secret:
            self.is_connected = False
            return False
            
        try:
            # Tenta buscar o saldo para validar as credenciais
            balance = self.exchange.fetch_balance()
            self.is_connected = True
            notifier.add_log("SUCCESS", "Conexão com a Corretora (Binance API) estabelecida com sucesso!")
            return True
        except Exception as e:
            self.is_connected = False
            notifier.add_log("WARNING", f"Modo Simulado Interno Ativo (Chaves de API não configuradas no .env)")
            return False

    def get_connection_status(self) -> dict:
        """
        Retorna o relatório visual de conexão para o Dashboard.
        """
        if self.is_connected:
            mode = "BINANCE TESTNET (Simulado Oficial)" if self.use_testnet else "CONTA REAL BINANCE"
            return {
                "status": "CONECTADO",
                "mode": mode,
                "badge": "🟢 CORRETORA CONECTADA",
                "details": f"Execução automática ativa via {mode}"
            }
        else:
            return {
                "status": "MODO_SIMULADO",
                "mode": "SIMULAÇÃO INTERNA",
                "badge": "🔵 SIMULAÇÃO INTERNA ATIVA",
                "details": "Conexão de simulação ativa (Insira as chaves no arquivo .env para operar na Binance)"
            }

if __name__ == '__main__':
    engine = LiveExecutionEngine()
    print(engine.get_connection_status())
