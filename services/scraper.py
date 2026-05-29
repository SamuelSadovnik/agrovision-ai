import time
import threading
from typing import Optional, Dict, Any
import httpx

# Serviço simples de scraping/consulta a API pública (Open-Meteo)
# Regras: mínimo 1 requisição a cada 2 segundos por processo, tratamento de erro

_lock = threading.Lock()
_last_call_at = 0.0
_min_interval = 2.0  # segundos entre chamadas para evitar sobrecarga

def fetch_weather(lat: float, lon: float, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Consulta a API pública Open-Meteo e retorna dados estruturados.

    Retorna None em caso de falha (para o chamador poder tratar com 503).
    """
    global _last_call_at
    with _lock:
        now = time.time()
        elapsed = now - _last_call_at
        if elapsed < _min_interval:
            time.sleep(_min_interval - elapsed)
        _last_call_at = time.time()

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current_weather=true"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&hourly=temperature_2m,precipitation"
        "&timezone=auto"
    )

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            # Estrutura mínima e limpeza
            result = {
                "source": "open-meteo",
                "queried_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "latitude": lat,
                "longitude": lon,
                "current": data.get("current_weather"),
                "daily": data.get("daily"),
                "hourly_sample": {
                    "temperature_2m": (data.get("hourly", {}).get("temperature_2m", [])[:24]),
                    "precipitation": (data.get("hourly", {}).get("precipitation", [])[:24])
                }
            }
            return result
    except httpx.HTTPStatusError as e:
        return {"error": "bad_status", "status_code": e.response.status_code}
    except Exception as e:
        return None
