import os
import json
import httpx
from typing import Optional, Dict, Any

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def _build_event_context(event: Optional[Dict[str, Any]]) -> str:
    if not event:
        return ""
    parts = []
    parts.append(f"Evento: {event.get('label')} em {event.get('event_time')}")
    if event.get('image_path'):
        parts.append(f"Imagem: {event.get('image_path')}")
    weather = event.get('weather')
    if weather and isinstance(weather, dict):
        cur = weather.get('current')
        if cur:
            parts.append(f"Clima atual: temp={cur.get('temperature')}°C, vento={cur.get('windspeed')}")
    return '\n'.join(parts)


def respond_to_message(message: str, event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Responder à mensagem usando OpenAI se `OPENAI_API_KEY` estiver setada,
    caso contrário tenta usar `llama_cpp` se disponível.
    Retorna dict com chave `text` e possivelmente `provider`.
    """
    # Force using local LLaMA via llama_cpp only. Build prompt with event context.
    context = _build_event_context(event)
    prompt = message
    if context:
        prompt = f"Contexto do evento:\n{context}\n\nPergunta: {message}"

    # determine model path (env override or auto-detect in models/)
    model_path = os.environ.get('LLAMA_MODEL_PATH')
    if not model_path:
        try:
            candidates = []
            for fn in os.listdir('models'):
                if fn.lower().endswith(('.bin', '.ggml', '.ggml.bin', '.gguf')):
                    candidates.append(os.path.join('models', fn))
            if candidates:
                candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
                model_path = candidates[0]
        except Exception:
            model_path = None

    if not model_path:
        return {"error": "no_llama_model_found", "details": "Place a ggml model in ./models/ or set LLAMA_MODEL_PATH", "provider": "llama"}

    try:
        from llama_cpp import Llama
    except Exception as e:
        # fallback to demo responder if llama_cpp is not available
        try:
            demo = _demo_respond(message, event)
            demo['provider'] = demo.get('provider', 'ggml-demo')
            demo['fallback_from'] = 'llama_cpp_not_installed'
            demo['llama_error'] = str(e)
            return demo
        except Exception:
            return {"error": "llama_cpp_not_installed", "details": str(e), "provider": "llama"}

    try:
        llm = Llama(model_path=model_path)
        resp = llm.create(prompt=prompt, max_tokens=256, temperature=0.2)
        text = ''.join([c.get('text','') for c in resp.get('choices', [])])
        return {"text": text, "provider": "llama_cpp", "model_path": model_path}
    except Exception as e:
        # If model load/runtime fails, fall back to the lightweight demo responder
        try:
            demo = _demo_respond(message, event)
            demo['provider'] = demo.get('provider', 'ggml-demo')
            demo['fallback_from'] = 'llama_runtime_error'
            demo['llama_error'] = str(e)
            return demo
        except Exception:
            return {"error": "llama_runtime_error", "details": str(e), "provider": "llama"}


def _demo_respond(message: str, event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Very small rule-based responder used for demo model file.
    Not a real model — useful for quick local testing without heavy downloads.
    """
    m = message.lower().strip()
    if any(g in m for g in ("oi", "ola", "olá", "hello", "hi")):
        text = "Oi — este é um modelo de demonstração. Pergunte algo sobre o evento ou clima."
    elif "clima" in m or "tempo" in m or "temperature" in m or "temperatura" in m:
        # try to extract real values from event if available
        if event and isinstance(event, dict):
            weather = event.get('weather')
            if weather and isinstance(weather, dict):
                cur = weather.get('current') or {}
                temp = cur.get('temperature') or cur.get('temp') or cur.get('temperature_2m')
                wind = cur.get('windspeed') or cur.get('wind_speed')
                parts = []
                if temp is not None:
                    parts.append(f"{temp}°C")
                if wind is not None:
                    parts.append(f"vento {wind} m/s")
                if parts:
                    text = "Demo: " + " — ".join(parts)
                else:
                    text = "Demo: tenho um evento, mas sem dados climáticos detalhados."
            else:
                text = "Demo: não tenho dados climáticos no evento, posso simular: 25°C, vento 5 km/h."
        else:
            text = "Demo: não tenho dados climáticos aqui, mas posso simular: 25°C, vento 5 km/h."
    else:
        text = f"Demo-echo: recebi '{message}'. (Modelo demo ativo)"
    return {"text": text, "provider": "ggml-demo"}
