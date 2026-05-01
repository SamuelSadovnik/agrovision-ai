# AgroVision AI 🚜👁️

Projeto desenvolvido para a disciplina de Engenharia de Software e ADS. O sistema monitora uma fonte de vídeo, detecta objetos relevantes utilizando visão computacional e permite a interpretação operacional do ambiente.

## Tecnologias Utilizadas
* **FastAPI:** Backend e rotas HTTP.
* **YOLO (v8):** Modelo de visão computacional para detecção de objetos (pessoas, carros, etc).
* **OpenCV:** Leitura e processamento de frames do stream de vídeo.
* **SQLite:** Persistência de eventos detectados.
* **Ollama (LLM):** Agente inteligente para análise contextual dos eventos.

## Como rodar o projeto localmente

1. Crie e ative o ambiente virtual:
   
```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .\.venv\Scripts\activate   # Windows

   