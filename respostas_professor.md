# Respostas para o professor

Parte 1 — Revisão da Arquitetura

- Camadas presentes:
  - Frontend: sim — `templates/index.html` (apenas exibição do stream e alertas).
  - Backend/API: sim — `app.py` (rotas, processamento e streaming).
  - Banco de dados: sim — SQLite via `save_event` / `list_events` em `app.py`.
  - Serviços internos: sim — processamento de vídeo e thread de detecção estão no backend.
  - Camada de IA/modelo: sim — `YOLO` instanciado em `app.py` (não isolado).
  - Camada de integração externa: implantei `services/scraper.py` e criei `/api/weather`.
  - Camada de web scraping: implementada em `services/scraper.py`.

- Respostas diretas às perguntas:
  - A interface está apenas exibindo dados ou também possui regra de negócio indevida?
    - Está apenas exibindo; a regra de negócio está no backend.
  - O backend concentra a lógica principal do sistema?
    - Sim, o core está em `app.py`.
  - O acesso ao banco está isolado em uma camada própria ou aparece espalhado pelo código?
    - Não está isolado; as chamadas ao SQLite estão em `app.py`.
  - A chamada ao modelo de IA/YOLO está separada da regra de negócio?
    - Não; o modelo é instanciado e usado em `app.py` junto com a lógica de detecção.
  - A nova camada de scraping será implementada como serviço separado ou ficará misturada?
    - Foi implementada como serviço separado (`services/scraper.py`).

Parte 2 — Revisão de Segurança

- Riscos principais que identifiquei:
  - Configurações hardcoded (`MODEL_PATH`, `DB_PATH`, `CAMERA_SOURCE`).
  - Rotas abertas sem autenticação (`/`, `/api/weather`).
  - Imagens gravadas em `static/captures` e servidas publicamente.
  - Falta validação de entradas (ex.: `lat`/`lon`).
  - Tratamento de erro padrão pode expor informações em produção.

Parte 3 — Melhoria do Código Gerado com IA (resumo)

- `app.py` — organização geral
  - Antes: tudo junto (config, DB, inferência, streaming, rotas).
  - Problema: alto acoplamento e dificuldade para testar.
  - O que fiz: isolei a integração externa em `services/scraper.py` e adicionei `/api/weather`.

- Persistência (`save_event` / `list_events`)
  - Antes: SQL direto em `app.py`.
  - Problema: mistura responsabilidades.
  - O que recomendo: mover para `services/db.py` (não movi ainda).

- Inferência (`process_stream` / YOLO)
  - Antes: inferência, regras e IO no mesmo loop.
  - Problema: mistura de responsabilidades.
  - O que recomendo: criar `services/vision.py` com o wrapper do modelo.

- Frontend (polling)
  - Antes: buscava a página inteira e substituía o HTML da barra lateral.
  - Problema: frágil e ineficiente.
  - Sugestão: criar endpoint JSON e atualizar apenas os dados na página.

- Scraping (`services/scraper.py`)
  - O que fiz: função `fetch_weather(lat, lon)` com `httpx`, timeout, tratamento de erro e rate-limit simples.
  - Por que: fornece contexto climático para as detecções.

Parte 4 — Implementação da camada de web scraping

- Dado coletado: previsão/condição meteorológica (Open‑Meteo).
- Por que: contexto climático ajuda a interpretar eventos e priorizar alertas.

- Requisitos técnicos atendidos:
  - Função/serviço separado: `services/scraper.py` com `fetch_weather(lat, lon)`.
  - Fonte pública/gratuita: Open‑Meteo (sem autenticação).
  - Tratamento de erro: retorno `None` ou erro; rota `/api/weather` devolve `503` quando indisponível.
  - Limite de requisições: delay mínimo de 2s entre chamadas (lock simples).
  - Dados estruturados: JSON com `current`, `daily` e `hourly_sample`.
  - Integração: rota `/api/weather` em `app.py`.



como rodar:



python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt


uvicorn app:app --reload --host 127.0.0.1 --port 8000