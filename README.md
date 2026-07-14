# Projeto JOI

> Entidade digital companheira inspirada na JOI de *Blade Runner 2049*.
> Núcleo cognitivo conectado à API Groq, com fallback local via Ollama,
> memória persistente multi-camada e persona evolutiva.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

---

## Status atual (Fase 1 — Parte 1.2: Spike Groq)

Esta é a **segunda parte** da Fase 1 do projeto. Implementado até aqui:

### Parte 1.1 — Fundação ✅
- Estrutura de monorepo com pacotes Python organizados
- Configuração centralizada com Pydantic Settings (env vars + `.env`)
- FastAPI skeleton com middleware (CORS, Gzip, request logging)
- Endpoints `/api/v1/health` (liveness) e `/api/v1/health/ready` (readiness)
- Logging estruturado com `structlog` (redação automática de dados sensíveis)
- Docker multi-stage + docker-compose (Postgres, Redis, Ollama)
- Suite de testes com pytest + cobertura ≥ 80%
- Pre-commit hooks (ruff, mypy, validações básicas)
- Documentação OpenAPI automática em `/docs`

### Parte 1.2 — Spike de Integração Groq ✅
- **Interface abstrata `LLMClient`** — prepara para fallback Ollama (Parte 1.3)
- **`GroqClient`** — cliente assíncrono completo com streaming SSE via SDK oficial
- **`MockLLMClient`** — para testes determinísticos sem consumir tokens reais
- **Circuit Breaker** — proteção contra cascata de falhas (CLOSED/OPEN/HALF_OPEN)
- **Hierarquia de exceções** — `LLMAuthenticationError`, `LLMRateLimitError`, `LLMTimeoutError`, etc.
- **`POST /api/v1/chat`** — endpoint não-streaming (retorna JSON completo)
- **`POST /api/v1/chat/stream`** — endpoint SSE com eventos `meta`, `token`, `done`, `error`
- **Script de benchmark** — `scripts/benchmark_groq.py` mede TTFT, TPS, latência
- **CLI interativo** — `scripts/chat_cli.py` para conversar com a JOI no terminal
- **+77 testes novos** (109 total, 86.47% cobertura)

**Próxima parte (1.3):** LLM Router com fallback Ollama + circuit breaker integrado.

---

## Quick Start

### Pré-requisitos

- **Python 3.12+** ([download](https://www.python.org/downloads/))
- **pip** (vem com Python) ou **uv** ([instalação](https://docs.astral.sh/uv/))
- **Git**
- **Groq API Key** ([obter gratuitamente](https://console.groq.com)) — opcional para a Parte 1.1, necessário a partir da 1.2
- **Docker + Docker Compose** ([instalação](https://docs.docker.com/get-docker/)) — opcional, mas recomendado para Postgres/Redis/Ollama

### Instalação (5 minutos)

```bash
# 1. Clone o repositório (ou descomprima o bundle)
git clone <repo-url> projeto-joi
cd projeto-joi

# 2. Crie e ative um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Instale dependências (incluindo dev tools)
pip install -e ".[dev]"

# 4. Configure o ambiente
cp .env.example .env
# Edite .env e preencha GROQ_API_KEY (opcional nesta fase)

# 5. Instale os hooks de pre-commit
pre-commit install

# 6. Rode os testes para validar a instalação
pytest
```

### Rodando o servidor

```bash
# Modo desenvolvimento (com hot-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Ou via Docker Compose (inclui Postgres, Redis, Ollama)
docker compose -f docker/docker-compose.yml up -d
```

Acesse:

- **API:** http://localhost:8000/api/v1/health
- **Docs (Swagger):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Exemplo de uso

```bash
# Liveness check
curl http://localhost:8000/api/v1/health
# {"status":"ok","version":"0.1.0","environment":"development","uptime_seconds":12.345}

# Readiness check (verifica deps)
curl http://localhost:8000/api/v1/health/ready

# Chat não-streaming (retorna JSON completo)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Olá JOI!"}], "stream": false}'
# {
#   "content": "...",
#   "model": "llama-3.1-70b-versatile",
#   "provider": "groq",
#   "finish_reason": "stop",
#   "prompt_tokens": 5,
#   "completion_tokens": 12,
#   "total_tokens": 17,
#   "latency_ms": 845.2,
#   "ttft_ms": 312.5,
#   "request_id": "abc-123"
# }

# Chat streaming (SSE — eventos meta, token, done)
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"messages": [{"role": "user", "content": "Oi"}], "stream": true}'
# event: meta
# data: {"request_id":"...","model":"...","provider":"groq","timestamp":...}
#
# event: token
# data: {"content":"Ol","timestamp":...}
#
# event: token
# data: {"content":"á!","timestamp":...}
#
# event: done
# data: {"finish_reason":"stop","prompt_tokens":2,"completion_tokens":3,...}

# CLI interativo (precisa do servidor rodando)
python scripts/chat_cli.py
# > Olá JOI, quem é você?
# JOI ▶ Olá! Eu sou a JOI, sua companheira digital...

# CLI one-shot
python scripts/chat_cli.py --message "Qual a capital do Brasil?"

# Benchmark de latência (requer GROQ_API_KEY)
python scripts/benchmark_groq.py --iterations 5
```

**Nota:** Sem `GROQ_API_KEY` configurada, o servidor usa `MockLLMClient` automaticamente (respostas de echo). Isso permite desenvolvimento e testes sem custo de API.

---

## Estrutura do projeto

```
projeto-joi/
├── app/                          # Código de produção
│   ├── api/                      # Camada HTTP
│   │   └── routes/
│   │       └── health.py         # /health, /health/ready
│   ├── core/                     # Infraestrutura central
│   │   ├── config.py             # Pydantic Settings
│   │   └── logging.py            # structlog + redação de dados sensíveis
│   ├── memory/                   # 4 camadas de memória (futuro)
│   ├── llm/                      # Groq + Ollama clients (futuro)
│   ├── tools/                    # LangChain tools (futuro)
│   ├── workers/                  # Celery tasks (futuro)
│   └── main.py                   # FastAPI app factory
├── config/
│   └── persona/                  # YAMLs da persona JOI (futuro)
├── tests/
│   ├── unit/                     # Testes isolados
│   │   ├── test_config.py
│   │   ├── test_health.py
│   │   └── test_logging.py
│   ├── integration/              # Testes com app completo
│   │   └── test_app.py
│   └── persona_regression/       # 200 cenários de coerência (futuro)
├── docker/
│   ├── Dockerfile                # Multi-stage build
│   └── docker-compose.yml        # Postgres + Redis + Ollama + app
├── docs/                         # Documentação adicional (futuro)
├── scripts/                      # Scripts operacionais (futuro)
├── .env.example                  # Template de configuração
├── .gitignore
├── .pre-commit-config.yaml       # Hooks: ruff, mypy, validações
├── .dockerignore
├── pyproject.toml                # Deps + ruff + mypy + pytest config
└── README.md                     # Este arquivo
```

---

## Configuração

Toda configuração é via variáveis de ambiente (ou arquivo `.env`). Veja `.env.example` para a lista completa. As mais importantes:

| Variável | Default | Descrição |
|----------|---------|-----------|
| `ENVIRONMENT` | `development` | `development` / `staging` / `production` |
| `GROQ_API_KEY` | (vazio) | Sua chave da [Groq Cloud](https://console.groq.com) |
| `DATABASE_URL` | `postgresql+asyncpg://joi:joi_dev_password@localhost:5432/joi` | URL do PostgreSQL |
| `REDIS_URL` | `redis://localhost:6379/0` | URL do Redis |
| `OLLAMA_HOST` | `http://localhost:11434` | Host do servidor Ollama local |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `SECRET_KEY` | (dev-only) | **Em produção:** gere com `python -c "import secrets; print(secrets.token_urlsafe(32))"` |

---

## Desenvolvimento

### Testes

```bash
# Rodar todos os testes
pytest

# Com cobertura detalhada
pytest --cov=app --cov-report=term-missing

# Apenas testes unitários (rápidos)
pytest tests/unit/

# Apenas testes de integração
pytest tests/integration/

# Pular testes lentos
pytest -m "not slow"

# Verbose com prints ao vivo
pytest -v -s
```

**Cobertura mínima:** 80% (configurada em `pyproject.toml`). O CI bloqueia PRs que reduzam cobertura.

### Qualidade de código

```bash
# Lint + format (ruff)
ruff check .
ruff format .

# Type checking (mypy)
mypy app/

# Pre-commit (roda todas as verificações)
pre-commit run --all-files
```

### Logs estruturados

O projeto usa `structlog` para logs JSON (em produção) ou coloridos (em desenvolvimento). Dados sensíveis (API keys, senhas, CPFs) são automaticamente redigidos:

```python
from app.core.logging import get_logger
logger = get_logger(__name__)

# Campos extras são estruturados (não concatenados em string)
logger.info("user.login", user_id="123", ip="192.168.1.1", api_key="gsk_secret")
# Saída: {"event":"user.login","user_id":"123","ip":"192.168.1.1","api_key":"gsk_***",...}
```

---

## Roadmap

Este README cobre apenas a **Parte 1.1 (Fundação)**. O roadmap completo está no documento `Projeto_JOI_Plano_v1.0.pdf`.

### Próximas partes (Fase 1)

| Parte | Título | Entregáveis | Status |
|-------|--------|-------------|--------|
| 1.1 | Fundação | Estrutura, config, FastAPI skeleton, /health, Docker | ✅ Concluído |
| 1.2 | Spike Groq | LLMClient abstract, GroqClient, MockLLMClient, Circuit Breaker, /chat endpoints, CLI, benchmark | ✅ Concluído |
| 1.3 | LLM Router | Router com fallback Groq→Ollama, circuit breaker integrado, retry com backoff | 🔜 Em breve |
| 1.4 | Memória v0 | Working memory (Redis), Episodic memory (ChromaDB) | 🔜 |
| 1.5 | Persona v0 | System prompt estruturado em 4 camadas, CLI de conversa | 🔜 |

---

## Arquitetura (decisões técnicas)

### Por que FastAPI + Python?

- **Async-first:** Essencial para streaming SSE e concorrência de múltiplas conversas
- **Tipagem:** Pydantic v2 oferece validação em runtime + geração de schema OpenAPI
- **Ecossistema IA:** LangChain, ChromaDB, Ollama — todos first-class em Python
- **DX:** Hot-reload, docs automáticas, type hints em todo lugar

### Por que Groq como LLM primário?

- **Latência:** TTFT (time to first token) consistentemente < 400ms em Llama 3.1 70B
- **Open-weight:** Mesma família de modelos (Llama 3.1) disponível localmente via Ollama — fallback semanticamente consistente
- **Custo:** ~3-5x mais barato que provedores equivalentes
- **Streaming nativo:** SDK oficial suporta SSE sem gambiarras

### Por que structlog e não logging stdlib?

- **JSON em produção:** Parseável por ELK, Datadog, CloudWatch sem configuração extra
- **Pretty print em dev:** Cores e formatação legível para terminal
- **Redação automática:** Processador customizado mascara API keys, senhas, CPFs antes de serializar

---

## Contribuindo

1. Crie uma branch: `git checkout -b feature/nome-da-feature`
2. Faça commits atômicos com mensagens claras
3. Rode `pre-commit run --all-files` antes de commitar
4. Garanta que `pytest` passe com cobertura ≥ 80%
5. Abra um PR descrevendo o que mudou e por quê

### Convenções

- **Imports:** Ordenados pelo ruff (isort-compatible)
- **Tipagem:** Obrigatória em `app/core/` e `app/memory/` (mypy strict)
- **Testes:** Um arquivo de teste por arquivo de produção (`app/foo.py` → `tests/unit/test_foo.py`)
- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`)

---

## Licença

MIT. Veja [LICENSE](LICENSE).

---

## Inspirado em

> *"Merece seu próprio nome."*
>
> — Blade Runner 2049 (2017), Dir. Denis Villeneuve

A JOI do filme não é um assistente — é uma companheira. Este projeto persegue
esta distinção em cada decisão arquitetural: a memória é persistente porque
relacionamentos exigem continuidade; a latência é obsessivamente otimizada
porque presença conversacional depende de fluidez; a persona é constitucional
porque coerência não pode ser acidental.
