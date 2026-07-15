# Projeto JOI — Setup Guide for Windows

> Guia completo de instalação automatizada para Windows 10/11.
> Tempo estimado: **5-10 minutos** (dependendo da velocidade da internet).

---

## 🚀 Quick Start (Recomendado)

### Opção A: Duplo-clique (mais fácil)

1. **Baixe o projeto** do GitHub:
   ```powershell
   git clone https://github.com/CryZenXs/Projeto-JOI.git
   cd Projeto-JOI
   ```

2. **Dê duplo-clique em `setup.bat`** no Explorador de Arquivos.
   - O Windows pode mostrar um aviso "SmartScreen" — clique em **"Mais informações"** → **"Executar mesmo assim"**.
   - Se aparecer um aviso de segurança do PowerShell, é normal — o script pede permissão para rodar.

3. **Siga as instruções na tela**:
   - O script instala Python, Git e outras dependências automaticamente (via `winget`).
   - Você precisará colar sua **Groq API key** (obtenha uma gratuita em https://console.groq.com).
   - Opcional: instalar Ollama, Docker, e puxar o modelo Llama 3.1.

4. **Pronto!** O script pergunta se você quer iniciar o servidor ao final.

---

### Opção B: PowerShell (mais controle)

Abra o **PowerShell** (não precisa ser Admin) e execute:

```powershell
# Clone o repositório
git clone https://github.com/CryZenXs/Projeto-JOI.git
cd Projeto-JOI

# Execute o setup
.\setup.ps1
```

Se aparecer erro de **execution policy**, rode:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup.ps1
```

---

### Opção C: Linha de comando com flags

```powershell
# Pular Docker e Ollama (instalação mínima)
.\setup.ps1 -SkipDocker -SkipOllama

# Recriar venv do zero (limpar instalação anterior)
.\setup.ps1 -CleanVenv

# Pular testes (instalação mais rápida)
.\setup.ps1 -SkipTests

# Ver todas as opções
.\setup.ps1 -Help
```

---

## 📋 Pré-requisitos

O script verifica e instala automaticamente:

| Requisito | Versão mínima | Instalação automática? |
|---|---|---|
| **Windows** | 10 (1809+) / 11 | — |
| **PowerShell** | 5.1+ (já vem no Win10/11) | — |
| **Python** | 3.12+ | ✅ via `winget` |
| **Git** | 2.40+ | ✅ via `winget` |
| **pip** | (vem com Python) | ✅ |
| **Docker Desktop** | (opcional) | ✅ via `winget` |
| **Ollama** | (opcional) | ✅ via `winget` |
| **Visual Studio Build Tools** | (apenas se compilar pacotes C) | ❌ manual |

### Quando você precisa instalar VS Build Tools manualmente?

A maioria dos pacotes do projeto tem wheels pré-compilados para Windows. Você só precisará dos Build Tools se vir erros como:
```
error: Microsoft Visual C++ 14.0 or greater is required
```

Instale em: https://visualstudio.microsoft.com/visual-cpp-build-tools/

Marque a opção **"Desktop development with C++"** durante a instalação.

---

## 🔑 O que o script pede durante o setup

### 1. Groq API Key (necessária para LLM)

- **Onde obter:** https://console.groq.com → Sign in → API Keys → Create API Key
- **Formato:** começa com `gsk_` (40+ caracteres)
- **Custo:** gratuito (14.000 requests/dia no plano free)
- **Validação:** o script verifica o formato `gsk_...`
- **Segurança:** a entrada é mascarada (não aparece na tela)

Se você pular esta etapa, o servidor usará o **MockLLMClient** (responde com eco), permitindo testar a API sem custo.

### 2. Porta do servidor (opcional)

- **Default:** `8000`
- Se a porta 8000 estiver ocupada, escolha outra (ex: `8080`, `3000`)

### 3. Ambiente (opcional)

- **`development`** (recomendado): logs verbosos, auto-reload, docs habilitadas
- **`production`**: logs mínimos, sem auto-reload, docs desabilitadas

### 4. Ollama (opcional, recomendado)

- **Por quê?** Quando a Groq cai ou atinge rate limit, o Ollama assume automaticamente.
- **Modelo:** `llama3.1:8b` (4.7 GB de download, roda em background)
- **Requer:** ~8 GB de RAM disponível

### 5. Docker Desktop (opcional)

- **Por quê?** Para rodar PostgreSQL, Redis e Ollama em containers
- **Requer:** reinício do computador após instalação
- **Alternativa:** instalar Postgres e Redis nativamente (mais complexo)

---

## 📂 O que o script cria

Após o setup, você terá:

```
Projeto-JOI/
├── .venv/                    # ← Virtual environment criado pelo script
│   └── Scripts/
│       ├── python.exe        # ← Use este Python para rodar o projeto
│       ├── pip.exe
│       ├── pytest.exe
│       ├── uvicorn.exe
│       └── pre-commit.exe
├── .env                      # ← Arquivo de configuração gerado
├── .env.backup.*             # ← Backup do .env anterior (se houver)
├── .env.example              # ← Template de referência (não editar)
├── app/
├── tests/
├── scripts/
├── docker/
├── setup.ps1                 # ← Script principal
└── setup.bat                 # ← Wrapper para duplo-clique
```

---

## 🎮 Como usar após o setup

### Ativar o ambiente virtual

Toda vez que abrir um novo terminal para trabalhar no projeto:

```powershell
cd Projeto-JOI
.\.venv\Scripts\Activate.ps1
```

Você verá `(.venv)` no prompt, indicando que o ambiente está ativo.

### Iniciar o servidor de desenvolvimento

```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --reload
```

Ou (com venv ativo):

```powershell
uvicorn app.main:app --reload
```

Acesse:
- **API:** http://localhost:8000/api/v1/health
- **Docs (Swagger):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Conversar com a JOI via CLI

```powershell
.\.venv\Scripts\python.exe scripts\chat_cli.py
```

### Rodar o benchmark de latência

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_groq.py --iterations 5
```

### Rodar os testes

```powershell
.\.venv\Scripts\pytest.exe -v
```

### Iniciar Docker Compose (Postgres + Redis + Ollama)

```powershell
docker compose -f docker\docker-compose.yml up -d
```

---

## 🔧 Solução de problemas

### Erro: "winget não é reconhecido como comando"

`winget` vem no Windows 10 (1809+) e Windows 11. Se não estiver disponível:

1. Abra a Microsoft Store
2. Pesquise por **"App Installer"**
3. Atualize para a versão mais recente

Alternativamente, instale os componentes manualmente:
- Python: https://www.python.org/downloads/
- Git: https://git-scm.com/download/win
- Docker: https://docs.docker.com/desktop/install/windows-install/
- Ollama: https://ollama.com/download/windows

### Erro: "Execution policy restricted"

Rode como Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Ou rode o `setup.bat` (faz bypass automaticamente).

### Erro: "Microsoft Visual C++ 14.0 or greater is required"

Alguns pacotes precisam compilar código C. Instale os Build Tools:
1. Baixe: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Marque **"Desktop development with C++"**
3. Reinstale: `.\setup.ps1 -CleanVenv`

### Erro: "Failed to install dependencies"

Causas mais comuns:
1. **Internet instável** — verifique sua conexão
2. **Antivírus bloqueando** — adicione a pasta do projeto às exceções
3. **Conflito de versões** — tente `.\setup.ps1 -CleanVenv`
4. **Permissões** — não rode em pasta do sistema (use `C:\Users\SeuUsuario\...`)

### Erro: "Port 8000 already in use"

Outro processo está usando a porta. Escolha outra na configuração:
```powershell
.\setup.ps1
# Quando pedir "Server port", digite: 8080
```

Ou mate o processo na porta 8000:
```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
Stop-Process -Id <PID_do_processo>
```

### Erro: "Ollama service not running"

Inicie o serviço manualmente:
```powershell
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
```

Verifique se está rodando:
```powershell
Invoke-WebRequest http://localhost:11434/api/tags -UseBasicParsing
```

### O servidor inicia mas `/api/v1/health/ready` retorna "degraded"

Isso é **esperado** se você não configurou todas as dependências. O status "degraded" significa que o servidor está rodando, mas algumas dependências externas (Groq, Ollama, Redis) não estão disponíveis.

Verifique:
- **Groq:** a chave API está correta no `.env`?
- **Ollama:** o serviço está rodando em `http://localhost:11434`?
- **Redis:** (opcional) está rodando? Inicie via Docker ou instale nativamente.

### Python instalado mas `python` não é reconhecido

O `winget` instala Python mas pode não adicionar ao PATH. Solução:
1. Abra "Editar as variáveis de ambiente do sistema"
2. Encontre `Path` em "Variáveis de usuário"
3. Adicione: `C:\Users\SEU_USUARIO\AppData\Local\Programs\Python\Python312\`
4. Adicione: `C:\Users\SEU_USUARIO\AppData\Local\Programs\Python\Python312\Scripts\`
5. Reinicie o PowerShell

---

## 🔄 Reconfigurar após setup inicial

Se você quiser mudar a configuração (ex: adicionar Groq key depois):

```powershell
.\setup.ps1
# O script detecta o .env existente e pergunta se quer reconfigurar
```

Ou edite o `.env` manualmente com qualquer editor de texto.

---

## 📞 Suporte

- **Issues:** https://github.com/CryZenXs/Projeto-JOI/issues
- **Documentação:** https://github.com/CryZenXs/Projeto-JOI#readme
- **Groq API:** https://console.groq.com/docs

---

## 📝 Notas sobre segurança

- O script **não envia suas chaves API** para nenhum servidor — tudo fica local no `.env`.
- O `.env` é incluído no `.gitignore` para que suas chaves não sejam commitadas acidentalmente.
- A entrada de chaves é mascarada (não aparece na tela enquanto você digita).
- O `SECRET_KEY` gerado é criptograficamente aleatório (32 bytes via `RandomNumberGenerator`).

Se você acidentalmente commitar sua chave API:
1. Revogue imediatamente em https://console.groq.com/keys
2. Gere uma nova chave
3. Reconfigure: `.\setup.ps1`
4. Considere usar `git filter-branch` ou BFG Repo-Cleaner para remover do histórico
