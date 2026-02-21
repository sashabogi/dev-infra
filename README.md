# dev-infra

**System-wide development infrastructure.**
*The invisible layer that makes everything else work.*

dev-infra is a companion project to [Foundation](https://github.com/sashabogi/foundation) -- a TypeScript MCP server with 41 tools for AI-assisted development. Where Foundation provides interactive, query-driven capabilities (codebase intelligence, multi-agent orchestration, structured memory), dev-infra adds the autonomous, event-driven infrastructure underneath: transparent inference routing, automatic memory capture, and safety automation.

All three components are named after characters from Isaac Asimov's *Foundation* series, continuing the naming convention established by Foundation's own tools (Demerzel, Seldon, Gaia).

---

## The Problem

Three things go wrong when you use Claude Code with a Max subscription at scale:

1. **Everything burns your weekly quota.** Background tasks, sub-agents, automated scripts -- they all consume the same premium Opus tokens you need for interactive work. There is no built-in way to route low-priority work to cheaper models.

2. **Context compaction destroys knowledge.** When Claude Code summarizes old messages to free tokens, specific facts vanish: URLs, config values, decision rationale, debugging discoveries. That knowledge is gone unless something captures it first.

3. **You forget to use your own tools.** You build 41 MCP tools but forget to invoke them. Safety checks, memory captures, and session checkpoints should fire automatically -- not depend on you remembering.

dev-infra solves all three.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Your Dev Machine                                │
│                                                  │
│  Interactive work --> Claude Code (Max Sub, $0)   │
│                                                  │
│  Automated work --> Daneel Proxy :8889            │
│       ├── DeepSeek     (primary, pennies)        │
│       ├── MiniMax      (bulk, pennies)           │
│       ├── Cerebras     (fast extraction, pennies)│
│       └── Ollama       (offline fallback, free)  │
│                                                  │
│  On compaction --> Memory Rescue (auto via hooks) │
│       └── SQLite + FTS5 (~/.dev-infra/memory.db) │
│                                                  │
│  On every tool call --> Safety gate (auto via     │
│                         hooks)                   │
└─────────────────────────────────────────────────┘
```

---

## Components

### Daneel -- Inference Routing Proxy

Named after R. Daneel Olivaw, the robot who secretly managed the galaxy for 20,000 years without anyone knowing. Daneel sits between your applications and LLM providers, invisibly routing traffic to the cheapest viable model.

- **FastAPI service** running at `localhost:8889`
- **Dual API compatibility** -- speaks both Anthropic Messages API (`/v1/messages`) and OpenAI Chat Completions (`/v1/chat/completions`)
- **Quality gate** scores every response (0.0-1.0), catching XML hallucinations, prompt injection, and incoherence. Failed responses are rejected and retried with the next provider in the failover chain. Daneel never escalates to Opus.
- **Circuit breaker** -- 3 consecutive failures from a provider triggers a 60-second cooldown
- **Safety layer** blocks destructive commands and scrubs credentials from responses
- **Cost tracking** with rolling 24-hour breakdown and Opus-equivalent savings calculator

To use Daneel with any project, add one line to the project's `.env`:

```
LLM_BASE_URL=http://localhost:8889/v1
```

The application thinks it is talking to an LLM provider. Daneel handles everything else.

### Memory Rescue -- Autonomous Knowledge Extraction

Memory Rescue fires automatically when Claude Code performs context compaction. It intercepts the pre-compaction context and extracts durable facts before they are destroyed.

Three parallel extractors run on every compaction event:

| Extractor | What It Captures |
|-----------|-----------------|
| **FactExtractor** | Names, dates, URLs, config values, version numbers |
| **DecisionExtractor** | What was chosen, what was rejected, and why |
| **SkillExtractor** | Debugging techniques, workarounds, patterns, gotchas |

Extraction calls route through Daneel to Cerebras (2000+ tokens/sec, pennies per run). Each memory is scored for importance on a 1-10 scale and filtered at a configurable threshold (default 7). Deduplication uses SHA-256 hashing. Storage is SQLite with FTS5 full-text search and BM25 ranking.

Memories are tagged by project and searchable across all projects.

### Hooks -- Claude Code Automation

Shell scripts that fire on Claude Code lifecycle events. No manual invocation required.

| Hook | Event | What It Does |
|------|-------|-------------|
| `pre-tool-safety.sh` | PreToolCall | Blocks `rm -rf /`, `dd`, fork bombs, `curl \| sh` before execution |
| `post-compaction.sh` | PostToolCall | Detects compaction events, dispatches Memory Rescue in background |
| `session-end.sh` | Stop | Final context extraction on session exit |
| `session-start.sh` | Manual | Loads relevant rescued memories for current project |

---

## Relationship to Foundation

Foundation is a TypeScript MCP server providing 41 tools organized into three modules:

- **Demerzel** -- Codebase intelligence (snapshots, symbol search, import graphs)
- **Seldon** -- Multi-agent orchestration (13+ providers, role-based agents, verification loops)
- **Gaia** -- Advanced memory (SQLite + FTS5, 5-tier hierarchy, BM25 + composite scoring)

dev-infra adds what Foundation cannot do as an MCP server:

| Capability | Foundation (MCP) | dev-infra |
|-----------|-----------------|-----------|
| Codebase search | Demerzel (9 tools) | -- |
| Multi-agent orchestration | Seldon (12 tools) | -- |
| Structured memory | Gaia (20 tools) | -- |
| Transparent inference routing | -- | Daneel proxy |
| Autonomous memory capture | -- | Memory Rescue + hooks |
| Safety automation | -- | PreToolCall hooks |
| Cost optimization | -- | Daneel quality gate + routing |

Together they form the complete stack:

```
Foundation  (interactive, query-driven)
  + dev-infra (autonomous, event-driven)
  = Complete AI development infrastructure
```

When both are installed, the integration points are:

- **Seldon + Daneel** -- Seldon routes background agent calls through Daneel for cost optimization
- **Memory Rescue + Gaia** -- Rescued memories can write to Gaia's memory store for unified search
- **Hooks + Foundation tools** -- Hooks automate Foundation tools that should fire on every session

---

## Installation

### Prerequisites

- Python 3.10+
- pip
- Claude Code (recommended, for hooks integration)

### Install

```bash
git clone https://github.com/sashabogi/dev-infra
cd dev-infra
bash install.sh
```

### Configure

```bash
# Add your API keys
vim ~/.dev-infra/.env

# Customize providers and routing
vim ~/.dev-infra/config.yaml
```

### Start

```bash
dev-infra start          # Start Daneel proxy
dev-infra status         # Check health
dev-infra costs          # See 24h cost breakdown
dev-infra search "query" # Search rescued memories
dev-infra stats          # Memory database stats
dev-infra stop           # Stop Daneel proxy
```

### Auto-start on login (macOS)

```bash
launchctl load ~/Library/LaunchAgents/com.dev-infra.daneel.plist
```

---

## Configuration

`~/.dev-infra/config.yaml` -- full reference:

```yaml
daneel:
  host: "0.0.0.0"
  port: 8889

providers:
  deepseek:
    endpoint: "https://api.deepseek.com/v1/chat/completions"
    api_key_env: "DEEPSEEK_API_KEY"
    model: "deepseek-chat"
    role: primary_background
    cost_per_million:
      input: 0.14
      output: 0.28

  minimax:
    endpoint: "https://api.minimax.chat/v1/text/chatcompletion_v2"
    api_key_env: "MINIMAX_API_KEY"
    model: "abab6.5s-chat"
    role: bulk
    cost_per_million:
      input: 0.10
      output: 0.10

  cerebras:
    endpoint: "https://api.cerebras.ai/v1/chat/completions"
    api_key_env: "CEREBRAS_API_KEY"
    model: "llama3.1-70b"
    role: fast_extraction
    cost_per_million:
      input: 0.60
      output: 0.60

  ollama:
    endpoint: "http://localhost:11434/v1/chat/completions"
    model: "llama3.1:8b"
    role: offline_fallback
    cost_per_million:
      input: 0.00
      output: 0.00

quality_gate:
  threshold: 0.5
  on_failure: reject

failover_chain:
  - deepseek
  - minimax
  - cerebras
  - ollama

rescue:
  extraction_model: cerebras
  importance_threshold: 7
  db_path: "~/.dev-infra/memory.db"
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/messages` | POST | Anthropic-compatible inference |
| `/v1/chat/completions` | POST | OpenAI-compatible inference |
| `/rescue` | POST | Trigger memory rescue on arbitrary context |
| `/memories/search` | GET | Search rescued memories (FTS5 + BM25) |
| `/health` | GET | Service health check |
| `/costs` | GET | Rolling 24-hour cost breakdown |
| `/providers` | GET | Provider health and status |
| `/quality` | GET | Quality gate statistics |

---

## Inspired By

This project was inspired by [The Cartu Method](https://github.com/jcartu/rasputin) -- specifically the Memory Rescue and inference routing concepts. Built from scratch for a different architecture: Claude Code Max subscription paired with the Foundation MCP ecosystem.

---

## License

MIT
