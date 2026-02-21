"""Interactive setup wizard for dev-infra."""

from __future__ import annotations

import datetime
import os
import shutil
import socket
import sys
from pathlib import Path

import httpx
import questionary
import yaml
from questionary import Style

# ── Styling ──────────────────────────────────────────────────────────────────

WIZARD_STYLE = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:green"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
        ("separator", "fg:#808080"),
        ("instruction", "fg:#808080"),
    ]
)

# ── ANSI helpers ─────────────────────────────────────────────────────────────

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

CONFIG_DIR = Path(os.path.expanduser("~/.dev-infra"))

BANNER = rf"""
{CYAN}{BOLD}
     _                _        __
  __| | _____   __   (_)_ __  / _|_ __ __ _
 / _` |/ _ \ \ / /   | | '_ \| |_| '__/ _` |
| (_| |  __/\ V /    | | | | |  _| | | (_| |
 \__,_|\___| \_/     |_|_| |_|_| |_|  \__,_|
{RESET}
{DIM}  Daneel inference proxy + memory rescue engine{RESET}
{DIM}  v0.1.0{RESET}
"""

PROVIDERS = [
    {
        "name": "deepseek",
        "label": "DeepSeek",
        "role": "primary_background",
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "needs_key": True,
        "cost_per_million": {"input": 0.14, "output": 0.28},
        "description": "Cheapest viable model — great for background tasks",
    },
    {
        "name": "xai",
        "label": "xAI (Grok)",
        "role": "secondary",
        "endpoint": "https://api.x.ai/v1/chat/completions",
        "model": "grok-4.1-fast",
        "needs_key": True,
        "cost_per_million": {"input": 0.20, "output": 0.50},
        "description": "Cheap + 2M context — $25 free credits on signup",
    },
    {
        "name": "kimi",
        "label": "Kimi (Moonshot)",
        "role": "secondary",
        "endpoint": "https://api.moonshot.ai/v1/chat/completions",
        "model": "kimi-k2-turbo-preview",
        "needs_key": True,
        "cost_per_million": {"input": 0.60, "output": 0.60},
        "description": "Key from platform.moonshot.ai — known auth issues, may not work",
        "known_models": [
            "kimi-k2.5",
            "kimi-k2-turbo-preview",
            "kimi-k2-thinking-turbo",
            "kimi-k2-thinking",
            "moonshot-v1-128k",
            "moonshot-v1-32k",
            "moonshot-v1-8k",
        ],
    },
    {
        "name": "openrouter",
        "label": "OpenRouter",
        "role": "bulk_tasks",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "needs_key": True,
        "cost_per_million": {"input": 0.40, "output": 0.40},
        "description": "Access to 200+ models — routes to cheapest",
    },
    {
        "name": "zai",
        "label": "Z.ai (Zhipu)",
        "role": "bulk_tasks",
        "endpoint": "https://api.zai.chat/v1/chat/completions",
        "model": "glm-4-flash",
        "needs_key": True,
        "cost_per_million": {"input": 0.10, "output": 0.10},
        "description": "GLM family — flash is cheapest, pick 4.7/5.0 from list",
    },
    {
        "name": "cerebras",
        "label": "Cerebras",
        "role": "compaction",
        "endpoint": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama-3.3-70b",
        "needs_key": True,
        "cost_per_million": {"input": 0.60, "output": 0.60},
        "description": "Ultra-fast inference — ideal for extraction",
    },
    {
        "name": "minimax",
        "label": "MiniMax",
        "role": "bulk_tasks",
        "endpoint": "https://api.minimax.chat/v1/text/chatcompletion_v2",
        "model": "MiniMax-Text-01",
        "needs_key": True,
        "cost_per_million": {"input": 0.30, "output": 2.40},
        "description": "Cheap input, expensive output — use for short responses",
    },
    {
        "name": "groq",
        "label": "Groq",
        "role": "compaction",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "needs_key": True,
        "cost_per_million": {"input": 0.59, "output": 0.79},
        "description": "Fast inference on Llama models",
    },
    {
        "name": "together",
        "label": "Together AI",
        "role": "bulk_tasks",
        "endpoint": "https://api.together.xyz/v1/chat/completions",
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "needs_key": True,
        "cost_per_million": {"input": 0.88, "output": 0.88},
        "description": "Wide model selection, competitive pricing",
    },
    {
        "name": "fireworks",
        "label": "Fireworks AI",
        "role": "bulk_tasks",
        "endpoint": "https://api.fireworks.ai/inference/v1/chat/completions",
        "model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "needs_key": True,
        "cost_per_million": {"input": 0.90, "output": 0.90},
        "description": "Fast open-source model hosting",
    },
    {
        "name": "perplexity",
        "label": "Perplexity",
        "role": "research",
        "endpoint": "https://api.perplexity.ai/chat/completions",
        "model": "sonar",
        "needs_key": True,
        "cost_per_million": {"input": 1.00, "output": 1.00},
        "description": "Search-augmented — use for research tasks only",
    },
    {
        "name": "ollama",
        "label": "Ollama (local)",
        "role": "fallback",
        "endpoint": "http://localhost:11434/v1/chat/completions",
        "model": "qwen2.5-coder:7b",
        "needs_key": False,
        "cost_per_million": {"input": 0, "output": 0},
        "description": "Free local fallback — no API key needed",
    },
]


# ── Utilities ────────────────────────────────────────────────────────────────


def _ok(msg: str) -> None:
    print(f"  {GREEN}\u2714{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}\u26a0{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}\u2718{RESET} {msg}")


def _header(step: int, total: int, title: str) -> None:
    print(f"\n{CYAN}{BOLD}[{step}/{total}]{RESET} {BOLD}{title}{RESET}\n")


def _test_provider(endpoint: str, model: str, api_key: str | None) -> tuple[bool, str]:
    """Send a tiny completion request to verify the provider is reachable."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 4,
    }

    try:
        resp = httpx.post(endpoint, json=body, headers=headers, timeout=15.0)
        if resp.status_code == 200:
            return True, "Connection successful"
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
    except httpx.ConnectError:
        return False, "Connection refused (service not running?)"
    except httpx.TimeoutException:
        return False, "Request timed out (15s)"
    except Exception as exc:
        return False, str(exc)[:120]


def _list_ollama_models() -> list[str]:
    """Fetch available models from Ollama's /api/tags endpoint."""
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def _list_provider_models(endpoint: str, api_key: str) -> list[str]:
    """Fetch available models from an OpenAI-compatible /v1/models endpoint."""
    # Derive models URL from chat completions URL
    # e.g. https://api.deepseek.com/v1/chat/completions -> https://api.deepseek.com/v1/models
    models_url = endpoint.rsplit("/chat/completions", 1)[0]
    if not models_url.endswith("/models"):
        models_url = models_url.rstrip("/") + "/models"

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.get(models_url, headers=headers, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            return sorted([m["id"] for m in models if isinstance(m, dict) and "id" in m])
    except Exception:
        pass
    return []


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _is_placeholder(value: str) -> bool:
    """Detect placeholder API keys that aren't real credentials."""
    v = value.lower().replace("-", " ").replace("_", " ")
    # Catch: "your-key-here", "your_api_key_here", "put-key-here", "replace-me", etc.
    placeholder_words = {"your", "here", "replace", "changeme", "todo", "fixme", "example", "placeholder", "insert", "paste"}
    words = set(v.split())
    # If 2+ placeholder words appear, it's almost certainly not a real key
    if len(words & placeholder_words) >= 2:
        return True
    # Also catch exact short patterns
    if v in {"your key here", "your key", "key here", "api key", "sk xxx", "test"}:
        return True
    # Real API keys are usually 20+ chars of alphanumeric/dash. Short values are suspicious.
    if len(value) < 12:
        return True
    return False


def _load_existing_keys() -> dict[str, str]:
    """Load existing API keys from ~/.dev-infra/.env, filtering out placeholders."""
    env_path = CONFIG_DIR / ".env"
    keys = {}
    if env_path.exists():
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        v = v.strip()
                        # Skip empty values and obvious placeholders
                        if not v or _is_placeholder(v):
                            continue
                        keys[k.strip()] = v
        except Exception:
            pass
    return keys


# ── Wizard Steps ─────────────────────────────────────────────────────────────


def step_providers(existing_keys: dict[str, str] | None = None) -> tuple[dict[str, dict], dict[str, str]]:
    """Step 1: Configure and test each provider. Returns (provider_configs, api_keys)."""
    _header(1, 6, "Provider Configuration")

    configured: dict[str, dict] = {}
    api_keys: dict[str, str] = {}

    for prov in PROVIDERS:
        name = prov["name"]
        label = prov["label"]
        role = prov["role"]

        desc = prov.get("description", "")
        print(f"  {BOLD}{label}{RESET} {DIM}({role}){RESET}")
        if desc:
            print(f"    {DIM}{desc}{RESET}")

        if prov["needs_key"]:
            env_var = f"{name.upper()}_API_KEY"
            existing_key = (existing_keys or {}).get(env_var, "")

            if existing_key:
                # Mask the existing key for display
                masked = existing_key[:4] + "..." + existing_key[-4:] if len(existing_key) > 12 else "****"
                print(f"    {DIM}Existing key found: {masked}{RESET}")
                keep_existing = questionary.confirm(
                    f"  Keep existing key for {label}?",
                    default=True,
                    style=WIZARD_STYLE,
                ).ask()
                if keep_existing is None:
                    raise KeyboardInterrupt
                if keep_existing:
                    key = existing_key
                else:
                    key = questionary.password(
                        f"  New API key for {label} (Enter to skip):",
                        style=WIZARD_STYLE,
                    ).ask()
                    if key is None:
                        raise KeyboardInterrupt
                    key = key.strip() or existing_key  # fall back to existing if blank
            else:
                key = questionary.password(
                    f"  API key for {label} (Enter to skip):",
                    style=WIZARD_STYLE,
                ).ask()
                if key is None:
                    raise KeyboardInterrupt
                key = key.strip()

            if not key:
                _warn(f"{label} skipped (no API key)")
                print()
                continue

            # Fetch available models and let user pick
            print(f"  Fetching models...", end="", flush=True)
            available_models = _list_provider_models(prov["endpoint"], key)
            print("\r" + " " * 40 + "\r", end="")  # clear line

            # Merge with known_models if provider defines them
            known = prov.get("known_models", [])
            if known:
                merged = list(dict.fromkeys(known + available_models))  # dedupe, known first
                available_models = merged

            if available_models:
                # Add "custom" option at the end
                default_model = prov["model"]
                choices = list(available_models)
                if default_model in choices:
                    # Move default to top
                    choices.remove(default_model)
                    choices.insert(0, default_model)
                choices.append("── Enter custom model ──")

                model = questionary.select(
                    f"  Pick a model ({len(available_models)} available):",
                    choices=choices,
                    default=choices[0],
                    style=WIZARD_STYLE,
                ).ask()
                if model is None:
                    raise KeyboardInterrupt

                if model == "── Enter custom model ──":
                    model = questionary.text(
                        f"  Model name:",
                        default=prov["model"],
                        style=WIZARD_STYLE,
                    ).ask()
                    if model is None:
                        raise KeyboardInterrupt
                    model = model.strip() or prov["model"]
            else:
                # Fallback to text input if models endpoint not available
                model = questionary.text(
                    f"  Model [{prov['model']}]:",
                    default=prov["model"],
                    style=WIZARD_STYLE,
                ).ask()
                if model is None:
                    raise KeyboardInterrupt
                model = model.strip() or prov["model"]

            # Test connection
            print(f"  Testing {label}...", end="", flush=True)
            ok, msg = _test_provider(prov["endpoint"], model, key)
            print("\r" + " " * 60 + "\r", end="")  # clear line

            if ok:
                _ok(f"{label} connected")
                configured[name] = {
                    "endpoint": prov["endpoint"],
                    "api_key_env": f"{name.upper()}_API_KEY",
                    "model": model,
                    "role": role,
                    "cost_per_million": prov["cost_per_million"],
                }
                api_keys[name] = key
            else:
                _fail(f"{label} failed: {msg}")
                keep = questionary.confirm(
                    f"  Save {label} config anyway (fix key later)?",
                    default=False,
                    style=WIZARD_STYLE,
                ).ask()
                if keep is None:
                    raise KeyboardInterrupt
                if keep:
                    configured[name] = {
                        "endpoint": prov["endpoint"],
                        "api_key_env": f"{name.upper()}_API_KEY",
                        "model": model,
                        "role": role,
                        "cost_per_million": prov["cost_per_million"],
                    }
                    api_keys[name] = key
        else:
            # Ollama — no key needed, check via /api/tags (model-independent)
            print(f"  Checking if Ollama is running...", end="", flush=True)
            try:
                resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
                ok = resp.status_code == 200
                msg = "Connection successful" if ok else f"HTTP {resp.status_code}"
            except httpx.ConnectError:
                ok, msg = False, "Connection refused (Ollama not running?)"
            except Exception as exc:
                ok, msg = False, str(exc)[:120]
            print("\r" + " " * 60 + "\r", end="")

            if ok:
                _ok("Ollama is running")

                models = _list_ollama_models()
                if models:
                    model = questionary.select(
                        "  Pick a model:",
                        choices=models,
                        default=prov["model"] if prov["model"] in models else models[0],
                        style=WIZARD_STYLE,
                    ).ask()
                    if model is None:
                        raise KeyboardInterrupt
                else:
                    model = questionary.text(
                        f"  Model [{prov['model']}]:",
                        default=prov["model"],
                        style=WIZARD_STYLE,
                    ).ask()
                    if model is None:
                        raise KeyboardInterrupt
                    model = model.strip() or prov["model"]

                configured[name] = {
                    "endpoint": prov["endpoint"],
                    "api_key_env": "",
                    "model": model,
                    "role": role,
                    "cost_per_million": prov["cost_per_million"],
                }
            else:
                _warn("Ollama not running (optional — skipping)")

        print()

    if not configured:
        _fail("No providers configured. At least one is required.")
        sys.exit(1)

    return configured, api_keys


def step_failover_chain(configured: dict[str, dict]) -> list[str]:
    """Step 2: Order the failover chain."""
    _header(2, 6, "Failover Chain")

    names = list(configured.keys())
    # Smart default: cheapest providers first, ollama always last
    def _sort_key(name):
        cfg = configured[name]
        cost = cfg.get("cost_per_million", {})
        avg_cost = (cost.get("input", 0) + cost.get("output", 0)) / 2
        # Ollama always last (it's the offline fallback)
        if name == "ollama":
            return (1, 0)
        return (0, avg_cost)

    default_order = sorted(names, key=_sort_key)

    print(f"  Configured providers: {', '.join(names)}")
    print(f"  Default chain: {' -> '.join(default_order)}")
    print()

    if len(default_order) > 1:
        reorder = questionary.confirm(
            "  Reorder the failover chain?",
            default=False,
            style=WIZARD_STYLE,
        ).ask()
        if reorder is None:
            raise KeyboardInterrupt

        if reorder:
            chain = []
            remaining = list(default_order)
            position = 1
            while remaining:
                choice = questionary.select(
                    f"  Position {position}:",
                    choices=remaining,
                    style=WIZARD_STYLE,
                ).ask()
                if choice is None:
                    raise KeyboardInterrupt
                chain.append(choice)
                remaining.remove(choice)
                position += 1
            _ok(f"Chain: {' -> '.join(chain)}")
            return chain

    _ok(f"Chain: {' -> '.join(default_order)}")
    return default_order


def step_quality_gate() -> dict:
    """Step 3: Quality gate settings."""
    _header(3, 6, "Quality Gate")

    print(f"  {DIM}The quality gate validates LLM outputs before returning them.{RESET}")
    print(f"  {DIM}Checks: XML hallucination, formatting violation, prompt injection{RESET}")
    print()

    threshold = questionary.text(
        "  Quality threshold (0.0-1.0) [0.5]:",
        default="0.5",
        validate=lambda v: 0.0 <= float(v) <= 1.0 if _is_float(v) else False,
        style=WIZARD_STYLE,
    ).ask()
    if threshold is None:
        raise KeyboardInterrupt

    on_failure = questionary.select(
        "  On quality failure:",
        choices=[
            questionary.Choice("reject  — return error to caller", value="reject"),
            questionary.Choice("skip    — pass through unvalidated", value="skip"),
        ],
        default="reject",
        style=WIZARD_STYLE,
    ).ask()
    if on_failure is None:
        raise KeyboardInterrupt

    _ok(f"Threshold: {threshold}, on failure: {on_failure}")

    return {
        "threshold": float(threshold),
        "checks": ["xml_hallucination", "formatting_violation", "prompt_injection"],
        "on_failure": on_failure,
    }


def step_rescue(configured: dict[str, dict]) -> dict:
    """Step 4: Rescue engine settings."""
    _header(4, 6, "Rescue Settings")

    print(f"  {DIM}The rescue engine extracts important memories from conversation contexts.{RESET}")
    print()

    provider_names = list(configured.keys())
    default_extraction = "cerebras" if "cerebras" in provider_names else provider_names[0]

    extraction_model = questionary.select(
        "  Provider for extraction:",
        choices=provider_names,
        default=default_extraction,
        style=WIZARD_STYLE,
    ).ask()
    if extraction_model is None:
        raise KeyboardInterrupt

    importance = questionary.text(
        "  Importance threshold (1-10) [7]:",
        default="7",
        validate=lambda v: v.isdigit() and 1 <= int(v) <= 10,
        style=WIZARD_STYLE,
    ).ask()
    if importance is None:
        raise KeyboardInterrupt

    max_chars = questionary.text(
        "  Max context chars [100000]:",
        default="100000",
        validate=lambda v: v.isdigit() and int(v) > 0,
        style=WIZARD_STYLE,
    ).ask()
    if max_chars is None:
        raise KeyboardInterrupt

    _ok(f"Extraction: {extraction_model}, threshold: {importance}, max chars: {max_chars}")

    return {
        "extraction_model": extraction_model,
        "importance_threshold": int(importance),
        "db_path": "~/.dev-infra/memory.db",
        "max_context_chars": int(max_chars),
    }


def step_daneel_port() -> dict:
    """Step 5: Daneel proxy port."""
    _header(5, 6, "Daneel Proxy")

    port_str = questionary.text(
        "  Port [8889]:",
        default="8889",
        validate=lambda v: v.isdigit() and 1024 <= int(v) <= 65535,
        style=WIZARD_STYLE,
    ).ask()
    if port_str is None:
        raise KeyboardInterrupt

    port = int(port_str)

    if _port_available(port):
        _ok(f"Port {port} is available")
    else:
        _warn(f"Port {port} is in use (Daneel may already be running — that's OK)")

    return {"host": "0.0.0.0", "port": port}


def step_summary_and_save(
    providers: dict[str, dict],
    api_keys: dict[str, str],
    failover_chain: list[str],
    quality_gate: dict,
    rescue: dict,
    daneel: dict,
) -> None:
    """Step 6: Show summary and save."""
    _header(6, 6, "Summary")

    # Build summary table
    print(f"  {BOLD}Providers:{RESET}")
    for name, conf in providers.items():
        key_status = f"{GREEN}\u2714 key set{RESET}" if name in api_keys else f"{DIM}no key{RESET}"
        if name == "ollama":
            key_status = f"{DIM}local{RESET}"
        print(f"    {name:<12} model={conf['model']:<24} {key_status}")

    print(f"\n  {BOLD}Failover:{RESET}  {' -> '.join(failover_chain)}")
    print(f"  {BOLD}Quality:{RESET}   threshold={quality_gate['threshold']}, on_failure={quality_gate['on_failure']}")
    print(f"  {BOLD}Rescue:{RESET}    provider={rescue['extraction_model']}, importance>={rescue['importance_threshold']}")
    print(f"  {BOLD}Daneel:{RESET}    {daneel['host']}:{daneel['port']}")
    print()

    confirm = questionary.confirm(
        "  Save configuration?",
        default=True,
        style=WIZARD_STYLE,
    ).ask()
    if confirm is None:
        raise KeyboardInterrupt
    if not confirm:
        print(f"\n  {YELLOW}Setup cancelled. No files written.{RESET}")
        return

    # Create config dir
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Backup existing files
    config_path = CONFIG_DIR / "config.yaml"
    env_path = CONFIG_DIR / ".env"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if config_path.exists():
        backup = config_path.with_suffix(f".yaml.bak.{ts}")
        shutil.copy2(config_path, backup)
        _ok(f"Backed up existing config to {backup.name}")

    if env_path.exists():
        backup = env_path.with_suffix(f".env.bak.{ts}")
        shutil.copy2(env_path, backup)
        _ok(f"Backed up existing .env to {backup.name}")

    # Build config dict
    config = {
        "daneel": daneel,
        "providers": providers,
        "quality_gate": quality_gate,
        "failover_chain": failover_chain,
        "rescue": rescue,
        "costs": {
            "rolling_window_hours": 24,
            "db_path": "~/.dev-infra/costs.db",
        },
    }

    # Write config.yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    _ok(f"Wrote {config_path}")

    # Write .env
    env_lines = []
    for name, key in api_keys.items():
        env_lines.append(f"{name.upper()}_API_KEY={key}")
    with open(env_path, "w") as f:
        f.write("\n".join(env_lines) + "\n")
    os.chmod(env_path, 0o600)
    _ok(f"Wrote {env_path} (permissions: 600)")

    print(f"\n  {GREEN}{BOLD}Setup complete!{RESET}")
    print(f"\n  Next steps:")
    print(f"    {CYAN}dev-infra start{RESET}    Launch Daneel proxy")
    print(f"    {CYAN}dev-infra status{RESET}   Check service health")
    print(f"    {CYAN}dev-infra costs{RESET}    View cost breakdown")
    print()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


# ── Main entry ───────────────────────────────────────────────────────────────


def run_wizard() -> None:
    """Run the full interactive setup wizard."""
    print(BANNER)

    # Detect existing config
    existing_config = CONFIG_DIR / "config.yaml"
    if existing_config.exists():
        _warn(f"Existing config found at {existing_config}")
        proceed = questionary.confirm(
            "  Reconfigure? (existing config will be backed up)",
            default=True,
            style=WIZARD_STYLE,
        ).ask()
        if proceed is None:
            raise KeyboardInterrupt
        if not proceed:
            print(f"  {DIM}Keeping existing config.{RESET}")
            return
        print()

    try:
        existing_keys = _load_existing_keys()
        providers, api_keys = step_providers(existing_keys)
        failover_chain = step_failover_chain(providers)
        quality_gate = step_quality_gate()
        rescue = step_rescue(providers)
        daneel = step_daneel_port()
        step_summary_and_save(providers, api_keys, failover_chain, quality_gate, rescue, daneel)
    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Setup cancelled.{RESET}")
        sys.exit(0)
