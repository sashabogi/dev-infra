"""CLI entry point for dev-infra: Daneel proxy + memory rescue."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _load_config() -> dict:
    paths = [
        Path(os.path.expanduser("~/.dev-infra/config.yaml")),
        Path(__file__).parent.parent / "config.example.yaml",
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    print("No config found. Copy config.example.yaml to ~/.dev-infra/config.yaml")
    sys.exit(1)


def _pid_file() -> Path:
    return Path(os.path.expanduser("~/.dev-infra/daneel.pid"))


def cmd_start(args: argparse.Namespace) -> None:
    """Start Daneel proxy."""
    load_dotenv()
    config = _load_config()

    host = config.get("daneel", {}).get("host", "0.0.0.0")
    port = config.get("daneel", {}).get("port", 8889)

    # Find the project root (where daneel package lives)
    project_root = Path(__file__).parent.parent

    if getattr(args, "foreground", False):
        # Foreground mode: run uvicorn in the current process (for launchd / systemd)
        pid_file = _pid_file()
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))

        def _cleanup(signum, frame):
            pid_file.unlink(missing_ok=True)
            sys.exit(0)

        signal.signal(signal.SIGTERM, _cleanup)
        signal.signal(signal.SIGINT, _cleanup)

        print(f"Daneel starting in foreground on {host}:{port} (PID {os.getpid()})")

        try:
            import uvicorn

            os.chdir(str(project_root))
            uvicorn.run(
                "daneel.server:app",
                host=host,
                port=port,
                log_level="info",
            )
        finally:
            pid_file.unlink(missing_ok=True)
        return

    # Background mode (default)
    pid_file = _pid_file()
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Daneel already running (PID {pid})")
            return
        except OSError:
            pid_file.unlink()

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "daneel.server:app",
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            "info",
        ],
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(proc.pid))
    print(f"Daneel started on {host}:{port} (PID {proc.pid})")


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop Daneel proxy."""
    pid_file = _pid_file()
    if not pid_file.exists():
        print("Daneel is not running")
        return

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Daneel stopped (PID {pid})")
    except OSError:
        print(f"Daneel process {pid} not found (stale PID file)")
    finally:
        pid_file.unlink(missing_ok=True)


def cmd_status(args: argparse.Namespace) -> None:
    """Show status of all services."""
    import httpx

    config = _load_config()
    port = config.get("daneel", {}).get("port", 8889)
    base = f"http://localhost:{port}"

    # Check Daneel
    pid_file = _pid_file()
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Daneel: running (PID {pid})")
        except OSError:
            print("Daneel: stopped (stale PID file)")
    else:
        print("Daneel: stopped")

    # Check health endpoint
    try:
        resp = httpx.get(f"{base}/health", timeout=5.0)
        if resp.status_code == 200:
            print(f"  Health: OK")
        else:
            print(f"  Health: ERROR ({resp.status_code})")
    except httpx.ConnectError:
        print("  Health: UNREACHABLE")
        return

    # Check providers
    try:
        resp = httpx.get(f"{base}/providers", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            for name, info in data.get("providers", {}).items():
                avail = "UP" if info.get("available") else "DOWN"
                reqs = info.get("total_requests", 0)
                fails = info.get("total_failures", 0)
                lat = info.get("avg_latency_ms", 0)
                print(f"  {name}: {avail} ({reqs} reqs, {fails} fails, {lat:.0f}ms avg)")
    except Exception:
        pass


def cmd_costs(args: argparse.Namespace) -> None:
    """Show cost breakdown."""
    import httpx

    config = _load_config()
    port = config.get("daneel", {}).get("port", 8889)

    try:
        resp = httpx.get(f"http://localhost:{port}/costs", timeout=5.0)
        if resp.status_code != 200:
            print(f"Error: {resp.status_code}")
            return

        data = resp.json()
        rolling = data.get("rolling", {})
        lifetime = data.get("lifetime", {})

        print(f"=== Rolling {rolling.get('window_hours', 24)}h Costs ===")
        for name, info in rolling.get("providers", {}).items():
            print(
                f"  {name}: ${info['cost_usd']:.6f} "
                f"({info['requests']} reqs, "
                f"q={info['avg_quality']:.2f}, "
                f"{info['avg_latency_ms']:.0f}ms)"
            )
        print(f"  TOTAL: ${rolling.get('total_cost_usd', 0):.6f}")
        print(f"  Opus equivalent: ${rolling.get('opus_equivalent_usd', 0):.4f}")
        print(
            f"  Savings: ${rolling.get('savings_usd', 0):.4f} "
            f"({rolling.get('savings_pct', 0):.1f}%)"
        )

        print(f"\n=== Lifetime ===")
        print(f"  Requests: {lifetime.get('total_requests', 0)}")
        print(f"  Actual cost: ${lifetime.get('actual_cost_usd', 0):.6f}")
        print(f"  Opus equivalent: ${lifetime.get('opus_equivalent_usd', 0):.4f}")
        print(f"  Total saved: ${lifetime.get('total_saved_usd', 0):.4f}")

    except httpx.ConnectError:
        print("Daneel is not running. Start it with: dev-infra start")


def cmd_search(args: argparse.Namespace) -> None:
    """Search rescued memories."""
    load_dotenv()
    config = _load_config()

    from rescue.engine import RescueEngine

    engine = RescueEngine(config)
    results = engine.search(
        args.query,
        project=args.project,
        category=args.category,
        limit=args.limit,
    )
    engine.close()

    if not results:
        print("No memories found.")
        return

    for r in results:
        imp = r.get("importance", 0)
        cat = r.get("category", "")
        sub = r.get("subcategory", "")
        proj = r.get("project", "")
        content = r.get("content", "")
        print(f"[{imp}/10] {cat}/{sub} ({proj or 'no project'})")
        print(f"  {content}")
        print()


def cmd_rescue(args: argparse.Namespace) -> None:
    """Manually trigger rescue on a text file."""
    load_dotenv()
    config = _load_config()

    file_path = Path(args.file).expanduser()
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    context = file_path.read_text(encoding="utf-8", errors="replace")
    print(f"Rescuing from {file_path} ({len(context)} chars)...")

    from rescue.engine import RescueEngine

    engine = RescueEngine(config)

    async def _run():
        return await engine.rescue_context(
            context,
            project=args.project,
            session_id=None,
        )

    committed = asyncio.run(_run())
    engine.close()

    print(f"\nCommitted {len(committed)} memories:")
    for m in committed:
        print(f"  [{m.importance}/10] {m.category}/{m.subcategory}: {m.content[:100]}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show memory database statistics."""
    load_dotenv()
    config = _load_config()

    from rescue.engine import RescueEngine

    engine = RescueEngine(config)
    stats = engine.get_stats()
    engine.close()

    print(f"=== Memory Database ===")
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"\n  By category:")
    for cat, count in stats.get("by_category", {}).items():
        print(f"    {cat}: {count}")
    print(f"\n  By project:")
    for proj, count in stats.get("by_project", {}).items():
        print(f"    {proj}: {count}")
    print(f"\n  Recent rescue runs:")
    for run in stats.get("recent_runs", []):
        print(
            f"    {run['created_at'][:19]} | {run.get('project', 'n/a')} | "
            f"{run['memories_extracted']} extracted, "
            f"{run['memories_committed']} committed | "
            f"{run['duration_ms']}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dev-infra",
        description="Daneel inference proxy + memory rescue engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start_parser = sub.add_parser("start", help="Start Daneel proxy")
    start_parser.add_argument(
        "--foreground",
        action="store_true",
        default=False,
        help="Run in foreground (for launchd/systemd)",
    )
    sub.add_parser("stop", help="Stop Daneel proxy")
    sub.add_parser("status", help="Show service status")
    sub.add_parser("costs", help="Show cost breakdown")

    search_parser = sub.add_parser("search", help="Search rescued memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--project", "-p", help="Filter by project")
    search_parser.add_argument(
        "--category", "-c", choices=["fact", "decision", "skill"]
    )
    search_parser.add_argument("--limit", "-l", type=int, default=10)

    rescue_parser = sub.add_parser("rescue", help="Rescue memories from a file")
    rescue_parser.add_argument("file", help="Path to text file")
    rescue_parser.add_argument("--project", "-p", help="Project name")

    sub.add_parser("stats", help="Show memory database statistics")

    args = parser.parse_args()

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "costs": cmd_costs,
        "search": cmd_search,
        "rescue": cmd_rescue,
        "stats": cmd_stats,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
