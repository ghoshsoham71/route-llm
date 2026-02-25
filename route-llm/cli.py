# llm_router/cli.py
"""
CLI entry point for route-llm.

Available commands:
  route-llm status [--config router.yaml] [--watch] [--interval N]
  route-llm dashboard [--config router.yaml]

Requires: pip install "route-llm[cli]"
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CLI dependencies missing. Install with: pip install 'route-llm[cli]'"
    ) from exc

from .router import LLMRouter

app = typer.Typer(
    name="route-llm",
    help="Adaptive rate-limit-aware LLM routing. Bring your own clients.",
    add_completion=False,
)
console = Console()


def _build_table(status: dict) -> Table:
    """Render provider status as a Rich table."""
    table = Table(title="LLM Router — Provider Status", show_lines=True)
    table.add_column("Provider", style="bold cyan", no_wrap=True)
    table.add_column("RPM Used")
    table.add_column("RPM Cap")
    table.add_column("TPM Used")
    table.add_column("TPM Cap")
    table.add_column("Headroom")
    table.add_column("Circuit")
    table.add_column("Latency")

    for name, info in status.items():
        rpm_used = info["rpm_used"]
        rpm_limit = info["rpm_limit"]
        tpm_used = info["tpm_used"]
        tpm_limit = info["tpm_limit"]
        headroom = info["headroom_pct"]
        circuit = info["circuit_open"]
        latency = info["avg_latency_ms"]

        # Bar visualisation
        filled = int(10 * (1 - headroom / 100)) if headroom < 100 else 0
        bar = "█" * filled + "░" * (10 - filled)

        circuit_str = "[red]OPEN  ✗[/red]" if circuit else "[green]CLOSED ✓[/green]"
        headroom_str = f"[red]{headroom}%[/red]" if headroom < 10 else f"{headroom}%"

        table.add_row(
            name,
            f"{rpm_used}/{rpm_limit}",
            bar,
            f"{tpm_used:,}/{tpm_limit:,}",
            "",
            headroom_str,
            circuit_str,
            f"{latency}ms",
        )

    return table


async def _fetch_status(config_path: Optional[str]) -> dict:
    router = _load_router(config_path)
    async with router:
        return await router.status()


def _load_router(config_path: Optional[str]) -> LLMRouter:
    if config_path:
        return LLMRouter.from_yaml(config_path)
    return LLMRouter.from_env()


@app.command()
def status(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to router.yaml"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Live-refresh like htop"),
    interval: int = typer.Option(3, "--interval", "-i", help="Refresh interval in seconds"),
) -> None:
    """Show current provider RPM, TPM, headroom, circuit state, and latency."""
    if watch:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                try:
                    st = asyncio.run(_fetch_status(config))
                    live.update(_build_table(st))
                    time.sleep(interval)
                except KeyboardInterrupt:
                    break
    else:
        st = asyncio.run(_fetch_status(config))
        console.print(_build_table(st))


@app.command()
def dashboard(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to router.yaml"),
    port: int = typer.Option(8501, "--port", "-p", help="Port for Streamlit dashboard"),
) -> None:
    """Launch the local Streamlit dashboard."""
    try:
        import streamlit  # noqa: F401  type: ignore[import]
    except ImportError:
        typer.echo(
            "Streamlit is required for the dashboard. "
            "Install with: pip install 'route-llm[dashboard]'",
            err=True,
        )
        raise typer.Exit(1)

    import subprocess
    import sys
    from pathlib import Path

    dashboard_script = Path(__file__).parent / "_dashboard.py"
    env_args = ["--server.port", str(port)]
    if config:
        env_args += ["--", "--config", config]

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_script)] + env_args,
        check=True,
    )
