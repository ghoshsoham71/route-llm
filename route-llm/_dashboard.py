# llm_router/_dashboard.py
"""
Local Streamlit dashboard for route-llm.

Launch via:
  route-llm dashboard --config router.yaml

Or directly:
  streamlit run llm_router/_dashboard.py -- --config router.yaml

Shows live provider headroom bars, requests routed, fallbacks triggered,
and average latency. Auto-refreshes every 3 seconds.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

try:
    import streamlit as st
    import plotly.graph_objects as go  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    print(
        "Dashboard dependencies missing. Install with: pip install 'route-llm[dashboard]'",
        file=sys.stderr,
    )
    sys.exit(1)

from llm_router import LLMRouter


def _load_router() -> LLMRouter:
    config_path = None
    args = sys.argv[1:]
    if "--config" in args:
        idx = args.index("--config")
        config_path = args[idx + 1]
    if config_path:
        return LLMRouter.from_yaml(config_path)
    return LLMRouter.from_env()


async def _get_status(router: LLMRouter) -> dict:
    return await router.status()


def render_dashboard() -> None:
    st.set_page_config(page_title="LLM Router Dashboard", page_icon="🔀", layout="wide")
    st.title("🔀 LLM Router — Live Status")

    router = _load_router()
    placeholder = st.empty()

    while True:
        status = asyncio.run(_get_status(router))

        with placeholder.container():
            cols = st.columns(len(status) or 1)

            for col, (name, info) in zip(cols, status.items()):
                with col:
                    headroom = info["headroom_pct"]
                    circuit = info["circuit_open"]
                    latency = info["avg_latency_ms"]

                    color = "red" if circuit or headroom < 10 else "green" if headroom > 50 else "orange"
                    st.markdown(f"### {name}")
                    st.markdown(
                        f"**Circuit:** {'🔴 OPEN' if circuit else '🟢 CLOSED'}"
                    )

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=headroom,
                            title={"text": "Headroom %"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": color},
                                "steps": [
                                    {"range": [0, 20], "color": "#ffcccc"},
                                    {"range": [20, 60], "color": "#fff3cc"},
                                    {"range": [60, 100], "color": "#ccffcc"},
                                ],
                            },
                        )
                    )
                    fig.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)

                    st.metric("RPM Used", f"{info['rpm_used']} / {info['rpm_limit']}")
                    st.metric("TPM Used", f"{info['tpm_used']:,} / {info['tpm_limit']:,}")
                    st.metric("Avg Latency", f"{latency} ms")

            st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    render_dashboard()
