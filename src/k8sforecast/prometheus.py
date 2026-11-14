from __future__ import annotations

import time

import pandas as pd
import requests

from .config import Config

SEL = 'namespace="{ns}",pod=~"{wl}-.*",container!="",container!="POD"'

QUERIES = {
    "cpu_cores": "sum(rate(container_cpu_usage_seconds_total{{" + SEL + "}}[5m]))",
    "mem_gib": "sum(container_memory_working_set_bytes{{" + SEL + "}})/1024/1024/1024",
    "pods": 'count(kube_pod_status_phase{{namespace="{ns}",pod=~"{wl}-.*",phase="Running"}} == 1)',
    # Spring Boot / Micrometer; remove or change if your service exposes a different request metric.
    "rps": 'sum(rate(http_server_requests_seconds_count{{namespace="{ns}",pod=~"{wl}-.*"}}[5m]))',
}

MAX_POINTS = 10000  # Prometheus rejects query_range responses above 11,000 points


def query_range(base_url: str, query: str, start: float, end: float, step: int,
                timeout: int = 60) -> pd.Series:
    """Run a range query in chunks so long lookbacks stay under Prometheus' point limit."""
    chunks, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + MAX_POINTS * step, end)
        resp = requests.get(
            f"{base_url}/api/v1/query_range",
            params={"query": query, "start": cursor, "end": chunk_end, "step": step},
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()["data"]["result"]
        if result:
            values = result[0]["values"]
            chunks.append(pd.Series(
                [float(v) for _, v in values],
                index=pd.to_datetime([t for t, _ in values], unit="s"),
            ))
        cursor = chunk_end + step
    if not chunks:
        return pd.Series(dtype=float)
    s = pd.concat(chunks)
    return s[~s.index.duplicated(keep="last")]


def fetch_workload(cfg: Config) -> pd.DataFrame:
    end = time.time()
    start = end - cfg.data.lookback_days * 86400
    cols = {}
    for name, template in QUERIES.items():
        if name == "rps" and not cfg.data.include_rps:
            continue
        q = template.format(ns=cfg.data.namespace, wl=cfg.data.workload)
        s = query_range(cfg.data.prometheus_url, q, start, end, cfg.data.step_seconds)
        if s.empty:
            print(f"warning: no data for {name}; skipping")
            continue
        cols[name] = s
    if "cpu_cores" not in cols or "mem_gib" not in cols:
        raise RuntimeError("cpu_cores and mem_gib are required; check namespace/workload labels")
    df = pd.DataFrame(cols)
    df.index.name = "timestamp"
    return df
