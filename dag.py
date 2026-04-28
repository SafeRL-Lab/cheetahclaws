"""Build a dependency DAG from tool calls and execute them level-by-level with parallelism.

Respects concurrent_safe flags: safe tools run in parallel within a level,
unsafe tools run sequentially after the parallel batch completes.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from tools import execute_tool


def _build_alias_map(tool_calls: list[dict]) -> dict[str, str]:
    alias_to_id: dict[str, str] = {}
    for tc in tool_calls:
        alias = tc["input"].pop("tool_call_alias", None)
        if alias:
            alias_to_id[alias] = tc["id"]
    return alias_to_id


def _build_dag_levels(tool_calls: list[dict]) -> tuple[list[list[dict]], dict[str, set[str]]]:
    """Return (levels, deps). Levels order tcs by dependency; deps maps tc_id -> its prerequisites."""
    if not tool_calls:
        return [], {}

    by_id: dict[str, dict] = {tc["id"]: tc for tc in tool_calls}
    tc_ids_in_turn = set(by_id.keys())

    alias_to_id = _build_alias_map(tool_calls)

    deps: dict[str, set[str]] = {}
    for tc in tool_calls:
        raw = tc["input"].pop("depends_on", None) or []
        resolved = [alias_to_id.get(d, d) for d in raw]
        deps[tc["id"]] = {d for d in resolved if d in tc_ids_in_turn}

    _add_implicit_write_deps(tool_calls, deps)

    remaining = set(by_id.keys())
    levels: list[list[dict]] = []
    while remaining:
        ready = {nid for nid in remaining
                 if not (deps.get(nid, set()) & remaining)}
        if not ready:
            levels.append([by_id[nid] for nid in remaining])
            break
        levels.append([by_id[nid] for nid in ready])
        remaining -= ready
    return levels, deps


def _compute_downstream(deps: dict[str, set[str]], seed_ids: set[str]) -> set[str]:
    """Return seed_ids plus every tc_id transitively depending on any seed."""
    dependents: dict[str, set[str]] = {}
    for tc_id, prereqs in deps.items():
        for prereq in prereqs:
            dependents.setdefault(prereq, set()).add(tc_id)

    downstream = set(seed_ids)
    frontier = list(seed_ids)
    while frontier:
        node = frontier.pop()
        for child in dependents.get(node, ()):
            if child not in downstream:
                downstream.add(child)
                frontier.append(child)
    return downstream


def _add_implicit_write_deps(
    tool_calls: list[dict],
    deps: dict[str, set[str]],
) -> None:
    last_write: dict[str, str] = {}
    for tc in tool_calls:
        if tc["name"] not in ("Write", "Edit", "NotebookEdit"):
            continue
        fp = os.path.normpath(tc["input"].get("file_path", tc["input"].get("notebook_path", "")))
        if not fp:
            continue
        prev = last_write.get(fp)
        if prev is not None:
            deps.setdefault(tc["id"], set()).add(prev)
        last_write[fp] = tc["id"]


def _execute_level(
    level: list[dict],
    results: dict[str, str],
    durations: dict[str, float],
    config: dict,
) -> None:
    from tool_registry import is_concurrent_safe

    if len(level) == 1:
        tc = level[0]
        t0 = time.monotonic()
        results[tc["id"]] = execute_tool(
            tc["name"], tc["input"],
            permission_mode="accept-all", config=config)
        durations[tc["id"]] = time.monotonic() - t0
        return

    parallel_tcs = [tc for tc in level if is_concurrent_safe(tc["name"])]
    sequential_tcs = [tc for tc in level if not is_concurrent_safe(tc["name"])]

    if parallel_tcs:
        _run_parallel(parallel_tcs, results, durations, config)

    for tc in sequential_tcs:
        t0 = time.monotonic()
        results[tc["id"]] = execute_tool(
            tc["name"], tc["input"],
            permission_mode="accept-all", config=config)
        durations[tc["id"]] = time.monotonic() - t0


def _run_parallel(
    tcs: list[dict],
    results: dict[str, str],
    durations: dict[str, float],
    config: dict,
) -> None:
    if len(tcs) == 1:
        tc = tcs[0]
        t0 = time.monotonic()
        results[tc["id"]] = execute_tool(
            tc["name"], tc["input"],
            permission_mode="accept-all", config=config)
        durations[tc["id"]] = time.monotonic() - t0
        return

    pool = ThreadPoolExecutor(max_workers=len(tcs))
    start_times: dict[str, float] = {}
    for tc in tcs:
        start_times[tc["id"]] = time.monotonic()
    futures = {
        pool.submit(execute_tool, tc["name"], tc["input"],
                    "accept-all", None, config): tc
        for tc in tcs
    }
    try:
        remaining_futs = set(futures)
        while remaining_futs:
            newly_done = {f for f in remaining_futs if f.done()}
            for fut in newly_done:
                tc = futures[fut]
                results[tc["id"]] = fut.result()
                durations[tc["id"]] = time.monotonic() - start_times[tc["id"]]
            remaining_futs -= newly_done
            if remaining_futs:
                time.sleep(0.1)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
