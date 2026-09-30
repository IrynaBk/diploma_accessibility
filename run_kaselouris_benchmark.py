import argparse
import csv
import html
import io
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from plot_kaselouris_violation_rates import generate_violation_rate_plots
from route_metrics import AGENT_PROFILES, GraphEdge, haversine_m, load_graph
from routing_implementations import run_registered_algorithm


OBSERVABILITY_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)

SUPPORTED_ALGORITHMS = (
    "kaselouris",
    "kaselouris_robust",
    "heliyon",
    "heliyon_robust",
    "vrupod",
    "robust_consensus",
)

# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

_WORKER_NODES: Dict[int, Tuple[float, float]] = {}
_WORKER_EDGES: Sequence[GraphEdge] = ()


def _init_route_worker(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
) -> None:
    """
    Install a city's graph once per worker process.

    This avoids sending the graph with every individual route task.
    """
    global _WORKER_NODES, _WORKER_EDGES

    _WORKER_NODES = nodes
    _WORKER_EDGES = edges


# ---------------------------------------------------------------------------
# Route evaluation
# ---------------------------------------------------------------------------

def _run_benchmark_case(task: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        city_id,
        area_id,
        path_index,
        path_count,
        start,
        goal,
        agent_name,
        observability_level,
        seed,
        algorithm,
        quality_bias,
        n_blocked,
        uncertainty_weight,
        robust_base_algorithm,
        robust_scenarios,
    ) = task

    started = time.perf_counter()

    algorithm_kwargs: Dict[str, Any] = {
        "city_id": city_id,
        "area_id": area_id,
        "quality_bias": quality_bias,
        "n_blocked": n_blocked,
    }
    if algorithm in {"kaselouris_robust", "heliyon_robust"}:
        algorithm_kwargs["uncertainty_weight"] = uncertainty_weight
    elif algorithm == "robust_consensus":
        algorithm_kwargs.update({
            "base_algorithm": robust_base_algorithm,
            "robust_scenarios": robust_scenarios,
        })
        if robust_base_algorithm in {"kaselouris_robust", "heliyon_robust"}:
            algorithm_kwargs["uncertainty_weight"] = uncertainty_weight

    result = run_registered_algorithm(
        algorithm,
        _WORKER_NODES,
        _WORKER_EDGES,
        start,
        goal,
        agent_name,
        missing_fraction=observability_level,
        seed=seed + path_index + int(observability_level * 100),
        **algorithm_kwargs,
    )

    result["runtime_s"] = round(
        time.perf_counter() - started,
        6,
    )

    summary = _route_summary(
        result,
        start,
        goal,
        _WORKER_NODES,
    )

    return {
        "city_id": city_id,
        "area_id": area_id,
        "path_index": path_index,
        "path_id": f"{city_id}:{path_index + 1}",
        "agent": agent_name,
        "algorithm": algorithm,
        "base_algorithm": result.get("base_algorithm"),
        "blocked_percent": observability_level,
        "success": summary["success"],
        "runtime_s": summary["runtime_s"],
        "length_m": summary["length_m"],
        "travel_cost": summary["travel_cost"],
        "violation_rate": summary["violation_rate"],
        "detour_ratio": summary["detour_ratio"],
        "expanded_nodes": None,
        "start_node": summary["start_node"],
        "goal_node": summary["goal_node"],
        "start_lat": summary["start_lat"],
        "start_lon": summary["start_lon"],
        "goal_lat": summary["goal_lat"],
        "goal_lon": summary["goal_lon"],
        "straight_line_m": summary["straight_line_m"],
        "tier": result.get("tier"),
        "violations": summary["violations"],
    }


# ---------------------------------------------------------------------------
# Graph / path generation
# ---------------------------------------------------------------------------

def graph_components(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
) -> List[Set[int]]:
    adjacency: Dict[int, List[int]] = {
        node_id: []
        for node_id in nodes
    }

    for edge in edges:
        if edge.start in nodes and edge.end in nodes:
            adjacency.setdefault(edge.start, []).append(edge.end)
            adjacency.setdefault(edge.end, []).append(edge.start)

    unseen = set(adjacency)

    components: List[Set[int]] = []

    while unseen:
        root = unseen.pop()

        stack = [root]
        component = {root}

        while stack:
            current = stack.pop()

            for neighbor in adjacency.get(current, []):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components


def generate_paths(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    count: int,
    seed: int,
    minimum_distance_m: float = 500.0,
) -> List[Tuple[int, int]]:
    generator = random.Random(seed)

    components = sorted(
        (
            component
            for component in graph_components(nodes, edges)
            if len(component) >= 2
        ),
        key=len,
        reverse=True,
    )[:10]

    if not components:
        raise ValueError(
            "The graph has no connected component with two nodes."
        )

    paths: List[Tuple[int, int]] = []

    attempts = 0
    max_attempts = count * 500

    while len(paths) < count and attempts < max_attempts:
        attempts += 1

        component = generator.choice(components)

        start, goal = generator.sample(
            tuple(component),
            2,
        )

        if haversine_m(
            nodes[start],
            nodes[goal],
        ) < minimum_distance_m:
            continue

        if (start, goal) not in paths and (goal, start) not in paths:
            paths.append((start, goal))

    if len(paths) < count:
        raise ValueError(
            f"Could generate only {len(paths)} "
            f"of {count} connected paths."
        )

    return paths


# ---------------------------------------------------------------------------
# Route summary
# ---------------------------------------------------------------------------

def _route_summary(
    result: Dict[str, Any],
    start: int,
    goal: int,
    nodes: Dict[int, Tuple[float, float]],
) -> Dict[str, Any]:
    start_lat, start_lon = nodes[start]
    goal_lat, goal_lon = nodes[goal]

    straight_line_m = haversine_m(
        (start_lat, start_lon),
        (goal_lat, goal_lon),
    )

    length_m = result.get("length_m")

    detour_ratio = (
        None
        if length_m is None or straight_line_m <= 0
        else length_m / straight_line_m
    )

    path_node_count = result.get("path_node_count")
    violation_count = result.get("violations")

    edge_count = max(
        (path_node_count - 1)
        if path_node_count is not None
        else 0,
        1,
    )

    violation_rate = (
        None
        if violation_count is None
        else violation_count / edge_count
    )

    return {
        "success": bool(result.get("success")),
        "path_node_count": path_node_count,
        "length_m": length_m,
        "travel_cost": result.get("travel_cost_s"),
        "runtime_s": result.get("runtime_s"),
        "violations": violation_count,
        "violation_rate": violation_rate,
        "detour_ratio": detour_ratio,
        "straight_line_m": straight_line_m,
        "start_node": start,
        "goal_node": goal,
        "start_lat": start_lat,
        "start_lon": start_lon,
        "goal_lat": goal_lat,
        "goal_lon": goal_lon,
    }


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def run_city_kaselouris(
    city_id: str,
    area_id: str,
    path_count: int,
    seed: int,
    minimum_distance_m: float,
    algorithm: str = "kaselouris",
    quality_bias: float = 1.0,
    n_blocked: int = 0,
    uncertainty_weight: float = 1.0,
    workers: int = 1,
    robust_base_algorithm: str = "kaselouris",
    robust_scenarios: int = 1,
) -> List[Dict[str, Any]]:
    print(
        f"\n{'=' * 70}",
        flush=True,
    )
    print(
        f"Starting city: {city_id} | algorithm: {algorithm}",
        flush=True,
    )
    print(
        f"Requested paths: {path_count} | workers: {workers}",
        flush=True,
    )
    print(
        f"{'=' * 70}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------

    print(
        f"[{city_id}] Loading graph...",
        flush=True,
    )

    load_started = time.perf_counter()

    nodes, edges = load_graph(
        city_id,
        area_id,
    )

    load_time = time.perf_counter() - load_started

    print(
        f"[{city_id}] Graph loaded: "
        f"{len(nodes):,} nodes, {len(edges):,} edges "
        f"in {load_time:.2f}s",
        flush=True,
    )

    valid_nodes = {
        node_id
        for edge in edges
        for node_id in (edge.start, edge.end)
    }

    safe_nodes = {
        node_id: coords
        for node_id, coords in nodes.items()
        if node_id in valid_nodes
    }

    # ------------------------------------------------------------------
    # Generate routes
    # ------------------------------------------------------------------

    print(
        f"[{city_id}] Generating {path_count} connected paths...",
        flush=True,
    )

    paths = generate_paths(
        safe_nodes,
        edges,
        path_count,
        seed,
        minimum_distance_m,
    )

    print(
        f"[{city_id}] Generated {len(paths)} paths.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Build tasks
    # ------------------------------------------------------------------

    tasks = [
        (
            city_id,
            area_id,
            path_index,
            path_count,
            start,
            goal,
            agent_name,
            observability_level,
            seed,
            algorithm,
            quality_bias,
            n_blocked,
            uncertainty_weight,
            robust_base_algorithm,
            robust_scenarios,
        )
        for path_index, (start, goal) in enumerate(paths)
        for observability_level in OBSERVABILITY_LEVELS
        for agent_name in AGENT_PROFILES
    ]

    total_tasks = len(tasks)

    print(
        f"[{city_id}] Benchmark tasks: {total_tasks:,}",
        flush=True,
    )

    if total_tasks == 0:
        return []

    # ------------------------------------------------------------------
    # Sequential execution
    # ------------------------------------------------------------------

    if workers <= 1:
        print(
            f"[{city_id}] Running sequentially...",
            flush=True,
        )

        _init_route_worker(
            safe_nodes,
            edges,
        )

        rows: List[Dict[str, Any]] = []

        for completed, task in enumerate(tasks, start=1):
            try:
                row = _run_benchmark_case(task)
                rows.append(row)

                print(
                    f"[{city_id}] "
                    f"{completed}/{total_tasks} "
                    f"path={row['path_index'] + 1}/{path_count} "
                    f"agent={row['agent']} "
                    f"obs={row['blocked_percent']:.2f} "
                    f"{'success' if row['success'] else 'failure'} "
                    f"runtime={row['runtime_s']:.3f}s",
                    flush=True,
                )

            except Exception as exc:
                print(
                    f"[{city_id}] ERROR on task "
                    f"{completed}/{total_tasks}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        return rows

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    print(
        f"[{city_id}] Starting {workers} worker processes...",
        flush=True,
    )

    rows = []

    benchmark_started = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_route_worker,
        initargs=(
            safe_nodes,
            edges,
        ),
    ) as executor:

        # Submit individually instead of using executor.map().
        #
        # This allows as_completed() to report routes immediately,
        # regardless of their order in the original task list.
        future_to_task = {
            executor.submit(
                _run_benchmark_case,
                task,
            ): task
            for task in tasks
        }

        print(
            f"[{city_id}] Submitted {len(future_to_task):,} tasks.",
            flush=True,
        )

        completed = 0

        for future in as_completed(future_to_task):
            task = future_to_task[future]

            completed += 1

            (
                task_city_id,
                task_area_id,
                path_index,
                task_path_count,
                start,
                goal,
                agent_name,
                observability_level,
                task_seed,
                task_algorithm,
                task_quality_bias,
                task_n_blocked,
                task_uncertainty_weight,
                task_robust_base_algorithm,
                task_robust_scenarios,
            ) = task

            try:
                row = future.result()

                rows.append(row)

                elapsed = time.perf_counter() - benchmark_started

                if completed > 0:
                    average_time = elapsed / completed
                    remaining = total_tasks - completed
                    eta = average_time * remaining
                else:
                    eta = 0.0

                print(
                    f"[{city_id}] "
                    f"{completed}/{total_tasks} "
                    f"({completed / total_tasks:.1%}) | "
                    f"path={row['path_index'] + 1}/{path_count} | "
                    f"agent={row['agent']} | "
                    f"obs={row['blocked_percent']:.2f} | "
                    f"{'SUCCESS' if row['success'] else 'FAIL'} | "
                    f"runtime={row['runtime_s']:.3f}s | "
                    f"ETA={eta:.1f}s",
                    flush=True,
                )

            except Exception as exc:
                print(
                    f"[{city_id}] "
                    f"WORKER ERROR "
                    f"{completed}/{total_tasks} | "
                    f"path={path_index + 1}/{task_path_count} | "
                    f"agent={agent_name} | "
                    f"obs={observability_level:.2f} | "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    elapsed = time.perf_counter() - benchmark_started

    print(
        f"[{city_id}] Finished parallel benchmark: "
        f"{len(rows)}/{total_tasks} successful task executions "
        f"in {elapsed:.2f}s",
        flush=True,
    )

    # Keep deterministic output order.
    rows.sort(
        key=lambda row: (
            row["path_index"],
            row["blocked_percent"],
            row["agent"],
        )
    )

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_reports(
    rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "No benchmark rows to report."
        )

    def mean(
        values: List[Optional[float]],
    ) -> Optional[float]:
        filtered = [
            value
            for value in values
            if value is not None
        ]

        if not filtered:
            return None

        return sum(filtered) / len(filtered)

    success_rate = (
        sum(
            1
            for row in rows
            if row.get("success")
        )
        / len(rows)
    )

    successful = [
        row
        for row in rows
        if row.get("success")
    ]

    mean_runtime = mean(
        [
            row.get("runtime_s")
            for row in successful
        ]
    )

    mean_length = mean(
        [
            row.get("length_m")
            for row in successful
        ]
    )

    mean_detour = mean(
        [
            row.get("detour_ratio")
            for row in successful
        ]
    )

    mean_violation = mean(
        [
            row.get("violation_rate")
            for row in successful
        ]
    )

    summary_rows = []

    grouped: Dict[
        Tuple[str, str, str, float],
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        key = (
            row["city_id"],
            row["agent"],
            row["algorithm"],
            float(row["blocked_percent"]),
        )

        grouped[key].append(row)

    for (
        city_id,
        agent,
        algorithm,
        blocked_percent,
    ), group in sorted(grouped.items()):

        successful_group = [
            item
            for item in group
            if item.get("success")
        ]

        summary_rows.append(
            {
                "city_id": city_id,
                "agent": agent,
                "algorithm": algorithm,
                "blocked_percent": blocked_percent,
                "paths": len(group),
                "success_rate": (
                    len(successful_group)
                    / len(group)
                ),
                "mean_runtime_s": mean(
                    [
                        item.get("runtime_s")
                        for item in successful_group
                    ]
                ),
                "mean_length_m": mean(
                    [
                        item.get("length_m")
                        for item in successful_group
                    ]
                ),
                "mean_travel_cost": mean(
                    [
                        item.get("travel_cost")
                        for item in successful_group
                    ]
                ),
                "mean_detour_ratio": mean(
                    [
                        item.get("detour_ratio")
                        for item in successful_group
                    ]
                ),
                "mean_violation_rate": mean(
                    [
                        item.get("violation_rate")
                        for item in successful_group
                    ]
                ),
            }
        )

    rows_json = json.dumps(
        rows,
        default=str,
    )

    summary_json = json.dumps(
        summary_rows,
        default=str,
    )

    benchmark_name = (
        "Kaselouris Benchmark"
        if rows
        and rows[0].get("algorithm") == "kaselouris"
        else (
            f"{rows[0].get('algorithm', 'algorithm').title()} Benchmark"
        )
    )

    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{benchmark_name}</title>
        <style>
            body {{
                font-family: sans-serif;
                padding: 24px;
                background: #f5f7f8;
            }}

            .card {{
                background: white;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 1px 4px rgba(0,0,0,.1);
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
            }}

            th, td {{
                border-bottom: 1px solid #dfe6e8;
                padding: 8px;
                text-align: left;
            }}

            th {{
                background: #12343b;
                color: white;
            }}

            .kpis {{
                display: grid;
                grid-template-columns:
                    repeat(5, minmax(160px, 1fr));
                gap: 16px;
            }}

            .kpi {{
                background: white;
                border-radius: 10px;
                padding: 18px;
                box-shadow: 0 1px 4px rgba(0,0,0,.08);
            }}

            .label {{
                color: #66777c;
                font-size: 12px;
            }}

            .value {{
                font-size: 24px;
                font-weight: 700;
            }}
        </style>
    </head>

    <body>
        <h1>{benchmark_name}</h1>

        <div class="kpis">
            <div class="kpi">
                <div class="label">Benchmark rows</div>
                <div class="value">{len(rows):,}</div>
            </div>

            <div class="kpi">
                <div class="label">Success rate</div>
                <div class="value">
                    {success_rate * 100:.1f}%
                </div>
            </div>

            <div class="kpi">
                <div class="label">Mean runtime</div>
                <div class="value">
                    {
                        (
                            mean_runtime * 1000
                            if mean_runtime is not None
                            else 0
                        )
                    :.2f} ms
                </div>
            </div>

            <div class="kpi">
                <div class="label">Mean length</div>
                <div class="value">
                    {
                        (
                            mean_length / 1000
                            if mean_length is not None
                            else 0
                        )
                    :.2f} km
                </div>
            </div>

            <div class="kpi">
                <div class="label">Mean violations</div>
                <div class="value">
                    {
                        (
                            mean_violation * 100
                            if mean_violation is not None
                            else 0
                        )
                    :.1f}%
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Summary</h2>

            <table>
                <tr>
                    <th>City</th>
                    <th>Agent</th>
                    <th>Algorithm</th>
                    <th>Observability</th>
                    <th>Paths</th>
                    <th>Success</th>
                    <th>Mean runtime</th>
                    <th>Mean length km</th>
                    <th>Mean detour</th>
                    <th>Mean violations</th>
                </tr>

                {
                    ''.join(
                        '<tr>'
                        f'<td>{html.escape(str(item["city_id"]))}</td>'
                        f'<td>{html.escape(str(item["agent"]))}</td>'
                        f'<td>{html.escape(str(item["algorithm"]))}</td>'
                        f'<td>{item["blocked_percent"]:.2f}</td>'
                        f'<td>{item["paths"]}</td>'
                        f'<td>{item["success_rate"]:.1%}</td>'
                        f'<td>{(item["mean_runtime_s"] * 1000 if item["mean_runtime_s"] is not None else 0):.2f}</td>'
                        f'<td>{(item["mean_length_m"] / 1000 if item["mean_length_m"] is not None else 0):.2f}</td>'
                        f'<td>{(item["mean_detour_ratio"] if item["mean_detour_ratio"] is not None else 0):.3f}</td>'
                        f'<td>{(item["mean_violation_rate"] * 100 if item["mean_violation_rate"] is not None else 0):.1f}%</td>'
                        '</tr>'
                        for item in summary_rows
                    )
                }
            </table>
        </div>

        <div class="card">
            <h2>Raw rows</h2>

            <table>
                <tr>
                    <th>City</th>
                    <th>Agent</th>
                    <th>Obs</th>
                    <th>Success</th>
                    <th>Length km</th>
                    <th>Cost</th>
                    <th>Violations</th>
                </tr>

                {
                    ''.join(
                        '<tr>'
                        f'<td>{html.escape(str(item["city_id"]))}</td>'
                        f'<td>{html.escape(str(item["agent"]))}</td>'
                        f'<td>{item["blocked_percent"]}</td>'
                        f'<td>{"✓" if item.get("success") else "✗"}</td>'
                        f'<td>{(item["length_m"] / 1000 if item["length_m"] is not None else 0):.2f}</td>'
                        f'<td>{item["travel_cost"] if item["travel_cost"] is not None else "—"}</td>'
                        f'<td>{item["violations"] if item["violations"] is not None else "—"}</td>'
                        '</tr>'
                        for item in rows
                    )
                }
            </table>
        </div>
    </body>
    </html>
    """

    (output_dir / "report.html").write_text(
        html_doc,
        encoding="utf-8",
    )

    (output_dir / "results.json").write_text(
        rows_json,
        encoding="utf-8",
    )

    (output_dir / "results.csv").write_text(
        _rows_to_csv(rows),
        encoding="utf-8",
    )

    (output_dir / "summary.csv").write_text(
        _summary_to_csv(summary_rows),
        encoding="utf-8",
    )

    plots_dir = output_dir / "plots"

    generate_violation_rate_plots(
        output_dir / "results.csv",
        plots_dir,
    )

    print(
        f"Report: {output_dir / 'report.html'}",
        flush=True,
    )

    print(
        f"Rows: {len(rows):,}",
        flush=True,
    )

    print(
        f"Paths: {len({row.get('path_id') for row in rows})}",
        flush=True,
    )

    print(
        f"Success: {success_rate:.1%}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _rows_to_csv(
    rows: List[Dict[str, Any]],
) -> str:
    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    output = io.StringIO()

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)

    return output.getvalue()


def _summary_to_csv(
    summary_rows: List[Dict[str, Any]],
) -> str:
    fieldnames = [
        "city_id",
        "agent",
        "algorithm",
        "blocked_percent",
        "paths",
        "success_rate",
        "mean_runtime_s",
        "mean_length_m",
        "mean_travel_cost",
        "mean_detour_ratio",
        "mean_violation_rate",
    ]

    output = io.StringIO()

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in summary_rows:
        writer.writerow(row)

    return output.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a routing algorithm across cities "
            "and observability levels."
        )
    )

    parser.add_argument(
        "--algorithm",
        choices=SUPPORTED_ALGORITHMS,
        default="kaselouris",
        help="routing implementation to benchmark",
    )

    parser.add_argument(
        "--robust-base-algorithm",
        choices=("kaselouris", "kaselouris_robust", "heliyon", "heliyon_robust", "vrupod"),
        default="kaselouris",
        help="base router wrapped by robust_consensus",
    )

    parser.add_argument(
        "--robust-scenarios",
        type=int,
        default=1,
        help="number of dataset-derived imputation scenarios in addition to the observed graph",
    )

    parser.add_argument(
        "--city",
        default="all",
        help="city_id or all",
    )

    parser.add_argument(
        "--area",
        default="all",
        help="area_id or all",
    )

    parser.add_argument(
        "--paths",
        type=int,
        default=5,
        help="number of routes per city",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
        help=(
            "worker processes for route evaluations; "
            "default is min(4, CPU count); "
            "set to 1 to disable parallelism"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    parser.add_argument(
        "--minimum-distance-m",
        type=float,
        default=500.0,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "benchmark_kaselouris_reports"
        ),
    )

    parser.add_argument(
        "--vrupod-quality-bias",
        type=float,
        default=1.0,
        help=(
            "quality-vs-latency tradeoff for VRUPOD routing; "
            "values > 1.0 bias toward more quality-sensitive routes"
        ),
    )

    parser.add_argument(
        "--vrupod-n-blocked",
        type=int,
        default=0,
        help=(
            "number of blocked edges used during VRUPOD evaluation; "
            "0 keeps the assessment focused on route quality "
            "rather than forced failures"
        ),
    )

    parser.add_argument(
        "--uncertainty-weight",
        type=float,
        default=1.5,
        help=(
            "uncertainty penalty weight for robust algorithms; "
            "higher values prefer safer but longer routes "
            "under low observability"
        ),
    )

    parser.add_argument(
        "--test-cities",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--exclude-cities",
        nargs="+",
        default=[],
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate workers
    # ------------------------------------------------------------------

    if args.workers < 1:
        parser.error(
            "--workers must be >= 1"
        )

    if args.paths < 1:
        parser.error(
            "--paths must be >= 1"
        )

    if args.robust_scenarios < 0:
        parser.error("--robust-scenarios must be >= 0")

    # ------------------------------------------------------------------
    # Resolve cities
    # ------------------------------------------------------------------

    if args.city == "all":
        from pymongo import MongoClient
        from dotenv import load_dotenv

        load_dotenv(".env")

        client = MongoClient(
            os.environ["MONGODB_URI"],
            serverSelectionTimeoutMS=30000,
        )

        try:
            database_name = os.environ.get(
                "MONGODB_DATABASE",
                "accessible_routing",
            )

            city_ids = sorted(
                client[database_name]
                .cities
                .distinct("city_id")
            )

        finally:
            client.close()

        if args.test_cities is not None:
            city_ids = [
                city_id
                for city_id in city_ids
                if city_id in args.test_cities
            ]

        city_ids = [
            city_id
            for city_id in city_ids
            if city_id not in args.exclude_cities
        ]

    else:
        city_ids = [args.city]

    if not city_ids:
        print(
            "No cities selected.",
            flush=True,
        )
        return

    print(
        f"Cities to benchmark: {len(city_ids)}",
        flush=True,
    )

    print(
        f"Algorithm: {args.algorithm}",
        flush=True,
    )

    print(
        f"Workers: {args.workers}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Run benchmark
    # ------------------------------------------------------------------

    rows: List[Dict[str, Any]] = []

    benchmark_started = time.perf_counter()

    for index, city_id in enumerate(city_ids):
        print(
            f"\nOverall city progress: "
            f"{index + 1}/{len(city_ids)}",
            flush=True,
        )

        city_rows = run_city_kaselouris(
            city_id,
            args.area,
            args.paths,
            args.seed + index,
            args.minimum_distance_m,
            algorithm=args.algorithm,
            quality_bias=(
                args.vrupod_quality_bias
                if args.algorithm == "vrupod"
                or (args.algorithm == "robust_consensus" and args.robust_base_algorithm == "vrupod")
                else 1.0
            ),
            n_blocked=(
                args.vrupod_n_blocked
                if args.algorithm == "vrupod"
                or (args.algorithm == "robust_consensus" and args.robust_base_algorithm == "vrupod")
                else 0
            ),
            uncertainty_weight=(
                args.uncertainty_weight
                if args.algorithm in {"kaselouris_robust", "heliyon_robust"}
                or (args.algorithm == "robust_consensus" and args.robust_base_algorithm in {"kaselouris_robust", "heliyon_robust"})
                else 1.0
            ),
            workers=args.workers,
            robust_base_algorithm=args.robust_base_algorithm,
            robust_scenarios=args.robust_scenarios,
        )

        rows.extend(city_rows)

        print(
            f"Completed city {city_id}: "
            f"{len(city_rows):,} rows",
            flush=True,
        )

    total_elapsed = (
        time.perf_counter()
        - benchmark_started
    )

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    print(
        f"\nAll cities completed in "
        f"{total_elapsed:.2f}s.",
        flush=True,
    )

    print(
        f"Writing reports to: "
        f"{args.output_dir.resolve()}",
        flush=True,
    )

    write_reports(
        rows,
        args.output_dir,
    )

    print(
        f"Wrote {len(rows):,} rows to "
        f"{args.output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
