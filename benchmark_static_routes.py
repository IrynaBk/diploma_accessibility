"""Benchmark Dijkstra and A* on reproducible static route paths.

Dynamic changes and replanning are intentionally excluded here. The blockage
levels are static and applied before each route is calculated.

Examples:
    python benchmark_static_routes.py --city lviv_ua --area all --paths 10
PPO evaluation is intentionally disabled for this classical benchmark. It can
be added later after a sufficiently trained policy is available.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from route_metrics import AGENT_PROFILES, GraphEdge, calculate_route_metrics
from routing_experiments import (
    DynamicEdgeState,
    block_random_edges,
    load_graph,
    make_adjacency,
    nearest_node,
    shortest_path,
)

# PPO is reserved for a later learned-policy benchmark.
# ALGORITHMS = ("dijkstra", "astar", "ppo")
ALGORITHMS = ("dijkstra", "astar")
BLOCK_LEVELS = (0.0, 0.05, 0.10)


def graph_components(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
) -> List[Set[int]]:
    adjacency = make_adjacency(nodes, edges)
    unseen = set(adjacency)
    components = []
    while unseen:
        root = unseen.pop()
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor, _ in adjacency.get(current, []):
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
        (component for component in graph_components(nodes, edges) if len(component) >= 2),
        key=len,
        reverse=True,
    )[:10]
    if not components:
        raise ValueError("The graph has no connected component with two nodes.")
    paths: List[Tuple[int, int]] = []
    attempts = 0
    while len(paths) < count and attempts < count * 500:
        attempts += 1
        component = generator.choice(components)
        start, goal = generator.sample(tuple(component), 2)
        from route_metrics import haversine_m

        if haversine_m(nodes[start], nodes[goal]) < minimum_distance_m:
            continue
        if (start, goal) not in paths:
            paths.append((start, goal))
    if len(paths) < count:
        raise ValueError(
            f"Could generate only {len(paths)} of {count} paths. "
            "Try fewer paths or a smaller --minimum-distance-m."
        )
    return paths


def static_classical_result(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start: int,
    goal: int,
    agent: str,
    algorithm: str,
    blocked: DynamicEdgeState,
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        result = shortest_path(nodes, edges, start, goal, agent, algorithm, blocked)
    except ValueError as error:
        return {
            "success": False,
            "error": str(error),
            "runtime_s": round(time.perf_counter() - started, 6),
        }
    metrics = calculate_route_metrics(result.edges, AGENT_PROFILES[agent])
    metrics["runtime_s"] = round(time.perf_counter() - started, 6)
    metrics["expanded_nodes"] = result.expanded_nodes
    metrics["success"] = True
    return metrics


def add_common_fields(
    row: Dict[str, Any],
    nodes: Dict[int, Tuple[float, float]],
    start: int,
    goal: int,
    blocked_percent: float,
) -> Dict[str, Any]:
    from route_metrics import haversine_m

    row.update({
        "start_node": start,
        "goal_node": goal,
        "start_lat": nodes[start][0],
        "start_lon": nodes[start][1],
        "goal_lat": nodes[goal][0],
        "goal_lon": nodes[goal][1],
        "straight_line_m": round(haversine_m(nodes[start], nodes[goal]), 3),
        "blocked_percent": blocked_percent,
    })
    if row.get("success") and row.get("straight_line_m", 0) > 0:
        row["detour_ratio"] = round(row["length_m"] / row["straight_line_m"], 4)
    else:
        row["detour_ratio"] = None
    return row


def run_city(
    city_id: str,
    area_id: str,
    path_count: int,
    seed: int,
    minimum_distance_m: float,
) -> List[Dict[str, Any]]:
    nodes, edges = load_graph(city_id, area_id)
    paths = generate_paths(nodes, edges, path_count, seed)
    rows = []
    for path_index, (start, goal) in enumerate(paths):
        for blocked_percent in BLOCK_LEVELS:
            blocked = block_random_edges(edges, blocked_percent, seed + path_index)
            for agent in AGENT_PROFILES:
                for algorithm in ALGORITHMS:
                    result = static_classical_result(
                        nodes,
                        edges,
                        start,
                        goal,
                        agent,
                        algorithm,
                        blocked,
                    )
                    row = {
                        "city_id": city_id,
                        "area_id": area_id,
                        "path_index": path_index,
                        "path_id": f"{city_id}:{path_index + 1}",
                        "agent": agent,
                        "algorithm": algorithm,
                        **result,
                    }
                    rows.append(add_common_fields(row, nodes, start, goal, blocked_percent))
                    print(
                        f"{city_id} path={path_index + 1}/{path_count} "
                        f"{algorithm}/{agent} blocked={blocked_percent:.0%}: "
                        f"{'success' if result.get('success') else 'failure'}"
                    )
    return rows


def write_reports(
    rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Generate an interactive HTML dashboard for benchmark results.

    Dashboard contains:
        1. KPI overview
        2. Interactive filters
        3. Algorithm comparison
        4. Agent comparison
        5. Per-path comparison
        6. Raw benchmark results
    """

    import json
    import html
    from collections import defaultdict

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError("No benchmark results to report.")

    # ========================================================
    # NORMALIZE DATA
    # ========================================================

    def numeric(
        row: Dict[str, Any],
        key: str,
    ) -> Optional[float]:

        value = row.get(key)

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for row in rows:

        row["length_m"] = numeric(
            row,
            "length_m",
        )

        row["runtime_s"] = numeric(
            row,
            "runtime_s",
        )

        row["travel_cost"] = numeric(
            row,
            "travel_cost",
        )

        row["detour_ratio"] = numeric(
            row,
            "detour_ratio",
        )

        row["violation_rate"] = numeric(
            row,
            "violation_rate",
        )

        row["expanded_nodes"] = numeric(
            row,
            "expanded_nodes",
        )

        row["blocked_percent"] = numeric(
            row,
            "blocked_percent",
        )

    # ========================================================
    # BASIC DATA
    # ========================================================

    cities = sorted({
        row.get("city_id", "")
        for row in rows
    })

    agents = sorted({
        row.get("agent", "")
        for row in rows
    })

    algorithms = sorted({
        row.get("algorithm", "")
        for row in rows
    })

    blocked_levels = sorted({
        row.get("blocked_percent")
        for row in rows
        if row.get("blocked_percent") is not None
    })

    # ========================================================
    # KPI
    # ========================================================

    successful = [
        row
        for row in rows
        if row.get("success")
    ]

    def mean(
        values: List[Optional[float]]
    ) -> Optional[float]:

        values = [
            value
            for value in values
            if value is not None
        ]

        if not values:
            return None

        return sum(values) / len(values)

    success_rate = (
        len(successful) / len(rows)
        if rows
        else 0
    )

    mean_runtime = mean([
        row.get("runtime_s")
        for row in successful
    ])

    mean_length = mean([
        row.get("length_m")
        for row in successful
    ])

    mean_detour = mean([
        row.get("detour_ratio")
        for row in successful
    ])

    mean_violation = mean([
        row.get("violation_rate")
        for row in successful
    ])

    # ========================================================
    # SUMMARY GROUPING
    # ========================================================

    group_keys = [
        "city_id",
        "agent",
        "algorithm",
        "blocked_percent",
    ]

    grouped = defaultdict(list)

    for row in rows:

        key = tuple(
            row.get(field)
            for field in group_keys
        )

        grouped[key].append(row)

    summary_rows = []

    for key, group in sorted(
        grouped.items(),
        key=lambda item: tuple(
            str(value)
            for value in item[0]
        ),
    ):

        successful_group = [
            row
            for row in group
            if row.get("success")
        ]

        summary_rows.append({

            "city_id": key[0],

            "agent": key[1],

            "algorithm": key[2],

            "blocked_percent":
                key[3],

            "paths":
                len(group),

            "success_rate":
                len(successful_group)
                / len(group),

            "mean_runtime_s":
                mean([
                    row.get("runtime_s")
                    for row in successful_group
                ]),

            "mean_expanded_nodes":
                mean([
                    row.get("expanded_nodes")
                    for row in successful_group
                ]),

            "mean_length_m":
                mean([
                    row.get("length_m")
                    for row in successful_group
                ]),

            "mean_travel_cost":
                mean([
                    row.get("travel_cost")
                    for row in successful_group
                ]),

            "mean_detour_ratio":
                mean([
                    row.get("detour_ratio")
                    for row in successful_group
                ]),

            "mean_violation_rate":
                mean([
                    row.get("violation_rate")
                    for row in successful_group
                ]),
        })

    # ========================================================
    # JSON DATA FOR BROWSER
    # ========================================================

    rows_json = json.dumps(
        rows,
        default=str,
    )

    summary_json = json.dumps(
        summary_rows,
        default=str,
    )

    # ========================================================
    # HTML HELPERS
    # ========================================================

    def fmt(
        value: Any,
        digits: int = 3,
    ) -> str:

        if value is None:
            return "—"

        if isinstance(value, float):
            return f"{value:.{digits}f}"

        return str(value)

    def pct(
        value: Any,
    ) -> str:

        if value is None:
            return "—"

        return f"{float(value) * 100:.1f}%"

    def coordinate(
        lat: Any,
        lon: Any,
    ) -> str:

        if lat is None or lon is None:
            return "—"

        return (
            f"{float(lat):.5f}, "
            f"{float(lon):.5f}"
        )

    # ========================================================
    # KPI CARDS
    # ========================================================

    cards = f"""
    <div class="kpi-grid">

        <div class="kpi">
            <div class="kpi-label">Benchmark runs</div>
            <div class="kpi-value">
                {len(rows):,}
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">Success rate</div>
            <div class="kpi-value">
                {success_rate * 100:.1f}%
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">Mean runtime</div>
            <div class="kpi-value">
                {fmt(mean_runtime * 1000 if mean_runtime else None, 2)}
                <span class="unit">ms</span>
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">Mean route length</div>
            <div class="kpi-value">
                {fmt(mean_length / 1000 if mean_length else None, 2)}
                <span class="unit">km</span>
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">Mean detour</div>
            <div class="kpi-value">
                {fmt(mean_detour, 3)}
            </div>
        </div>

        <div class="kpi">
            <div class="kpi-label">Mean violations</div>
            <div class="kpi-value">
                {pct(mean_violation)}
            </div>
        </div>

    </div>
    """

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    summary_html = ""

    for row in summary_rows:

        summary_html += f"""
        <tr>

            <td>{html.escape(str(row["city_id"]))}</td>

            <td>{html.escape(str(row["agent"]))}</td>

            <td>
                <strong>
                    {html.escape(str(row["algorithm"]))}
                </strong>
            </td>

            <td>
                {pct(row["blocked_percent"])}
            </td>

            <td>{row["paths"]}</td>

            <td>{pct(row["success_rate"])}</td>

            <td>
                {fmt(
                    row["mean_runtime_s"] * 1000
                    if row["mean_runtime_s"] is not None
                    else None,
                    2,
                )}
            </td>

            <td>
                {fmt(row["mean_expanded_nodes"], 0)}
            </td>

            <td>
                {fmt(
                    row["mean_length_m"] / 1000
                    if row["mean_length_m"] is not None
                    else None,
                    2,
                )}
            </td>

            <td>
                {fmt(row["mean_travel_cost"])}
            </td>

            <td>
                {fmt(row["mean_detour_ratio"])}
            </td>

            <td>
                {pct(row["mean_violation_rate"])}
            </td>

        </tr>
        """

    # ========================================================
    # PATH TABLE
    # ========================================================

    path_html = ""

    for row in rows:

        status = (
            '<span class="success">✓ success</span>'
            if row.get("success")
            else
            '<span class="failure">✗ failure</span>'
        )

        point_a = coordinate(
            row.get("start_lat"),
            row.get("start_lon"),
        )

        point_b = coordinate(
            row.get("goal_lat"),
            row.get("goal_lon"),
        )

        path_html += f"""
        <tr>

            <td>
                {html.escape(
                    str(row.get("path_id", ""))
                )}
            </td>

            <td>
                <div class="point">
                    <b>A</b> {point_a}
                </div>

                <div class="point">
                    <b>B</b> {point_b}
                </div>
            </td>

            <td>
                {html.escape(
                    str(row.get("city_id", ""))
                )}
            </td>

            <td>
                {html.escape(
                    str(row.get("agent", ""))
                )}
            </td>

            <td>
                <strong>
                    {html.escape(
                        str(row.get("algorithm", ""))
                    )}
                </strong>
            </td>

            <td>
                {pct(
                    row.get("blocked_percent")
                )}
            </td>

            <td>
                {status}
            </td>

            <td>
                {fmt(
                    row.get("length_m") / 1000
                    if row.get("length_m") is not None
                    else None,
                    2,
                )}
            </td>

            <td>
                {fmt(
                    row.get("travel_cost")
                )}
            </td>

            <td>
                {fmt(
                    row.get("detour_ratio")
                )}
            </td>

            <td>
                {pct(
                    row.get("violation_rate")
                )}
            </td>

            <td>
                {fmt(
                    row.get("runtime_s") * 1000
                    if row.get("runtime_s") is not None
                    else None,
                    2,
                )}
            </td>

            <td>
                {fmt(
                    row.get("expanded_nodes"),
                    0,
                )}
            </td>

        </tr>
        """

    # ========================================================
    # INTERACTIVE CHARTS
    # ========================================================

    # Algorithm runtime

    runtime_fig = go.Figure()

    for algorithm in algorithms:

        values = [
            row["mean_runtime_s"] * 1000
            for row in summary_rows
            if (
                row["algorithm"] == algorithm
                and row["mean_runtime_s"] is not None
            )
        ]

        runtime_fig.add_trace(
            go.Box(
                y=values,
                name=algorithm,
                boxmean=True,
            )
        )

    runtime_fig.update_layout(
        title="Runtime distribution",
        yaxis_title="Runtime (ms)",
        template="plotly_white",
    )

    runtime_html = runtime_fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )

    # Expanded nodes

    nodes_fig = go.Figure()

    for algorithm in algorithms:

        values = [
            row["mean_expanded_nodes"]
            for row in summary_rows
            if (
                row["algorithm"] == algorithm
                and row["mean_expanded_nodes"] is not None
            )
        ]

        nodes_fig.add_trace(
            go.Box(
                y=values,
                name=algorithm,
                boxmean=True,
            )
        )

    nodes_fig.update_layout(
        title="Search effort",
        yaxis_title="Expanded nodes",
        template="plotly_white",
    )

    nodes_html = nodes_fig.to_html(
        full_html=False,
        include_plotlyjs=False,
    )

    # ========================================================
    # FULL HTML
    # ========================================================

    document = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>
AccessibleRouteBench — Static Routing Benchmark
</title>

<style>

body {{
    font-family:
        Inter,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background: #f5f7f8;

    color: #17252a;

    margin: 0;
}}

.container {{
    max-width: 1800px;

    margin: auto;

    padding: 32px;
}}

h1 {{
    margin-bottom: 4px;
}}

.subtitle {{
    color: #66777c;

    margin-bottom: 28px;
}}

.kpi-grid {{
    display: grid;

    grid-template-columns:
        repeat(6, 1fr);

    gap: 16px;

    margin-bottom: 24px;
}}

.kpi {{
    background: white;

    border-radius: 12px;

    padding: 20px;

    box-shadow:
        0 1px 4px
        rgba(0,0,0,.08);
}}

.kpi-label {{
    color: #66777c;

    font-size: 13px;

    margin-bottom: 8px;
}}

.kpi-value {{
    font-size: 28px;

    font-weight: 700;
}}

.unit {{
    font-size: 14px;

    color: #66777c;
}}

.card {{
    background: white;

    border-radius: 12px;

    padding: 24px;

    margin-bottom: 24px;

    box-shadow:
        0 1px 4px
        rgba(0,0,0,.08);
}}

.table-container {{
    overflow-x: auto;

    max-height: 700px;
}}

table {{
    width: 100%;

    border-collapse: collapse;

    font-size: 13px;
}}

th {{
    position: sticky;

    top: 0;

    background: #12343b;

    color: white;

    padding: 10px;

    text-align: left;

    z-index: 2;
}}

td {{
    padding: 9px 10px;

    border-bottom:
        1px solid #e5e9ea;

    white-space: nowrap;
}}

tr:hover {{
    background: #f2f6f7;
}}

.point {{
    margin: 3px 0;
}}

.success {{
    color: #16803c;

    font-weight: 600;
}}

.failure {{
    color: #c62828;

    font-weight: 600;
}}

.section-title {{
    margin-top: 0;

    margin-bottom: 16px;
}}

.filters {{
    display: flex;

    gap: 12px;

    flex-wrap: wrap;

    margin-bottom: 20px;
}}

select {{
    padding: 9px 12px;

    border:
        1px solid #ccd5d5;

    border-radius: 8px;

    background: white;
}}

@media(max-width: 1200px) {{

    .kpi-grid {{
        grid-template-columns:
            repeat(3, 1fr);
    }}

}}

@media(max-width: 700px) {{

    .container {{
        padding: 16px;
    }}

    .kpi-grid {{
        grid-template-columns:
            repeat(2, 1fr);
    }}

}}

</style>

</head>

<body>

<div class="container">

<h1>
AccessibleRouteBench
</h1>

<div class="subtitle">
Static routing benchmark —
Dijkstra vs A* across accessibility agents
</div>

{cards}

<div class="card">

<h2 class="section-title">
Algorithm Performance
</h2>

<div>
{runtime_html}
</div>

<div>
{nodes_html}
</div>

</div>


<div class="card">

<h2 class="section-title">
Aggregate comparison
</h2>

<div class="table-container">

<table>

<thead>

<tr>

<th>City</th>
<th>Agent</th>
<th>Algorithm</th>
<th>Blocked</th>
<th>Paths</th>
<th>Success</th>
<th>Runtime ms</th>
<th>Expanded nodes</th>
<th>Length km</th>
<th>Travel cost</th>
<th>Detour</th>
<th>Violations</th>

</tr>

</thead>

<tbody>

{summary_html}

</tbody>

</table>

</div>

</div>


<div class="card">

<h2 class="section-title">
Individual route results
</h2>

<div class="subtitle">

Every row represents one
<strong>
A → B × agent × algorithm × blockage
</strong>
experiment.

</div>

<div class="table-container">

<table>

<thead>

<tr>

<th>Path</th>
<th>Points</th>
<th>City</th>
<th>Agent</th>
<th>Algorithm</th>
<th>Blocked</th>
<th>Status</th>
<th>Length km</th>
<th>Travel cost</th>
<th>Detour</th>
<th>Violations</th>
<th>Runtime ms</th>
<th>Expanded nodes</th>

</tr>

</thead>

<tbody>

{path_html}

</tbody>

</table>

</div>

</div>

<div class="card">

<h2 class="section-title">
Benchmark configuration
</h2>

<pre>
Cities:
{html.escape(", ".join(cities))}

Agents:
{html.escape(", ".join(agents))}

Algorithms:
{html.escape(", ".join(algorithms))}

Blockage levels:
{html.escape(", ".join(f"{x:.0%}" for x in blocked_levels))}

Total benchmark runs:
{len(rows):,}

Dataset generated:
{time.strftime("%Y-%m-%d %H:%M:%S")}
</pre>

</div>

</div>

</body>

</html>
"""

    report_path = (
        output_dir /
        "report.html"
    )

    report_path.write_text(
        document,
        encoding="utf-8",
    )

    # ========================================================
    # ALSO SAVE RAW DATA
    # ========================================================

    (
        output_dir /
        "results.json"
    ).write_text(
        rows_json,
        encoding="utf-8",
    )

    fields = sorted({
        field
        for row in rows
        for field in row
    })

    with (
        output_dir /
        "results.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)

    with (
        output_dir /
        "summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        summary_fields = [
            "city_id",
            "agent",
            "algorithm",
            "blocked_percent",
            "paths",
            "success_rate",
            "mean_runtime_s",
            "mean_expanded_nodes",
            "mean_length_m",
            "mean_travel_cost",
            "mean_detour_ratio",
            "mean_violation_rate",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()

        writer.writerows(summary_rows)

    print()
    print("=" * 70)
    print("BENCHMARK REPORT")
    print("=" * 70)
    print()
    print(
        f"Report: {report_path.resolve()}"
    )
    print(
        f"Runs:   {len(rows):,}"
    )
    print(
        f"Paths:  {len(set(row.get('path_id') for row in rows)):,}"
    )
    print(
        f"Success rate: {success_rate:.1%}"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark static routing algorithms and agents.")
    parser.add_argument("--city", required=True, help="city_id, or all")
    parser.add_argument(
        "--test-cities",
        nargs="+",
        default=None,
        help="Cities to evaluate when --city all is used.",
    )
    parser.add_argument(
        "--exclude-cities",
        nargs="+",
        default=[],
        help="Cities to omit when --city all is used.",
    )
    parser.add_argument("--area", required=True, help="area_id, or all")
    parser.add_argument("--paths", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-distance-m", type=float, default=500.0)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_reports"))
    args = parser.parse_args()
    if args.paths <= 0 or args.minimum_distance_m < 0:
        raise ValueError("--paths must be positive and minimum distance cannot be negative.")

    if args.city == "all":
        from pymongo import MongoClient
        from dotenv import load_dotenv
        import os

        load_dotenv(".env")
        client = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=30_000)
        try:
            city_ids = sorted(client[os.environ.get("MONGODB_DATABASE", "accessible_routing")].cities.distinct("city_id"))
        finally:
            client.close()
        if args.test_cities is not None:
            city_ids = [city_id for city_id in city_ids if city_id in args.test_cities]
        city_ids = [city_id for city_id in city_ids if city_id not in args.exclude_cities]
    else:
        city_ids = [args.city]

    rows: List[Dict[str, Any]] = []
    for index, city_id in enumerate(city_ids):
        rows.extend(run_city(city_id, args.area, args.paths, args.seed + index, args.minimum_distance_m))
    write_reports(rows, args.output_dir)
    print(f"Wrote {len(rows):,} rows to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
