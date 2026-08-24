from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_rows_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load benchmark rows and restore numeric and boolean CSV values."""

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))

    for row in rows:
        for field, value in list(row.items()):
            if value == "" or value == "—":
                row[field] = None
                continue
            if value.lower() in {"true", "false"}:
                row[field] = value.lower() == "true"
                continue
            try:
                row[field] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                pass
    return rows


def add_aggregation_dashboard(
    rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Append an aggregation dashboard to the existing report.html.

    Aggregations:
        1. By agent
        2. By city
        3. By area type
        4. By blockage level

    Algorithm is retained as a comparison dimension in every aggregation.
    """

    from collections import defaultdict

    report_path = output_dir / "report.html"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Existing report not found: {report_path}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    METRICS = [
        ("success_rate", "Success Rate", "%"),
        ("runtime_s", "Runtime", "s"),
        ("expanded_nodes", "Expanded Nodes", ""),
        ("length_m", "Route Length", "m"),
        ("travel_cost", "Travel Cost", ""),
        ("detour_ratio", "Detour Ratio", "x"),
        ("violation_rate", "Violation Rate", "%"),
    ]

    def mean(
        group: List[Dict[str, Any]],
        field: str,
    ) -> Optional[float]:
        values = []

        for row in group:
            value = row.get(field)

            if value is None:
                continue

            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue

        if not values:
            return None

        return sum(values) / len(values)

    def aggregate(
        rows: List[Dict[str, Any]],
        dimensions: Sequence[str],
    ) -> List[Dict[str, Any]]:
        groups = defaultdict(list)

        for row in rows:
            key = tuple(row.get(d) for d in dimensions)
            groups[key].append(row)

        results = []

        for key, group in groups.items():
            successful = [
                row for row in group
                if row.get("success") is True
            ]

            result = {
                dimension: value
                for dimension, value in zip(dimensions, key)
            }

            result.update({
                "paths": len(group),
                "success_rate": (
                    len(successful) / len(group)
                    if group
                    else None
                ),
                "runtime_s": mean(group, "runtime_s"),
                "expanded_nodes": mean(group, "expanded_nodes"),
                "length_m": mean(successful, "length_m"),
                "travel_cost": mean(successful, "travel_cost"),
                "detour_ratio": mean(successful, "detour_ratio"),
                "violation_rate": mean(successful, "violation_rate"),
            })

            results.append(result)

        return results

    def fmt(
        value: Any,
        field: str,
    ) -> str:
        if value is None:
            return "—"

        if field == "success_rate":
            return f"{float(value) * 100:.1f}%"

        if field == "violation_rate":
            return f"{float(value) * 100:.1f}%"

        if field == "runtime_s":
            return f"{float(value):.4f}"

        if field == "expanded_nodes":
            return f"{float(value):,.1f}"

        if field == "length_m":
            return f"{float(value):,.1f}"

        if field == "detour_ratio":
            return f"{float(value):.3f}"

        if field == "travel_cost":
            return f"{float(value):,.3f}"

        return str(value)

    def dimension_title(dimension: str) -> str:
        return {
            "agent": "Agent Type",
            "city_id": "City",
            "area_type": "Area Type",
            "blocked_percent": "Blockage Level",
        }.get(dimension, dimension)

    # ------------------------------------------------------------------
    # Aggregation configurations
    #
    # Algorithm remains a dimension in all aggregations.
    # ------------------------------------------------------------------

    aggregation_configs = [
        (
            "agent",
            ["agent", "algorithm"],
            "By Agent Type",
        ),
        (
            "city",
            ["city_id", "algorithm"],
            "By City",
        ),
        (
            "area_type",
            ["area_type", "algorithm"],
            "By Area Type",
        ),
        (
            "blockage",
            ["blocked_percent", "algorithm"],
            "By Blockage Level",
        ),
    ]

    # ------------------------------------------------------------------
    # Build HTML
    # ------------------------------------------------------------------

    sections = []

    sections.append("""
    <section id="aggregation-dashboard">

    <h1>Aggregation Dashboard</h1>

    <p class="dashboard-description">
        Aggregated comparison of routing performance across agent types,
        cities, area types, and network blockage levels.
        Dijkstra and A* are kept as separate algorithm groups.
    </p>

    <div class="metric-legend">
        <strong>Metrics:</strong>
        Success Rate · Runtime · Expanded Nodes · Route Length ·
        Travel Cost · Detour Ratio · Violation Rate
    </div>
    """)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    for section_id, dimensions, title in aggregation_configs:

        aggregated = aggregate(rows, dimensions)

        # Sort nicely
        aggregated.sort(
            key=lambda row: tuple(
                str(row.get(d))
                for d in dimensions
            )
        )

        headers = (
            [dimension_title(d) for d in dimensions]
            + [
                "Paths",
                "Success Rate",
                "Runtime (s)",
                "Expanded Nodes",
                "Route Length (m)",
                "Travel Cost",
                "Detour Ratio",
                "Violation Rate",
            ]
        )

        table_rows = []

        for row in aggregated:
            cells = []

            for dimension in dimensions:
                value = row.get(dimension)

                if dimension == "blocked_percent":
                    value = (
                        f"{float(value) * 100:.0f}%"
                        if value is not None
                        else "—"
                    )

                cells.append(
                    f"<td>{html.escape(str(value))}</td>"
                )

            cells.extend([
                f"<td>{row['paths']}</td>",
                f"<td>{fmt(row['success_rate'], 'success_rate')}</td>",
                f"<td>{fmt(row['runtime_s'], 'runtime_s')}</td>",
                f"<td>{fmt(row['expanded_nodes'], 'expanded_nodes')}</td>",
                f"<td>{fmt(row['length_m'], 'length_m')}</td>",
                f"<td>{fmt(row['travel_cost'], 'travel_cost')}</td>",
                f"<td>{fmt(row['detour_ratio'], 'detour_ratio')}</td>",
                f"<td>{fmt(row['violation_rate'], 'violation_rate')}</td>",
            ])

            table_rows.append(
                "<tr>" + "".join(cells) + "</tr>"
            )

        header_html = "".join(
            f"<th>{html.escape(header)}</th>"
            for header in headers
        )

        section_html = f"""
        <div class="aggregation-section">

            <h2>{html.escape(title)}</h2>

            <div class="table-wrapper">

                <table class="aggregation-table">

                    <thead>
                        <tr>
                            {header_html}
                        </tr>
                    </thead>

                    <tbody>
                        {''.join(table_rows)}
                    </tbody>

                </table>

            </div>

        </div>
        """

        sections.append(section_html)

    # ------------------------------------------------------------------
    # Interactive path-level explorer
    # ------------------------------------------------------------------

    # Serialize data safely for JavaScript.
    js_rows = json.dumps(
        rows,
        ensure_ascii=False,
        default=str,
    )

    sections.append(f"""
    <div class="aggregation-section">

        <h2>Path-Level Explorer</h2>

        <p>
            Select a city, area, agent, algorithm and blockage level
            to inspect individual A → B experiments.
        </p>

        <div class="filters">

            <label>
                City
                <select id="filter-city">
                    <option value="">All</option>
                </select>
            </label>

            <label>
                Area
                <select id="filter-area">
                    <option value="">All</option>
                </select>
            </label>

            <label>
                Agent
                <select id="filter-agent">
                    <option value="">All</option>
                </select>
            </label>

            <label>
                Algorithm
                <select id="filter-algorithm">
                    <option value="">All</option>
                </select>
            </label>

            <label>
                Blockage
                <select id="filter-blockage">
                    <option value="">All</option>
                </select>
            </label>

        </div>

        <div id="filtered-results"></div>

    </div>

    <script>

    const benchmarkRows = {js_rows};

    const filterDefinitions = {{
        "filter-city": "city_id",
        "filter-area": "area_id",
        "filter-agent": "agent",
        "filter-algorithm": "algorithm",
        "filter-blockage": "blocked_percent"
    }};

    function uniqueValues(field) {{

        return [...new Set(
            benchmarkRows
                .map(row => row[field])
                .filter(value => value !== null && value !== undefined)
        )].sort();

    }}

    Object.entries(filterDefinitions).forEach(
        ([elementId, field]) => {{

            const select = document.getElementById(elementId);

            uniqueValues(field).forEach(value => {{

                const option = document.createElement("option");

                option.value = value;
                option.textContent =
                    field === "blocked_percent"
                        ? `${{Number(value) * 100}}%`
                        : value;

                select.appendChild(option);

            }});

            select.addEventListener(
                "change",
                renderFilteredResults
            );

        }}
    );

    function renderFilteredResults() {{

        let filtered = benchmarkRows.filter(row => {{

            return Object.entries(filterDefinitions).every(
                ([elementId, field]) => {{

                    const selected =
                        document.getElementById(elementId).value;

                    return (
                        selected === "" ||
                        String(row[field]) === selected
                    );

                }}
            );

        }});

        const columns = [
            ["path_id", "Path"],
            ["start_lat", "A Lat"],
            ["start_lon", "A Lon"],
            ["goal_lat", "B Lat"],
            ["goal_lon", "B Lon"],
            ["agent", "Agent"],
            ["algorithm", "Algorithm"],
            ["blocked_percent", "Blocked"],
            ["success", "Success"],
            ["runtime_s", "Runtime"],
            ["expanded_nodes", "Expanded"],
            ["length_m", "Length"],
            ["travel_cost", "Travel Cost"],
            ["detour_ratio", "Detour"],
            ["violation_rate", "Violations"]
        ];

        let html = `
            <div class="table-wrapper">
            <table class="aggregation-table">
            <thead>
            <tr>
        `;

        columns.forEach(
            ([field, title]) => {{
                html += `<th>${{title}}</th>`;
            }}
        );

        html += `
            </tr>
            </thead>
            <tbody>
        `;

        filtered.forEach(row => {{

            html += "<tr>";

            columns.forEach(([field]) => {{

                let value = row[field];

                if (field === "blocked_percent") {{
                    value = value === null || value === undefined
                        ? "—"
                        : `${{Number(value) * 100}}%`;
                }}

                if (field === "success") {{
                    value = value ? "✓" : "✗";
                }}

                if (
                    typeof value === "number" &&
                    field !== "blocked_percent"
                ) {{
                    value = Number(value).toFixed(3);
                }}

                if (value === null || value === undefined) {{
                    value = "—";
                }}

                html += `<td>${{value}}</td>`;

            }});

            html += "</tr>";

        }});

        html += `
            </tbody>
            </table>
            </div>
        `;

        html += `
            <p class="result-count">
                Showing <strong>${{filtered.length}}</strong>
                experiment rows.
            </p>
        `;

        document.getElementById(
            "filtered-results"
        ).innerHTML = html;

    }}

    renderFilteredResults();

    </script>
    """)

    sections.append("""
    </section>
    """)

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------

    css = """
    <style>

    #aggregation-dashboard {
        margin-top: 60px;
        padding-top: 30px;
        border-top: 4px solid #12343b;
    }

    #aggregation-dashboard h1 {
        font-size: 30px;
        margin-bottom: 8px;
    }

    #aggregation-dashboard h2 {
        margin-top: 42px;
        font-size: 22px;
    }

    .dashboard-description {
        color: #555;
        max-width: 900px;
        line-height: 1.6;
    }

    .metric-legend {
        margin: 20px 0;
        padding: 14px 18px;
        background: #f2f6f7;
        border-left: 4px solid #176b87;
        border-radius: 4px;
    }

    .aggregation-section {
        margin-top: 35px;
    }

    .table-wrapper {
        overflow-x: auto;
        max-width: 100%;
        border: 1px solid #d8dddd;
        border-radius: 6px;
    }

    .aggregation-table {
        border-collapse: collapse;
        width: 100%;
        min-width: 1000px;
        font-size: 13px;
    }

    .aggregation-table th {
        background: #12343b;
        color: white;
        padding: 9px 10px;
        text-align: left;
        position: sticky;
        top: 0;
        white-space: nowrap;
    }

    .aggregation-table td {
        border-bottom: 1px solid #e1e5e5;
        padding: 8px 10px;
        white-space: nowrap;
    }

    .aggregation-table tr:hover {
        background: #f5f8f8;
    }

    .filters {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin: 20px 0;
        padding: 18px;
        background: #f5f7f7;
        border-radius: 8px;
    }

    .filters label {
        display: flex;
        flex-direction: column;
        gap: 5px;
        font-weight: 600;
        font-size: 13px;
    }

    .filters select {
        min-width: 150px;
        padding: 7px 9px;
        border: 1px solid #c7cece;
        border-radius: 5px;
        background: white;
    }

    .result-count {
        color: #666;
        margin-top: 10px;
    }

    </style>
    """

    # ------------------------------------------------------------------
    # Append dashboard to existing HTML
    # ------------------------------------------------------------------

    existing_html = report_path.read_text(
        encoding="utf-8"
    )

    dashboard_html = (
        css
        + "\n"
        + "\n".join(sections)
    )

    if "</body>" in existing_html:

        existing_html = existing_html.replace(
            "</body>",
            dashboard_html + "\n</body>",
        )

    else:

        existing_html += dashboard_html

    report_path.write_text(
        existing_html,
        encoding="utf-8",
    )

    print(
        f"Added aggregation dashboard to {report_path.resolve()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append an aggregation dashboard to an existing benchmark report."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the raw benchmark results.csv file.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="Directory containing report.html to update.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    rows = load_rows_from_csv(args.csv)
    if not rows:
        raise ValueError(f"CSV file contains no benchmark rows: {args.csv}")

    add_aggregation_dashboard(rows, args.report_dir)


if __name__ == "__main__":
    main()