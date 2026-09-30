"""Create descriptive plots for the accessible routing MongoDB dataset.

Examples:
    python dataset_plots.py --city tallinn_ee
    python dataset_plots.py --city all --output-dir dataset_plots
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient


ACCESSIBILITY_FIELDS = (
    "surface",
    "smoothness",
    "width_m",
    "incline_percent",
    "wheelchair",
    "kerb",
    "ramp",
    "sidewalk",
    "tactile_paving",
)

FEATURES = (
    "surface",
    "smoothness",
    "width_m",
    "incline_percent",
    "wheelchair",
    "kerb",
    "ramp",
    "sidewalk",
    "tactile_paving",
)


def fetch_edges(city_id: Optional[str]) -> List[Dict[str, Any]]:
    load_dotenv()
    uri = os.environ.get("MONGODB_URI")
    database_name = os.environ.get("MONGODB_DATABASE", "accessible_routing")
    if not uri:
        raise RuntimeError("Set MONGODB_URI in the environment or .env file.")
    client = MongoClient(uri, serverSelectionTimeoutMS=30_000)
    try:
        query = {"city_id": city_id} if city_id else {}
        return list(client[database_name].edges.find(
            query,
            {
                "_id": 0,
                "city_id": 1,
                "area_id": 1,
                "highway": 1,
                "features": 1,
            },
        ))
    finally:
        client.close()


def grouped_edges(edges: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        result[edge.get("city_id", "unknown")].append(edge)
    return dict(result)


def known_value(edge: Dict[str, Any], feature: str) -> Any:
    features = edge.get("features", {})
    if feature == "incline_percent":
        value = features.get("incline_percent")
        if value is None:
            value = features.get("dem", {}).get("dem_incline_percent")
        return value
    return features.get(feature)


def value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"unknown", "none", "null", "nan", "n/a", "na", "not_available", "not available"}
    return False


def normalize_area_type(area_id: Optional[str]) -> str:
    if area_id is None:
        return "unknown"
    area_key = str(area_id).strip().lower()
    if area_key in {"central_dense", "dense_central"}:
        return "central"
    return area_key


def compute_missingness_summary(edges: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        normalized_area = normalize_area_type(edge.get("area_id"))
        group_key = (edge.get("city_id", "unknown"), normalized_area)
        grouped[group_key].append(edge)

    summary: List[Dict[str, Any]] = []
    for (city_id, area_id), group_edges in sorted(grouped.items()):
        totals = {field: 0 for field in ACCESSIBILITY_FIELDS}
        missing = {field: 0 for field in ACCESSIBILITY_FIELDS}

        for edge in group_edges:
            for field in ACCESSIBILITY_FIELDS:
                value = known_value(edge, field)
                totals[field] += 1
                if value_is_missing(value):
                    missing[field] += 1

        field_rates = {}
        for field in ACCESSIBILITY_FIELDS:
            count = totals[field]
            if count:
                field_rates[field] = missing[field] / count
            else:
                field_rates[field] = 0.0

        mean_missing_rate = float(np.mean(list(field_rates.values()))) if field_rates else 0.0
        summary.append({
            "city_id": city_id,
            "area_id": area_id,
            "total_edges": len(group_edges),
            "mean_missing_rate": mean_missing_rate,
            "field_rates": field_rates,
        })

    return summary


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_dir / name, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_city_summary(city_id: str, edges: List[Dict[str, Any]], output_dir: Path) -> None:
    road_types = Counter(edge.get("highway", "unknown") for edge in edges)
    surfaces = Counter(known_value(edge, "surface") or "unknown" for edge in edges)
    observability = []
    for edge in edges:
        known = sum(known_value(edge, feature) is not None for feature in FEATURES)
        observability.append(known / len(FEATURES))

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(road_types.keys(), road_types.values(), color="#176b87")
    axes[0, 0].set_title("Pedestrian edges by road type")
    axes[0, 0].tick_params(axis="x", rotation=55)
    axes[0, 1].bar(surfaces.keys(), surfaces.values(), color="#d97706")
    axes[0, 1].set_title("Surface distribution")
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[1, 0].hist(observability, bins=[i / 10 for i in range(11)], color="#0f766e", edgecolor="white")
    axes[1, 0].set_title("Edge observability")
    axes[1, 0].set_xlabel("Known accessibility attributes / 9")
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.9,
        f"{city_id}\n\nEdges: {len(edges):,}\n"
        f"Mean observability: {sum(observability) / len(observability) * 100:.1f}%\n"
        f"Incline coverage: {sum(known_value(edge, 'incline_percent') is not None for edge in edges) / len(edges) * 100:.1f}%\n"
        f"Width coverage: {sum(known_value(edge, 'width_m') is not None for edge in edges) / len(edges) * 100:.1f}%",
        va="top",
        fontsize=13,
    )
    save_figure(figure, output_dir, f"{city_id}_summary.png")


def plot_city_distributions(city_id: str, edges: List[Dict[str, Any]], output_dir: Path) -> None:
    inclines = [abs(float(value)) for edge in edges if (value := known_value(edge, "incline_percent")) is not None]
    widths = [float(value) for edge in edges if (value := known_value(edge, "width_m")) is not None]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    if inclines:
        axes[0].hist(inclines, bins=40, range=(0, min(max(inclines), 50)), color="#be123c")
        axes[0].set_title("Absolute incline (capped at 50%)")
        axes[0].set_xlabel("Incline (%)")
    else:
        axes[0].text(0.5, 0.5, "No incline data", ha="center")
        axes[0].set_title("Absolute incline")
    if widths:
        axes[1].hist(widths, bins=40, range=(0, min(max(widths), 5)), color="#7c3aed")
        axes[1].set_title("Mapped width (capped at 5 m)")
        axes[1].set_xlabel("Width (m)")
    else:
        axes[1].text(0.5, 0.5, "No width data", ha="center")
        axes[1].set_title("Mapped width")
    save_figure(figure, output_dir, f"{city_id}_numeric_distributions.png")


def plot_missingness_heatmap(summary: List[Dict[str, Any]], output_dir: Path) -> None:
    city_order = sorted({row["city_id"] for row in summary})
    area_order = sorted({row["area_id"] for row in summary})
    matrix = np.full((len(city_order), len(area_order)), np.nan, dtype=float)

    row_index = {city: idx for idx, city in enumerate(city_order)}
    col_index = {area: idx for idx, area in enumerate(area_order)}

    for row in summary:
        i = row_index[row["city_id"]]
        j = col_index[row["area_id"]]
        matrix[i, j] = row["mean_missing_rate"]

    figure, ax = plt.subplots(figsize=(max(8, len(area_order) * 1.6), max(5, len(city_order) * 0.9)))
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(area_order)))
    ax.set_xticklabels(area_order, rotation=35, ha="right")
    ax.set_yticks(range(len(city_order)))
    ax.set_yticklabels(city_order)
    ax.set_title("Missing accessibility attributes by city and area type")
    ax.set_xlabel("Area type")
    ax.set_ylabel("City")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                text = "n/a"
            else:
                text = f"{value * 100:.0f}%"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    colorbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Missing rate (fraction)")
    save_figure(figure, output_dir, "accessibility_missingness_by_city_area.png")


def print_missingness_report(summary: List[Dict[str, Any]]) -> None:
    print("\nAccessibility missingness report (by city and area type)")
    for row in sorted(summary, key=lambda item: (item["city_id"], item["area_id"])):
        print(
            f"{row['city_id']} | {row['area_id']} | edges={row['total_edges']:,} | "
            f"mean missing rate={row['mean_missing_rate'] * 100:.1f}%"
        )
        for field, rate in sorted(row["field_rates"].items(), key=lambda item: item[1], reverse=True):
            print(f"  - {field}: {rate * 100:.1f}% missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot descriptive statistics for the routing dataset.")
    parser.add_argument("--city", default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_plots"))
    args = parser.parse_args()

    edges = fetch_edges(None if args.city == "all" else args.city)
    for city_id, city_edges in grouped_edges(edges).items():
        plot_city_summary(city_id, city_edges, args.output_dir)
        plot_city_distributions(city_id, city_edges, args.output_dir)
        print(f"{city_id}: plotted {len(city_edges):,} edges")

    missingness_summary = compute_missingness_summary(edges)
    plot_missingness_heatmap(missingness_summary, args.output_dir)
    print_missingness_report(missingness_summary)
    print(f"Plots written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
