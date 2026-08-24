"""Find routes in the stored accessibility graph and calculate route metrics.

The agent profiles are experimental routing preferences, not medical definitions.

Examples:
    python route_metrics.py --city lviv_ua --area dense_central \
        --start-lat 49.840 --start-lon 24.030 \
        --end-lat 49.850 --end-lon 24.045

    python route_metrics.py --city kyiv_ua --area all \
        --start-lat 50.445 --start-lon 30.515 \
        --end-lat 50.455 --end-lon 30.525 --agent all
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pymongo import MongoClient


AGENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "wheelchair": {
        "max_incline_percent": 5,
        "preferred_incline_percent": 3,
        "min_width_m": 1.2,
        "avoid_stairs": True,
        "surface_penalties": {
            "smooth_paved": 0.0,
            "paved_stones": 0.5,
            "cobblestone": 2.0,
            "gravel": 3.0,
            "unpaved": 5.0,
        },
        "weights": {
            "distance": 1.0,
            "incline": 3.0,
            "surface": 3.0,
            "width": 3.0,
            "stairs": 10.0,
            "unknown": 2.0,
        },
    },
    "limited_mobility": {
        "max_incline_percent": 8,
        "preferred_incline_percent": 5,
        "min_width_m": 1.0,
        "avoid_stairs": True,
        "surface_penalties": {
            "smooth_paved": 0.0,
            "paved_stones": 0.3,
            "cobblestone": 1.0,
            "gravel": 2.0,
            "unpaved": 3.0,
        },
        "weights": {
            "distance": 1.0,
            "incline": 2.0,
            "surface": 2.0,
            "width": 2.0,
            "stairs": 7.0,
            "unknown": 1.5,
        },
    },
    "low_endurance": {
        "max_incline_percent": 10,
        "preferred_incline_percent": 5,
        "min_width_m": 0.9,
        "avoid_stairs": False,
        "surface_penalties": {
            "smooth_paved": 0.0,
            "paved_stones": 0.2,
            "cobblestone": 0.8,
            "gravel": 1.5,
            "unpaved": 2.0,
        },
        "weights": {
            "distance": 2.0,
            "incline": 2.5,
            "surface": 1.5,
            "width": 1.0,
            "stairs": 4.0,
            "unknown": 1.0,
        },
    },
}

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class GraphEdge:
    edge_id: Any
    start: int
    end: int
    length_m: float
    features: Dict[str, Any]
    raw_tags: Dict[str, Any]


@dataclass
class RouteResult:
    agent: str
    start_node: int
    end_node: int
    edge_ids: List[Any]
    metrics: Dict[str, Any]


def haversine_m(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    """Calculate distance between (latitude, longitude) pairs in metres."""

    first_lat, first_lon = map(math.radians, first)
    second_lat, second_lon = map(math.radians, second)
    delta_lat = second_lat - first_lat
    delta_lon = second_lon - first_lon
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(first_lat)
        * math.cos(second_lat)
        * math.sin(delta_lon / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.asin(math.sqrt(value))


def edge_length_m(edge: GraphEdge, nodes: Dict[int, Tuple[float, float]]) -> float:
    if edge.length_m > 0:
        return edge.length_m

    node_ids = edge.raw_tags.get("_node_ids", [edge.start, edge.end])
    geometry = [nodes[node_id] for node_id in node_ids if node_id in nodes]
    return sum(
        haversine_m(first, second)
        for first, second in zip(geometry, geometry[1:])
    )


def load_graph(
    city_id: str,
    area_id: str,
) -> Tuple[Dict[int, Tuple[float, float]], List[GraphEdge]]:
    load_dotenv()
    uri = os.environ.get("MONGODB_URI")
    database_name = os.environ.get("MONGODB_DATABASE", "accessible_routing")
    if not uri:
        raise RuntimeError("Set MONGODB_URI in the environment or .env file.")

    client = MongoClient(uri, serverSelectionTimeoutMS=30_000)
    try:
        database = client[database_name]
        area_filter = {"city_id": city_id}
        if area_id != "all":
            area_filter["area_id"] = area_id

        area_documents = list(
            database.cities.find(
                area_filter,
                {"_id": 0, "area_id": 1},
            )
        )
        if not area_documents:
            available = sorted(
                document["area_id"]
                for document in database.cities.find(
                    {"city_id": city_id},
                    {"_id": 0, "area_id": 1},
                )
            )
            if not available:
                raise ValueError(
                    f"Unknown city '{city_id}' or it has no saved areas."
                )
            raise ValueError(
                f"Unknown area '{area_id}' for {city_id}. "
                f"Available areas: {', '.join(available)}"
            )

        node_documents = database.nodes.find(
            area_filter,
            {"osm_node_id": 1, "lat": 1, "lon": 1},
        )
        edge_documents = database.edges.find(
            area_filter,
            {"osm_way_id": 1, "node_ids": 1, "geometry": 1, "features": 1, "raw_tags": 1},
        )

        nodes = {
            int(document["osm_node_id"]): (
                float(document["lat"]),
                float(document["lon"]),
            )
            for document in node_documents
        }
        edges = []
        for document in edge_documents:
            node_ids = [int(node_id) for node_id in document.get("node_ids", [])]
            if len(node_ids) < 2 or node_ids[0] not in nodes or node_ids[-1] not in nodes:
                continue
            geometry = document.get("geometry", [])
            length_m = sum(
                haversine_m((first[1], first[0]), (second[1], second[0]))
                for first, second in zip(geometry, geometry[1:])
            )
            raw_tags = dict(document.get("raw_tags", {}))
            raw_tags["_node_ids"] = node_ids
            edges.append(
                GraphEdge(
                    edge_id=document.get("osm_way_id"),
                    start=node_ids[0],
                    end=node_ids[-1],
                    length_m=length_m,
                    features=dict(document.get("features", {})),
                    raw_tags=raw_tags,
                )
            )
        return nodes, edges
    finally:
        client.close()


def nearest_node(
    nodes: Dict[int, Tuple[float, float]],
    point: Tuple[float, float],
) -> int:
    if not nodes:
        raise ValueError("The selected area contains no nodes.")
    return min(nodes, key=lambda node_id: haversine_m(nodes[node_id], point))


def is_oneway(edge: GraphEdge) -> bool:
    return str(edge.raw_tags.get("oneway", "")).lower() in {"yes", "true", "1"}


def get_incline(edge: GraphEdge) -> Tuple[Optional[float], str]:
    osm_incline = edge.features.get("incline_percent")
    if osm_incline is not None:
        return float(osm_incline), "osm"

    dem = edge.features.get("dem", {})
    dem_incline = dem.get("dem_incline_percent")
    if dem_incline is not None:
        return float(dem_incline), "fabdem_v1_2"

    return None, "unknown"


def build_adjacency(
    edges: Sequence[GraphEdge],
    nodes: Dict[int, Tuple[float, float]],
) -> Dict[int, List[Tuple[int, GraphEdge]]]:
    adjacency: Dict[int, List[Tuple[int, GraphEdge]]] = {}
    for edge in edges:
        node_ids = edge.raw_tags.get("_node_ids", [edge.start, edge.end])
        node_ids = [node_id for node_id in node_ids if node_id in nodes]
        for segment_index, (start, end) in enumerate(zip(node_ids, node_ids[1:])):
            segment = GraphEdge(
                edge_id=edge.edge_id,
                start=start,
                end=end,
                length_m=haversine_m(nodes[start], nodes[end]),
                features=edge.features,
                raw_tags={
                    **edge.raw_tags,
                    "_segment_index": segment_index,
                },
            )
            adjacency.setdefault(start, []).append((end, segment))
            if not is_oneway(edge):
                adjacency.setdefault(end, []).append((start, segment))
    return adjacency


def classify_edge(edge: GraphEdge, profile: Dict[str, Any]) -> Dict[str, Any]:
    features = edge.features
    incline, incline_source = get_incline(edge)
    width = features.get("width_m")
    surface = features.get("surface")
    stairs = features.get("stairs") is True or edge.raw_tags.get("highway") == "steps"

    violations: List[str] = []
    critical: List[str] = []
    if incline is not None and abs(float(incline)) > profile["max_incline_percent"]:
        violations.append("excessive_incline")
    if width is not None and float(width) < profile["min_width_m"]:
        violations.append("insufficient_width")
    if stairs and profile["avoid_stairs"]:
        violations.append("stairs")
        critical.append("stairs")
    if surface in {"unpaved", "gravel", "cobblestone"}:
        violations.append("unacceptable_surface")

    unknown_features = sum(
        features.get(name) is None
        for name in ("surface", "smoothness", "width_m", "incline_percent", "wheelchair", "kerb", "ramp")
    )
    weights = profile["weights"]
    cost = edge.length_m / 1000 * weights["distance"]
    if incline is None:
        cost += weights["unknown"]
    else:
        excess_incline = max(0.0, abs(float(incline)) - profile["preferred_incline_percent"])
        cost += excess_incline * weights["incline"]
    if surface is None:
        cost += weights["unknown"]
    else:
        cost += profile["surface_penalties"].get(surface, 1.0) * weights["surface"]
    if width is None:
        cost += weights["unknown"]
    else:
        cost += max(0.0, profile["min_width_m"] - float(width)) * weights["width"]
    if stairs:
        cost += weights["stairs"]
    cost += unknown_features * weights["unknown"] / 7

    return {
        "cost": cost,
        "violations": violations,
        "critical_violations": critical,
        "stairs": stairs,
        "incline_percent": None if incline is None else float(incline),
        "incline_source": incline_source,
        "width_m": None if width is None else float(width),
        "surface": surface,
        "unknown_features": unknown_features,
    }


def find_route(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start: Tuple[float, float],
    end: Tuple[float, float],
    agent_name: str,
) -> RouteResult:
    if agent_name not in AGENT_PROFILES:
        raise ValueError(f"Unknown agent '{agent_name}'. Choose from: {', '.join(AGENT_PROFILES)}")

    start_node = nearest_node(nodes, start)
    end_node = nearest_node(nodes, end)
    adjacency = build_adjacency(edges, nodes)
    profile = AGENT_PROFILES[agent_name]
    distances = {start_node: 0.0}
    previous: Dict[int, Tuple[Optional[int], Optional[GraphEdge]]] = {start_node: (None, None)}
    queue: List[Tuple[float, int]] = [(0.0, start_node)]

    while queue:
        current_cost, current_node = heapq.heappop(queue)
        if current_cost > distances.get(current_node, math.inf):
            continue
        if current_node == end_node:
            break
        for next_node, edge in adjacency.get(current_node, []):
            candidate = current_cost + classify_edge(edge, profile)["cost"]
            if candidate < distances.get(next_node, math.inf):
                distances[next_node] = candidate
                previous[next_node] = (current_node, edge)
                heapq.heappush(queue, (candidate, next_node))

    if end_node not in previous:
        raise ValueError("No route exists between the nearest graph nodes for this area and agent.")

    route_edges: List[GraphEdge] = []
    current = end_node
    while current != start_node:
        previous_node, edge = previous[current]
        if previous_node is None or edge is None:
            raise RuntimeError("Route reconstruction failed.")
        route_edges.append(edge)
        current = previous_node
    route_edges.reverse()
    return RouteResult(
        agent=agent_name,
        start_node=start_node,
        end_node=end_node,
        edge_ids=[edge.edge_id for edge in route_edges],
        metrics=calculate_route_metrics(
            route_edges,
            profile,
            start_snap_distance_m=haversine_m(nodes[start_node], start),
            end_snap_distance_m=haversine_m(nodes[end_node], end),
        ),
    )


def calculate_route_metrics(
    route_edges: Sequence[GraphEdge],
    profile: Dict[str, Any],
    start_snap_distance_m: Optional[float] = None,
    end_snap_distance_m: Optional[float] = None,
) -> Dict[str, Any]:
    classifications = [classify_edge(edge, profile) for edge in route_edges]
    total_length = sum(edge.length_m for edge in route_edges)
    violations = [violation for item in classifications for violation in item["violations"]]
    critical = [violation for item in classifications for violation in item["critical_violations"]]
    known_inclines = [item["incline_percent"] for item in classifications if item["incline_percent"] is not None]
    known_widths = [item["width_m"] for item in classifications if item["width_m"] is not None]
    surface_counts: Dict[str, int] = {}
    for item in classifications:
        surface = item["surface"] or "unknown"
        surface_counts[surface] = surface_counts.get(surface, 0) + 1

    result: Dict[str, Any] = {
        "edge_count": len(route_edges),
        "length_m": round(total_length, 3),
        "travel_cost": round(sum(classify_edge(edge, profile)["cost"] for edge in route_edges), 3),
        "violating_edge_count": len({index for index, item in enumerate(classifications) if item["violations"]}),
        "violation_count": len(violations),
        "violation_rate": round(
            len({index for index, item in enumerate(classifications) if item["violations"]}) / len(route_edges),
            4,
        ) if route_edges else 0.0,
        "violations": {name: violations.count(name) for name in sorted(set(violations))},
        "critical_violation_count": len(critical),
        "has_critical_violation": bool(critical),
        "critical_violations": {name: critical.count(name) for name in sorted(set(critical))},
        "stairs_edge_count": sum(item["stairs"] for item in classifications),
        "stairs_fraction": round(sum(item["stairs"] for item in classifications) / len(route_edges), 4) if route_edges else 0.0,
        "incline": {
            "mean_abs_percent": round(sum(abs(value) for value in known_inclines) / len(known_inclines), 3) if known_inclines else None,
            "max_abs_percent": round(max((abs(value) for value in known_inclines), default=0.0), 3) if known_inclines else None,
            "above_max_fraction": round(sum(abs(value) > profile["max_incline_percent"] for value in known_inclines) / len(route_edges), 4) if route_edges else 0.0,
        },
        "width": {
            "mean_m": round(sum(known_widths) / len(known_widths), 3) if known_widths else None,
            "min_m": round(min(known_widths), 3) if known_widths else None,
            "below_min_fraction": round(sum(value < profile["min_width_m"] for value in known_widths) / len(route_edges), 4) if route_edges else 0.0,
        },
        "surface_distribution": {name: round(count / len(route_edges), 4) for name, count in surface_counts.items()} if route_edges else {},
        "unknown_feature_count": sum(item["unknown_features"] for item in classifications),
        "unknown_feature_rate": round(sum(item["unknown_features"] for item in classifications) / (len(route_edges) * 7), 4) if route_edges else 0.0,
    }
    if start_snap_distance_m is not None and end_snap_distance_m is not None:
        result["endpoint_snap_distance_m"] = {
            "start_m": round(start_snap_distance_m, 3),
            "end_m": round(end_snap_distance_m, 3),
            "total_m": round(start_snap_distance_m + end_snap_distance_m, 3),
        }
    return result


def plot_route_comparison(
    routes: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    successful = [route for route in routes if "metrics" in route]
    if not successful:
        return

    labels = [route["agent"] for route in successful]
    metrics = [route["metrics"] for route in successful]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    charts = [
        (axes[0, 0], "length_m", "Route length (m)", "#176b87"),
        (axes[0, 1], "travel_cost", "Agent travel cost", "#d97706"),
        (axes[1, 0], "violation_rate", "Violation rate", "#be123c"),
        (axes[1, 1], "incline.mean_abs_percent", "Mean absolute incline (%)", "#7c3aed"),
    ]
    for axis, metric_name, title, color in charts:
        values = []
        for metric in metrics:
            value: Any = metric
            for part in metric_name.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            values.append(0 if value is None else float(value))
        axis.bar(labels, values, color=color)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        if metric_name == "violation_rate":
            axis.set_ylim(0, 1)
            axis.set_ylabel("Fraction of route edges")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate agent-specific accessibility metrics for a route.")
    parser.add_argument("--city", required=True)
    parser.add_argument(
        "--area",
        required=True,
        help="Saved area_id, or 'all' to use the union of all city areas.",
    )
    parser.add_argument("--start-lat", required=True, type=float)
    parser.add_argument("--start-lon", required=True, type=float)
    parser.add_argument("--end-lat", required=True, type=float)
    parser.add_argument("--end-lon", required=True, type=float)
    parser.add_argument("--agent", choices=[*AGENT_PROFILES, "all"], default="all")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Save an agent comparison plot in this directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    nodes, edges = load_graph(args.city, args.area)
    agent_names: Iterable[str] = AGENT_PROFILES if args.agent == "all" else [args.agent]
    output = {
        "city_id": args.city,
        "area_id": args.area,
        "requested_start": {"lat": args.start_lat, "lon": args.start_lon},
        "requested_end": {"lat": args.end_lat, "lon": args.end_lon},
        "graph": {"node_count": len(nodes), "edge_count": len(edges)},
        "routes": [],
    }
    for agent_name in agent_names:
        try:
            route = find_route(
                nodes,
                edges,
                (args.start_lat, args.start_lon),
                (args.end_lat, args.end_lon),
                agent_name,
            )
            output["routes"].append({
                "agent": route.agent,
                "start_node": route.start_node,
                "end_node": route.end_node,
                "edge_ids": route.edge_ids,
                "metrics": route.metrics,
            })
        except ValueError as error:
            output["routes"].append({"agent": agent_name, "error": str(error)})
    if args.plot_dir is not None:
        plot_route_comparison(
            output["routes"],
            args.plot_dir / f"{args.city}_{args.area}_route_comparison.png",
        )
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
