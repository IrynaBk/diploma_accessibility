"""Adapter layer for pluggable routing implementations.

This keeps the project's existing route experiments generic: each implementation
can live in its own file, but the routing experiment runner calls a common
interface to obtain a route result.
"""

from __future__ import annotations

import heapq
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from heliyon import (
    Config as HeliyonConfig,
    SegmentSpec,
    UserProfile,
    ai_to_level,
    evaluate_segment,
    paper_example_profile,
)
from kaselouris import Config as KaselourisConfig
from kaselouris import classify_edges, route as kaselouris_route
from route_metrics import AGENT_PROFILES, GraphEdge, classify_edge
from vrupod import (
    evaluate_city_vrupod_policy,
    train_city_vrupod_policy,
)


PROFILE_MAP = {
    "wheelchair": "mobility",
    "limited_mobility": "mobility",
    "low_endurance": "able",
}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text.endswith("%"):
            text = text[:-1]
        match = re.match(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$", text)
        if match:
            return float(text)
    return None


def _normalize_heliyon_surface(value: Any) -> Optional[str]:
    if value is None:
        return None
    surface = str(value).strip().lower().replace("_", " ")
    mapping = {
        "asphalt": "asphalt",
        "concrete": "concrete",
        "brick": "brick",
        "cobblestone": "cobblestone",
        "gravel": "gravel",
        "paving stones": "brick",
        "paving_stones": "brick",
        "sett": "cobblestone",
        "sett paving": "cobblestone",
        "sand": "gravel",
        "unpaved": "gravel",
    }
    return mapping.get(surface, surface or None)


def _heliyon_quality_from_features(features: Dict[str, Any]) -> Optional[float]:
    smoothness = features.get("smoothness")
    if smoothness is None:
        return None
    smoothness_key = str(smoothness).lower()
    quality_map = {
        "excellent": 10.0,
        "good": 8.0,
        "intermediate": 6.0,
        "bad": 4.0,
        "very_bad": 2.0,
        "horrible": 1.0,
        "very_horrible": 0.5,
        "impassable": 0.0,
    }
    return quality_map.get(smoothness_key)


def _edge_to_heliyon_segment(edge: GraphEdge) -> SegmentSpec:
    attrs = dict(edge.features)
    width = attrs.get("width_m")
    if width is None:
        width = attrs.get("width")
    width_val = _as_float(width)

    incline_value = attrs.get("incline_percent")
    if incline_value is None and "incline" in attrs:
        incline_value = attrs.get("incline")
    if incline_value is None:
        incline_value = attrs.get("grade_abs")
    if incline_value is not None:
        if isinstance(incline_value, str):
            match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", incline_value)
            if match:
                incline_value = float(match.group(1))
            else:
                incline_value = None
        else:
            incline_value = float(incline_value)
    if incline_value is not None and isinstance(incline_value, (int, float)) and "grade_abs" in attrs and attrs.get("grade_abs") is not None:
        incline_value = float(attrs["grade_abs"]) * 100.0

    cross_value = attrs.get("cross_slope")
    if cross_value is None:
        cross_value = attrs.get("incline:across")
    if cross_value is not None:
        if isinstance(cross_value, str):
            match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", cross_value)
            cross_value = float(match.group(1)) if match else None
        else:
            cross_value = float(cross_value)

    height = attrs.get("height_change_cm")
    if height is None:
        height = attrs.get("height_change")
    if height is not None:
        height = _as_float(height)
        if height is not None and "cm" in str(attrs.get("height_change", "")).lower():
            height = height / 100.0
    height_cm = None if height is None else height * 100.0 if height <= 1.0 else height

    return SegmentSpec(
        length_m=float(edge.length_m),
        width_m=width_val,
        long_slope_pct=None if incline_value is None else (float(incline_value), float(incline_value)),
        cross_slope_pct=None if cross_value is None else (abs(float(cross_value)), abs(float(cross_value))),
        surface=_normalize_heliyon_surface(attrs.get("surface") or attrs.get("surface_type")),
        surface_quality=_heliyon_quality_from_features(attrs) or _as_float(attrs.get("surface_quality")),
        height_change_cm=height_cm,
        is_ramp=bool(str(attrs.get("ramp", "")).lower() in {"yes", "true", "1"}),
    )


def _heliyon_profile_for_agent(agent_name: str) -> UserProfile:
    profile = paper_example_profile()
    if agent_name == "wheelchair":
        profile.importance.update({"width": 1.8, "cross_slope": 1.6, "long_slope": 1.9, "texture": 1.4, "surface_quality": 1.3, "height_change": 1.7})
    elif agent_name == "limited_mobility":
        profile.importance.update({"width": 1.5, "cross_slope": 1.5, "long_slope": 1.6, "texture": 1.2, "surface_quality": 1.2, "height_change": 1.5})
    elif agent_name == "low_endurance":
        profile.importance.update({"width": 1.2, "cross_slope": 1.4, "long_slope": 1.3, "texture": 1.1, "surface_quality": 1.0, "height_change": 1.3})
    return profile


def _heliyon_route_segment_cost(edge: GraphEdge, profile: UserProfile, cfg: HeliyonConfig) -> Tuple[float, float, float, str]:
    spec = _edge_to_heliyon_segment(edge)
    result = evaluate_segment(spec, profile, cfg)
    ai_mean = result.ai_mean
    tier = ai_to_level(ai_mean, cfg)
    score = 1.0 / max(ai_mean, 1e-6)
    violations = 1 if ai_mean < 0.35 else 0
    return score, ai_mean, float(result.cost), tier


def _heliyon_route(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
    agent_name: str,
    cfg: Optional[HeliyonConfig] = None,
    uncertainty_weight: float = 0.0,
    algorithm_name: str = "heliyon",
) -> Dict[str, Any]:
    cfg = cfg or HeliyonConfig()
    profile = _heliyon_profile_for_agent(agent_name)

    adjacency: Dict[int, List[Tuple[int, GraphEdge, float]]] = {node_id: [] for node_id in nodes}
    for edge in edges:
        if edge.start not in nodes or edge.end not in nodes:
            continue
        edge_cost, _, _, _ = _heliyon_route_segment_cost(edge, profile, cfg)
        if uncertainty_weight > 0.0:
            attrs = edge.features
            factor_fields = (
                ("width_m", "width"),
                ("cross_slope", "incline:across"),
                ("incline_percent", "incline", "grade_abs"),
                ("surface", "surface_type"),
                ("smoothness", "surface_quality"),
                ("height_change_cm", "height_change"),
            )
            unknown_count = sum(
                not any(attrs.get(field) is not None for field in fields)
                for fields in factor_fields
            )
            # Scale the risk in the same distance-weighted objective as Heliyon's
            # segment score, so unknown attributes influence long and short edges
            # proportionally without changing the regular algorithm.
            edge_cost += uncertainty_weight * (unknown_count / len(factor_fields))
        adjacency.setdefault(edge.start, []).append((edge.end, edge, edge_cost))
        if not edge.oneway:
            adjacency.setdefault(edge.end, []).append((edge.start, edge, edge_cost))

    if start_node not in adjacency or goal_node not in adjacency:
        return {
            "algorithm": algorithm_name,
            "agent": agent_name,
            "profile": "heliyon",
            "tier": "none",
            "success": False,
            "path_node_count": 0,
            "path_nodes": [],
            "length_m": None,
            "travel_cost_s": None,
            "violations": None,
            "ai_mean": None,
            "details": {"accessible": False, "alternative": False, "fallback": True},
        }

    distances: Dict[int, float] = {start_node: 0.0}
    previous: Dict[int, int] = {}
    heap: List[Tuple[float, int]] = [(0.0, start_node)]

    while heap:
        current_cost, current = heapq.heappop(heap)
        if current_cost > distances.get(current, float("inf")):
            continue
        if current == goal_node:
            break
        for neighbor, edge, edge_cost in adjacency.get(current, []):
            new_cost = current_cost + edge_cost * max(edge.length_m, 1.0)
            if new_cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_cost
                previous[neighbor] = current
                heapq.heappush(heap, (new_cost, neighbor))

    if goal_node not in distances:
        return {
            "algorithm": algorithm_name,
            "agent": agent_name,
            "profile": "heliyon",
            "tier": "none",
            "success": False,
            "path_node_count": 0,
            "path_nodes": [],
            "length_m": None,
            "travel_cost_s": None,
            "violations": None,
            "ai_mean": None,
            "details": {"accessible": False, "alternative": False, "fallback": True},
        }

    path_nodes: List[int] = []
    cursor = goal_node
    while cursor in previous or cursor == start_node:
        path_nodes.append(cursor)
        if cursor == start_node:
            break
        cursor = previous[cursor]
    path_nodes.reverse()

    traversed_edges: List[GraphEdge] = []
    for index in range(len(path_nodes) - 1):
        current = path_nodes[index]
        next_node = path_nodes[index + 1]
        candidates = [edge for neighbor, edge, _ in adjacency.get(current, []) if neighbor == next_node]
        if candidates:
            traversed_edges.append(candidates[0])

    total_length = sum(edge.length_m for edge in traversed_edges)
    violation_count = 0
    total_ai = 0.0
    for edge in traversed_edges:
        _, ai, _, _ = _heliyon_route_segment_cost(edge, profile, cfg)
        total_ai += ai
        if ai < 0.35:
            violation_count += 1

    mean_ai = total_ai / len(traversed_edges) if traversed_edges else 0.0
    tier = ai_to_level(mean_ai, cfg)
    return {
        "algorithm": algorithm_name,
        "agent": agent_name,
        "profile": "heliyon",
        "tier": {"very low": "inaccessible", "low": "alternative", "medium": "accessible", "high": "accessible"}.get(tier, tier),
        "success": bool(traversed_edges),
        "path_node_count": len(path_nodes),
        "path_nodes": list(path_nodes),
        "length_m": round(total_length, 3),
        "travel_cost_s": round(sum((1.0 / max(_heliyon_route_segment_cost(edge, profile, cfg)[1], 1e-6)) * max(edge.length_m, 1.0) for edge in traversed_edges), 3),
        "violations": violation_count,
        "ai_mean": mean_ai,
        "details": {"accessible": mean_ai >= 0.35, "alternative": mean_ai >= 0.2, "fallback": False},
    }


def find_connected_route_pair(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_point: Tuple[float, float],
    goal_point: Tuple[float, float],
) -> Tuple[int, int]:
    valid_nodes = {node_id for edge in edges for node_id in (edge.start, edge.end)}
    candidate_nodes = {node_id: coords for node_id, coords in nodes.items() if node_id in valid_nodes}
    if not candidate_nodes:
        raise ValueError("The selected graph contains no valid route nodes.")

    from route_metrics import haversine_m, nearest_node

    graph = nx.Graph()
    graph.add_nodes_from(candidate_nodes)
    for edge in edges:
        if edge.start in candidate_nodes and edge.end in candidate_nodes:
            graph.add_edge(edge.start, edge.end)

    start_node = nearest_node(candidate_nodes, start_point)
    goal_node = nearest_node(candidate_nodes, goal_point)
    if start_node in graph and goal_node in graph and nx.has_path(graph, start_node, goal_node):
        return start_node, goal_node

    best_pair: Optional[Tuple[float, int, int]] = None
    for component in nx.connected_components(graph):
        component_nodes = {node_id: candidate_nodes[node_id] for node_id in component}
        if not component_nodes:
            continue
        comp_start = nearest_node(component_nodes, start_point)
        comp_goal = nearest_node(component_nodes, goal_point)
        if not nx.has_path(graph, comp_start, comp_goal):
            continue
        score = (
            haversine_m(component_nodes[comp_start], start_point)
            + haversine_m(component_nodes[comp_goal], goal_point)
        )
        if best_pair is None or score < best_pair[0]:
            best_pair = (score, comp_start, comp_goal)

    if best_pair is None:
        raise ValueError("No connected route pair exists in the selected graph.")
    return best_pair[1], best_pair[2]


def _resolve_connected_pair(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
) -> Tuple[int, int]:
    from route_metrics import nearest_node

    valid_nodes = {node_id for edge in edges for node_id in (edge.start, edge.end)}
    filtered_nodes = {node_id: coords for node_id, coords in nodes.items() if node_id in valid_nodes}
    if not filtered_nodes:
        raise ValueError("The selected graph contains no valid route nodes.")

    graph = nx.Graph()
    graph.add_nodes_from(filtered_nodes)
    for edge in edges:
        if edge.start in filtered_nodes and edge.end in filtered_nodes:
            graph.add_edge(edge.start, edge.end)

    if start_node not in filtered_nodes:
        start_node = nearest_node(filtered_nodes, next(iter(filtered_nodes.values())))
    if goal_node not in filtered_nodes:
        goal_node = nearest_node(filtered_nodes, next(iter(filtered_nodes.values())))

    if nx.has_path(graph, start_node, goal_node):
        return start_node, goal_node

    start_point = nodes.get(start_node, next(iter(filtered_nodes.values())))
    goal_point = nodes.get(goal_node, next(iter(filtered_nodes.values())))
    return find_connected_route_pair(filtered_nodes, edges, start_point, goal_point)


def _normalize_to_kaselouris_attrs(edge: GraphEdge) -> Dict[str, Any]:
    attrs: Dict[str, Any] = dict(edge.features)

    if attrs.get("width_m") is not None:
        attrs["width"] = float(attrs["width_m"])
    if attrs.get("incline_percent") is not None:
        incline = float(attrs["incline_percent"])
        attrs["incline"] = f"{incline}%"
        attrs["grade_abs"] = abs(incline) / 100.0
    if attrs.get("surface") is not None:
        attrs["surface"] = attrs["surface"]
    if attrs.get("tactile_paving") is not None:
        attrs["tactile_paving"] = attrs["tactile_paving"]
    attrs.setdefault("highway", edge.highway or "footway")
    attrs.setdefault("length", float(edge.length_m))
    return attrs


def build_kaselouris_graph(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node_id, (lat, lon) in nodes.items():
        graph.add_node(node_id, lat=lat, lon=lon)

    for edge in edges:
        attrs = _normalize_to_kaselouris_attrs(edge)
        data = {**attrs, "length": float(edge.length_m)}
        graph.add_edge(
            edge.start,
            edge.end,
            key=edge.edge_id,
            **data,
        )
        if not edge.oneway:
            reverse_data = {**attrs, "length": float(edge.length_m)}
            graph.add_edge(
                edge.end,
                edge.start,
                key=edge.edge_id,
                **reverse_data,
            )

    return graph


def _edge_uncertainty_penalty(edge_data: Dict[str, Any], profile_name: str) -> float:
    profile_key = f"acc_{profile_name}"
    accessible = bool(edge_data.get(profile_key, False))
    if accessible:
        return 0.0
    unknown = edge_data.get(f"{profile_key}_unknown", [])
    fails = edge_data.get(f"{profile_key}_fail", [])
    unknown_count = len(unknown) if isinstance(unknown, list) else 0
    fail_count = len(fails) if isinstance(fails, list) else 0
    return 1.0 + 0.75 * unknown_count + 1.5 * fail_count


def _robust_kaselouris_path(
    graph: nx.MultiDiGraph,
    start_node: int,
    goal_node: int,
    profile_name: str,
    uncertainty_weight: float = 1.0,
) -> Tuple[List[int], float, int, float]:
    speed = 1.4
    dist: Dict[int, float] = {start_node: 0.0}
    prev: Dict[int, Optional[int]] = {start_node: None}
    heap: List[Tuple[float, int]] = [(0.0, start_node)]

    while heap:
        current_cost, current_node = heapq.heappop(heap)
        if current_cost > dist.get(current_node, float("inf")):
            continue
        if current_node == goal_node:
            break

        for _, neighbor, _, data in graph.edges(current_node, keys=True, data=True):
            edge_length = float(data.get("length", 0.0) or 0.0)
            edge_risk = _edge_uncertainty_penalty(data, profile_name)
            is_accessible = bool(data.get(f"acc_{profile_name}", False))
            edge_penalty = 0.0 if is_accessible else uncertainty_weight * edge_risk * 100.0
            step_cost = edge_length / speed + edge_penalty
            alternative_cost = current_cost + step_cost
            if alternative_cost < dist.get(neighbor, float("inf")):
                dist[neighbor] = alternative_cost
                prev[neighbor] = current_node
                heapq.heappush(heap, (alternative_cost, neighbor))

    if goal_node not in dist:
        return [], float("inf"), 0, 0.0

    path_nodes: List[int] = []
    current = goal_node
    while current is not None:
        path_nodes.append(current)
        current = prev.get(current)
    path_nodes.reverse()

    violations = 0
    total_length = 0.0
    for index in range(len(path_nodes) - 1):
        current = path_nodes[index]
        nxt = path_nodes[index + 1]
        edge_attr = graph.get_edge_data(current, nxt)
        if not edge_attr:
            continue
        edge = next(iter(edge_attr.values()))
        total_length += float(edge.get("length", 0.0) or 0.0)
        if not bool(edge.get(f"acc_{profile_name}", False)):
            violations += 1

    return path_nodes, dist[goal_node], violations, total_length


def run_kaselouris_experiment(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
    agent_name: str = "wheelchair",
    missing_fraction: Optional[float] = None,
    seed: int = 0,
    _skip_pair_resolution: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    valid_nodes = {node_id for edge in edges for node_id in (edge.start, edge.end)}
    filtered_nodes = {node_id: coords for node_id, coords in nodes.items() if node_id in valid_nodes}
    if not filtered_nodes:
        raise ValueError("The selected graph contains no valid route nodes.")

    if not _skip_pair_resolution:
        start_node, goal_node = _resolve_connected_pair(filtered_nodes, edges, start_node, goal_node)

    if missing_fraction is not None:
        from routing_experiments import apply_observability_mask

        edges = apply_observability_mask(edges, float(missing_fraction), seed=seed)

    profile_name = PROFILE_MAP.get(agent_name, "mobility")
    graph = build_kaselouris_graph(filtered_nodes, edges)
    cfg = classify_edges(graph, cfg=KaselourisConfig(missing_policy="pass", missing_by_criterion={"obstacle": "pass"}))
    route_result = kaselouris_route(graph, start_node, goal_node, profile_name, cfg, compute_baseline=False)

    recommended = route_result.recommended
    return {
        "algorithm": "kaselouris",
        "agent": agent_name,
        "profile": profile_name,
        "tier": route_result.tier,
        "success": recommended is not None,
        "path_node_count": len(recommended.nodes) if recommended else 0,
        "path_nodes": list(recommended.nodes) if recommended else [],
        "length_m": round(recommended.length, 3) if recommended else None,
        "travel_cost_s": round(recommended.cost, 3) if recommended else None,
        "violations": recommended.violations if recommended else None,
        "details": {
            "accessible": route_result.accessible is not None,
            "alternative": route_result.alternative is not None,
            "fallback": route_result.fallback is not None,
        },
    }


def run_kaselouris_robust_experiment(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
    agent_name: str = "wheelchair",
    missing_fraction: Optional[float] = None,
    seed: int = 0,
    uncertainty_weight: float = 1.0,
    _skip_pair_resolution: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    valid_nodes = {node_id for edge in edges for node_id in (edge.start, edge.end)}
    filtered_nodes = {node_id: coords for node_id, coords in nodes.items() if node_id in valid_nodes}
    if not filtered_nodes:
        raise ValueError("The selected graph contains no valid route nodes.")

    if not _skip_pair_resolution:
        start_node, goal_node = _resolve_connected_pair(filtered_nodes, edges, start_node, goal_node)
    if missing_fraction is not None:
        from routing_experiments import apply_observability_mask

        edges = apply_observability_mask(edges, float(missing_fraction), seed=seed)

    profile_name = PROFILE_MAP.get(agent_name, "mobility")
    graph = build_kaselouris_graph(filtered_nodes, edges)
    cfg = classify_edges(graph, cfg=KaselourisConfig(missing_policy="pass", missing_by_criterion={"obstacle": "pass"}))
    route_result = kaselouris_route(graph, start_node, goal_node, profile_name, cfg, compute_baseline=False)

    robust_path_nodes, _, robust_violations, robust_length = _robust_kaselouris_path(
        graph,
        start_node,
        goal_node,
        profile_name,
        uncertainty_weight=uncertainty_weight,
    )

    recommended = route_result.recommended
    base_nodes = list(recommended.nodes) if recommended is not None else []
    if recommended is not None and route_result.tier in {"accessible", "alternative"}:
        base_violations = recommended.violations
        robust_penalty = robust_violations * 500.0 + robust_length * 0.05
        base_penalty = base_violations * 500.0 + recommended.length * 0.05
        use_robust = robust_penalty < base_penalty
        chosen_nodes = robust_path_nodes if use_robust else base_nodes
        chosen_violations = robust_violations if use_robust else base_violations
        chosen_length = robust_length if use_robust else recommended.length
        chosen_cost = robust_length / max(1.4, 1e-6) if use_robust else recommended.cost
        chosen_tier = "accessible" if chosen_violations == 0 else "alternative" if chosen_violations <= 1 else "inaccessible"
        success = bool(chosen_nodes)
    else:
        chosen_nodes = robust_path_nodes
        chosen_violations = robust_violations
        chosen_length = robust_length
        chosen_cost = robust_length / max(1.4, 1e-6)
        chosen_tier = "accessible" if chosen_violations == 0 else "alternative" if chosen_violations <= 1 else "inaccessible"
        success = bool(chosen_nodes)

    return {
        "algorithm": "kaselouris_robust",
        "agent": agent_name,
        "profile": profile_name,
        "tier": chosen_tier if chosen_nodes else "none",
        "success": success,
        "path_node_count": len(chosen_nodes),
        "path_nodes": list(chosen_nodes),
        "length_m": round(chosen_length, 3) if chosen_length is not None else None,
        "travel_cost_s": round(chosen_cost, 3) if chosen_cost is not None else None,
        "violations": chosen_violations if chosen_nodes else None,
        "details": {
            "accessible": chosen_violations == 0,
            "alternative": chosen_violations <= 1,
            "fallback": chosen_violations > 1,
        },
    }


def run_heliyon_experiment(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
    agent_name: str = "wheelchair",
    missing_fraction: Optional[float] = None,
    seed: int = 0,
    _skip_pair_resolution: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    valid_nodes = {node_id for edge in edges for node_id in (edge.start, edge.end)}
    filtered_nodes = {node_id: coords for node_id, coords in nodes.items() if node_id in valid_nodes}
    if not filtered_nodes:
        raise ValueError("The selected graph contains no valid route nodes.")

    if not _skip_pair_resolution:
        start_node, goal_node = _resolve_connected_pair(filtered_nodes, edges, start_node, goal_node)
    if missing_fraction is not None:
        from routing_experiments import apply_observability_mask

        edges = apply_observability_mask(edges, float(missing_fraction), seed=seed)

    return _heliyon_route(filtered_nodes, edges, start_node, goal_node, agent_name)


def run_heliyon_robust_experiment(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
    agent_name: str = "wheelchair",
    missing_fraction: Optional[float] = None,
    seed: int = 0,
    uncertainty_weight: float = 1.0,
    _skip_pair_resolution: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Prefer routes with fewer unknown accessibility attributes, with Heliyon fallback."""
    valid_nodes = {node_id for edge in edges for node_id in (edge.start, edge.end)}
    filtered_nodes = {node_id: coords for node_id, coords in nodes.items() if node_id in valid_nodes}
    if not filtered_nodes:
        raise ValueError("The selected graph contains no valid route nodes.")

    if not _skip_pair_resolution:
        start_node, goal_node = _resolve_connected_pair(filtered_nodes, edges, start_node, goal_node)
    if missing_fraction is not None:
        from routing_experiments import apply_observability_mask

        edges = apply_observability_mask(edges, float(missing_fraction), seed=seed)

    regular = _heliyon_route(filtered_nodes, edges, start_node, goal_node, agent_name)
    robust = _heliyon_route(
        filtered_nodes, edges, start_node, goal_node, agent_name,
        uncertainty_weight=max(0.0, float(uncertainty_weight)),
        algorithm_name="heliyon_robust",
    )
    if not robust.get("success"):
        chosen = regular
    elif not regular.get("success"):
        chosen = robust
    else:
        robust_penalty = (robust.get("violations") or 0) * 500.0 + (robust.get("length_m") or 0.0) * 0.05
        regular_penalty = (regular.get("violations") or 0) * 500.0 + (regular.get("length_m") or 0.0) * 0.05
        chosen = robust if robust_penalty < regular_penalty else regular

    return {**chosen, "algorithm": "heliyon_robust"}


def run_vrupod_experiment(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
    agent_name: str = "wheelchair",
    missing_fraction: Optional[float] = None,
    seed: int = 0,
    city_id: Optional[str] = None,
    area_id: Optional[str] = None,
    model_dir: str = "models/vrupod",
    route_count: int = 4,
    episodes: int = 300,
    train_if_missing: bool = True,
    max_steps: int = 500,
    n_blocked: int = 0,
    quality_bias: float = 1.0,
    _skip_pair_resolution: bool = False,
) -> Dict[str, Any]:
    valid_nodes = {node_id for edge in edges for node_id in (edge.start, edge.end)}
    filtered_nodes = {node_id: coords for node_id, coords in nodes.items() if node_id in valid_nodes}
    if not filtered_nodes:
        raise ValueError("The selected graph contains no valid route nodes.")

    if missing_fraction is not None:
        from routing_experiments import apply_observability_mask

        edges = apply_observability_mask(edges, float(missing_fraction), seed=seed)

    if not _skip_pair_resolution:
        start_node, goal_node = _resolve_connected_pair(filtered_nodes, edges, start_node, goal_node)
    benchmark_city = city_id or area_id or "unknown_city"
    model_root = Path(model_dir) / benchmark_city / agent_name

    if train_if_missing and not model_root.exists():
        train_city_vrupod_policy(
            benchmark_city,
            filtered_nodes,
            edges,
            agent_name=agent_name,
            route_count=max(2, route_count),
            seed=seed,
            episodes=episodes,
            output_dir=model_dir,
        )

    evaluations = evaluate_city_vrupod_policy(
        benchmark_city,
        filtered_nodes,
        edges,
        [(start_node, goal_node)],
        agent_name=agent_name,
        model_dir=model_dir,
        seed=seed,
        max_steps=max_steps,
        n_blocked=n_blocked,
        quality_bias=quality_bias,
    )
    if not evaluations:
        return {
            "algorithm": "vrupod",
            "agent": agent_name,
            "profile": "vrupod",
            "tier": "inaccessible",
            "success": False,
            "path_node_count": 0,
            "path_nodes": [],
            "length_m": None,
            "travel_cost_s": None,
            "violations": None,
            "details": {"accessible": False, "alternative": False, "fallback": True},
        }

    trip = evaluations[0]
    route_length = 0.0
    edge_map: Dict[Tuple[int, int], float] = {}
    for edge in edges:
        if edge.start in filtered_nodes and edge.end in filtered_nodes:
            edge_map[(edge.start, edge.end)] = float(edge.length_m)
            if not edge.oneway:
                edge_map[(edge.end, edge.start)] = float(edge.length_m)

    for prev, nxt in zip(trip["path"], trip["path"][1:]):
        route_length += edge_map.get((prev, nxt), 0.0)

    return {
        "algorithm": "vrupod",
        "agent": agent_name,
        "profile": "vrupod",
        "tier": "accessible" if trip.get("reached") else "inaccessible",
        "success": bool(trip.get("reached")),
        "path_node_count": len(trip.get("path", [])),
        "path_nodes": list(trip.get("path", [])),
        "length_m": round(route_length, 3) if route_length else None,
        "travel_cost_s": round(float(trip.get("cost", 0.0)), 3),
        "violations": 0 if trip.get("reached") else None,
        "details": {"accessible": trip.get("reached"), "alternative": False, "fallback": False},
    }


_ROBUST_IMPUTATION_FIELDS = {
    "width_m": ("width_m", "width"),
    "incline_percent": ("incline_percent", "incline"),
    "cross_slope": ("cross_slope", "incline:across"),
    "surface": ("surface", "surface_type"),
    "smoothness": ("smoothness",),
    "height_change_cm": ("height_change_cm", "height_change"),
}


def _robust_field_value(edge: GraphEdge, feature: str, aliases: Sequence[str]) -> Any:
    # Algorithms use feature values only. Never recover an absent or masked
    # value from raw OSM tags.
    if feature in edge.features:
        return edge.features[feature]
    for alias in aliases:
        if alias in edge.features:
            return edge.features[alias]
    return None


def _make_robust_scenarios(
    edges: Sequence[GraphEdge],
    seed: int,
    scenario_count: int,
) -> List[List[GraphEdge]]:
    rng = random.Random(seed)
    pools: Dict[str, List[Any]] = {}
    for feature, aliases in _ROBUST_IMPUTATION_FIELDS.items():
        values = [
            _robust_field_value(edge, feature, aliases)
            for edge in edges
        ]
        pools[feature] = [value for value in values if value is not None]

    scenarios: List[List[GraphEdge]] = [list(edges)]
    for _ in range(max(0, scenario_count)):
        scenario: List[GraphEdge] = []
        for edge in edges:
            features = dict(edge.features)
            for feature, aliases in _ROBUST_IMPUTATION_FIELDS.items():
                if _robust_field_value(edge, feature, aliases) is None and pools[feature]:
                    features[feature] = rng.choice(pools[feature])
            scenario.append(GraphEdge(
                edge_id=edge.edge_id,
                start=edge.start,
                end=edge.end,
                length_m=edge.length_m,
                features=features,
                raw_tags={}, oneway=edge.oneway, node_ids=edge.node_ids, highway=edge.highway,
            ))
        scenarios.append(scenario)
    return scenarios


def _score_common_route(
    path_nodes: Sequence[int],
    edges: Sequence[GraphEdge],
    profile: Dict[str, Any],
) -> Tuple[int, int, int, float, float]:
    adjacency: Dict[Tuple[int, int], List[GraphEdge]] = {}
    for edge in edges:
        adjacency.setdefault((edge.start, edge.end), []).append(edge)
        if not edge.oneway:
            adjacency.setdefault((edge.end, edge.start), []).append(edge)

    violating_edges = critical_edges = unknown_edges = 0
    length = 0.0
    for start, end in zip(path_nodes, path_nodes[1:]):
        candidates = adjacency.get((start, end), [])
        if not candidates:
            return (10**6, 10**6, 10**6, 10**6, float("inf"))
        edge = min(candidates, key=lambda candidate: classify_edge(candidate, profile)["cost"])
        classification = classify_edge(edge, profile)
        length += float(edge.length_m)
        violating_edges += bool(classification["violations"])
        critical_edges += bool(classification["critical_violations"])
        missing_factors = sum(
            _robust_field_value(edge, feature, aliases) is None
            for feature, aliases in _ROBUST_IMPUTATION_FIELDS.items()
        )
        unknown_edges += bool(classification["unknown_features"] or missing_factors)
    return critical_edges, violating_edges, unknown_edges, length, float(len(path_nodes) > 1)


def run_robust_consensus_experiment(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
    agent_name: str = "wheelchair",
    missing_fraction: Optional[float] = None,
    seed: int = 0,
    base_algorithm: str = "kaselouris",
    robust_scenarios: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run any registered router across dataset-derived imputations and rank paths by shared risk."""
    if base_algorithm == "robust_consensus" or base_algorithm not in IMPLEMENTATIONS:
        raise ValueError(f"Unsupported base algorithm '{base_algorithm}'.")
    if agent_name not in AGENT_PROFILES:
        raise ValueError(f"Unknown agent '{agent_name}'.")

    valid_nodes = {node_id for edge in edges for node_id in (edge.start, edge.end)}
    filtered_nodes = {node_id: coords for node_id, coords in nodes.items() if node_id in valid_nodes}
    if not filtered_nodes:
        raise ValueError("The selected graph contains no valid route nodes.")
    start_node, goal_node = _resolve_connected_pair(filtered_nodes, edges, start_node, goal_node)
    if missing_fraction is not None:
        from routing_experiments import apply_observability_mask

        edges = apply_observability_mask(edges, float(missing_fraction), seed=seed)

    scenarios = _make_robust_scenarios(edges, seed + 1, robust_scenarios)
    candidates: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for scenario_index, scenario_edges in enumerate(scenarios):
        print(
            f"[robust_consensus] base={base_algorithm} "
            f"scenario={scenario_index + 1}/{len(scenarios)}",
            flush=True,
        )
        result = run_registered_algorithm(
            base_algorithm, filtered_nodes, scenario_edges, start_node, goal_node,
            agent_name, missing_fraction=None, seed=seed + scenario_index,
            _skip_pair_resolution=True,
            **kwargs,
        )
        path = tuple(result.get("path_nodes") or ())
        if path and path not in candidates:
            candidates[path] = result

    if not candidates:
        return {
            "algorithm": "robust_consensus", "base_algorithm": base_algorithm,
            "agent": agent_name, "profile": "shared_accessibility", "tier": "none",
            "success": False, "path_node_count": 0, "path_nodes": [],
            "length_m": None, "travel_cost_s": None, "violations": None,
            "details": {"accessible": False, "alternative": False, "fallback": True},
        }

    profile = AGENT_PROFILES[agent_name]
    ranked: List[Tuple[Tuple[float, ...], Tuple[int, ...], Dict[str, Any]]] = []
    for path, result in candidates.items():
        measures = [_score_common_route(path, scenario, profile) for scenario in scenarios]
        worst_critical = max(item[0] for item in measures)
        worst_violations = max(item[1] for item in measures)
        mean_violations = sum(item[1] for item in measures) / len(measures)
        mean_unknown_edges = sum(item[2] for item in measures) / len(measures)
        path_length = measures[0][3]
        score = (worst_critical, worst_violations, mean_violations, mean_unknown_edges, path_length)
        ranked.append((score, path, result))

    score, chosen_path, chosen_result = min(ranked, key=lambda item: item[0])
    worst_violations = int(score[1])
    tier = "accessible" if worst_violations == 0 else "alternative" if worst_violations <= 1 else "inaccessible"
    details = dict(chosen_result.get("details") or {})
    details.update({
        "robust_method": "scenario_consensus",
        "scenario_count": len(scenarios),
        "worst_case_violating_edges": worst_violations,
        "expected_violating_edges": round(score[2], 3),
        "expected_unknown_edges": round(score[3], 3),
        "shared_evaluator": "route_metrics.classify_edge",
    })
    return {
        **chosen_result,
        "algorithm": "robust_consensus",
        "base_algorithm": base_algorithm,
        "profile": "shared_accessibility",
        "path_nodes": list(chosen_path),
        "path_node_count": len(chosen_path),
        "length_m": round(score[4], 3),
        "violations": worst_violations,
        "tier": tier,
        "details": details,
    }


IMPLEMENTATIONS = {
    "kaselouris": run_kaselouris_experiment,
    "kaselouris_robust": run_kaselouris_robust_experiment,
    "heliyon": run_heliyon_experiment,
    "heliyon_robust": run_heliyon_robust_experiment,
    "vrupod": run_vrupod_experiment,
    "robust_consensus": run_robust_consensus_experiment,
}


def run_registered_algorithm(
    name: str,
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
    agent_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    if name not in IMPLEMENTATIONS:
        raise ValueError(f"Unsupported routing implementation '{name}'. Available: {', '.join(sorted(IMPLEMENTATIONS))}")
    algorithm_edges = [
        GraphEdge(
            edge_id=edge.edge_id,
            start=edge.start,
            end=edge.end,
            length_m=edge.length_m,
            features=dict(edge.features),
            raw_tags={},
            oneway=edge.oneway,
            node_ids=edge.node_ids,
            highway=edge.highway,
        )
        for edge in edges
    ]
    return IMPLEMENTATIONS[name](nodes, algorithm_edges, start_node, goal_node, agent_name, **kwargs)
