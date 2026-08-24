"""Classical, dynamic, and PPO routing experiments.

This module is deliberately separate from route_metrics.py. It uses the same
GraphEdge and agent profiles, while adding dynamic availability and experiment
scenarios. Agent profiles are experimental preferences, not medical definitions.

Examples:
    python routing_experiments.py --city lviv_ua --area all \
        --start-lat 49.840 --start-lon 24.030 \
        --end-lat 49.850 --end-lon 24.045 --algorithm astar

    python routing_experiments.py --city lviv_ua --area all \
        --start-lat 49.840 --start-lon 24.030 \
        --end-lat 49.850 --end-lon 24.045 --algorithm dynamic_astar \
        --blocked-percent 0.05 --dynamic-block-step 10
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx

from route_metrics import (
    AGENT_PROFILES,
    GraphEdge,
    calculate_route_metrics,
    classify_edge,
    load_graph,
    nearest_node,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gives a useful error at use time
    gym = None
    spaces = None


@dataclass
class DynamicEdgeState:
    """Mutable episode state for temporary construction or closure events."""

    blocked_edges: Set[Any] = field(default_factory=set)
    reasons: Dict[Any, str] = field(default_factory=dict)

    def is_available(self, edge_id: Any) -> bool:
        return edge_id not in self.blocked_edges

    def block(self, edge_id: Any, reason: str = "temporary_blockage") -> None:
        self.blocked_edges.add(edge_id)
        self.reasons[edge_id] = reason

    def unblock(self, edge_id: Any) -> None:
        self.blocked_edges.discard(edge_id)
        self.reasons.pop(edge_id, None)


@dataclass
class SearchResult:
    nodes: List[int]
    edges: List[GraphEdge]
    cost: float
    expanded_nodes: Optional[int] = None


def make_adjacency(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    dynamic_state: Optional[DynamicEdgeState] = None,
) -> Dict[int, List[Tuple[int, GraphEdge]]]:
    adjacency: Dict[int, List[Tuple[int, GraphEdge]]] = {}
    for edge in edges:
        if dynamic_state is not None and not dynamic_state.is_available(edge.edge_id):
            continue
        node_ids = edge.raw_tags.get("_node_ids", [edge.start, edge.end])
        node_ids = [node_id for node_id in node_ids if node_id in nodes]
        for start, end in zip(node_ids, node_ids[1:]):
            segment = GraphEdge(
                edge_id=edge.edge_id,
                start=start,
                end=end,
                length_m=_segment_length(nodes, start, end),
                features=edge.features,
                raw_tags={**edge.raw_tags, "_segment_start": start, "_segment_end": end},
            )
            adjacency.setdefault(start, []).append((end, segment))
            if not is_oneway(edge):
                reverse_segment = GraphEdge(
                    edge_id=edge.edge_id,
                    start=end,
                    end=start,
                    length_m=segment.length_m,
                    features=edge.features,
                    raw_tags={
                        **segment.raw_tags,
                        "_reverse": True,
                    },
                )
                adjacency.setdefault(end, []).append((start, reverse_segment))
    return adjacency


def _segment_length(nodes: Dict[int, Tuple[float, float]], start: int, end: int) -> float:
    from route_metrics import haversine_m

    return haversine_m(nodes[start], nodes[end])


def is_oneway(edge: GraphEdge) -> bool:
    return str(edge.raw_tags.get("oneway", "")).lower() in {"yes", "true", "1"}


def heuristic(nodes: Dict[int, Tuple[float, float]], node: int, goal: int, distance_weight: float) -> float:
    from route_metrics import haversine_m

    return haversine_m(nodes[node], nodes[goal]) / 1000 * distance_weight


def build_networkx_graph(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    agent_name: str,
    dynamic_state: Optional[DynamicEdgeState] = None,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    profile = AGENT_PROFILES[agent_name]
    for edge in edges:
        if dynamic_state is not None and not dynamic_state.is_available(edge.edge_id):
            continue
        node_ids = edge.raw_tags.get("_node_ids", [edge.start, edge.end])
        node_ids = [node_id for node_id in node_ids if node_id in nodes]
        for start, end in zip(node_ids, node_ids[1:]):
            segment = GraphEdge(
                edge_id=edge.edge_id,
                start=start,
                end=end,
                length_m=_segment_length(nodes, start, end),
                features=edge.features,
                raw_tags={**edge.raw_tags, "_segment_start": start, "_segment_end": end},
            )
            weight = classify_edge(segment, profile)["cost"]
            _add_weighted_edge(graph, start, end, segment, weight)
            if not is_oneway(edge):
                reverse_segment = GraphEdge(
                    edge_id=edge.edge_id,
                    start=end,
                    end=start,
                    length_m=segment.length_m,
                    features=edge.features,
                    raw_tags={**segment.raw_tags, "_reverse": True},
                )
                _add_weighted_edge(graph, end, start, reverse_segment, weight)
    return graph


def _add_weighted_edge(
    graph: nx.DiGraph,
    start: int,
    end: int,
    edge: GraphEdge,
    weight: float,
) -> None:
    existing = graph.get_edge_data(start, end)
    if existing is None or weight < existing["weight"]:
        graph.add_edge(start, end, weight=weight, route_edge=edge)


def shortest_path(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
    agent_name: str,
    algorithm: str = "astar",
    dynamic_state: Optional[DynamicEdgeState] = None,
) -> SearchResult:
    if start_node == goal_node:
        return SearchResult([start_node], [], 0.0, 0)
    if algorithm not in {"dijkstra", "astar"}:
        raise ValueError(f"Unsupported search algorithm: {algorithm}")
    graph = build_networkx_graph(nodes, edges, agent_name, dynamic_state)
    try:
        if algorithm == "dijkstra":
            path_nodes = nx.dijkstra_path(graph, start_node, goal_node, weight="weight")
        else:
            path_nodes = nx.astar_path(
                graph,
                start_node,
                goal_node,
                heuristic=lambda current, goal: heuristic(
                    nodes,
                    current,
                    goal,
                    AGENT_PROFILES[agent_name]["weights"]["distance"],
                ),
                weight="weight",
            )
    except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
        raise ValueError(
            "No available route exists between the selected nodes."
        ) from error
    path_edges = [
        graph[path_nodes[index]][path_nodes[index + 1]]["route_edge"]
        for index in range(len(path_nodes) - 1)
    ]
    path_cost = sum(
        graph[path_nodes[index]][path_nodes[index + 1]]["weight"]
        for index in range(len(path_nodes) - 1)
    )
    return SearchResult(path_nodes, path_edges, path_cost)


def connected_start_goal(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    random_generator: random.Random,
) -> Tuple[int, int]:
    adjacency = make_adjacency(nodes, edges)
    node_ids = list(adjacency)
    for _ in range(100):
        start, goal = random_generator.sample(node_ids, 2)
        try:
            shortest_path(nodes, edges, start, goal, "low_endurance", "dijkstra")
            return start, goal
        except ValueError:
            continue
    raise ValueError("Could not generate a connected start/goal pair.")


def block_random_edges(
    edges: Sequence[GraphEdge],
    percentage: float,
    seed: int,
) -> DynamicEdgeState:
    if not 0 <= percentage <= 1:
        raise ValueError("Blocked percentage must be between 0 and 1.")
    edge_ids = list(dict.fromkeys(edge.edge_id for edge in edges))
    random_generator = random.Random(seed)
    count = int(len(edge_ids) * percentage)
    selected = random_generator.sample(edge_ids, count)
    state = DynamicEdgeState()
    for edge_id in selected:
        state.block(edge_id, "scenario_blockage")
    return state


def run_dynamic_astar(
    nodes: Dict[int, Tuple[float, float]],
    edges: Sequence[GraphEdge],
    start_node: int,
    goal_node: int,
    agent_name: str,
    blocked_edges: Optional[DynamicEdgeState] = None,
    dynamic_block_step: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    state = blocked_edges or DynamicEdgeState()
    current = start_node
    traversed: List[GraphEdge] = []
    replans = 0
    replanning_time = 0.0
    route_changes = 0
    random_generator = random.Random(seed)
    initial_route = shortest_path(nodes, edges, current, goal_node, agent_name, "astar", state)
    active_route = initial_route.edges
    while current != goal_node:
        next_index = next(
            (index for index, edge in enumerate(active_route) if edge.start == current),
            None,
        )
        if next_index is None:
            started = time.perf_counter()
            try:
                replanned = shortest_path(nodes, edges, current, goal_node, agent_name, "astar", state)
            except ValueError:
                return {
                    "success": False,
                    "edges": traversed,
                    "number_of_replans": replans + 1,
                    "replanning_time_s": replanning_time + (time.perf_counter() - started),
                    "route_changes": route_changes,
                    "blocked_edge_count": len(state.blocked_edges),
                }
            replanning_time += time.perf_counter() - started
            replans += 1
            route_changes += 1
            active_route = replanned.edges
            next_index = 0
        edge = active_route[next_index]
        if not state.is_available(edge.edge_id):
            started = time.perf_counter()
            try:
                replanned = shortest_path(nodes, edges, current, goal_node, agent_name, "astar", state)
            except ValueError:
                return {
                    "success": False,
                    "edges": traversed,
                    "number_of_replans": replans + 1,
                    "replanning_time_s": replanning_time + (time.perf_counter() - started),
                    "route_changes": route_changes,
                    "blocked_edge_count": len(state.blocked_edges),
                }
            replanning_time += time.perf_counter() - started
            replans += 1
            route_changes += 1
            active_route = replanned.edges
            continue
        traversed.append(edge)
        current = edge.end if edge.start == current else edge.start
        active_route = active_route[next_index + 1:]
        if dynamic_block_step is not None and len(traversed) == dynamic_block_step and active_route:
            candidate = random_generator.choice(active_route)
            state.block(candidate.edge_id, "dynamic_change")
    return {
        "success": True,
        "edges": traversed,
        "number_of_replans": replans,
        "replanning_time_s": replanning_time,
        "route_changes": route_changes,
        "blocked_edge_count": len(state.blocked_edges),
    }


class AccessibleRoutingEnv(gym.Env if gym is not None else object):
    """Gymnasium environment with padded neighbor observations and masks."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        nodes: Dict[int, Tuple[float, float]],
        edges: Sequence[GraphEdge],
        agent_name: str = "wheelchair",
        max_neighbors: int = 16,
        seed: int = 0,
        dynamic_schedule: Optional[Dict[int, Sequence[Any]]] = None,
        max_episode_steps: int = 150,
    ) -> None:
        if gym is None:
            raise ImportError("Install gymnasium to use AccessibleRoutingEnv.")
        self.nodes = nodes
        self.edges = list(edges)
        self.agent_name = agent_name
        self.max_neighbors = max_neighbors
        self.random_generator = random.Random(seed)
        self.dynamic_schedule = dynamic_schedule or {}
        self.max_episode_steps = max_episode_steps
        self.adjacency = make_adjacency(nodes, edges)
        self.dynamic_state = DynamicEdgeState()
        self.current_node = 0
        self.goal_node = 0
        self.steps = 0
        self.action_space = spaces.Discrete(max_neighbors)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
            "neighbors": spaces.Box(-np.inf, np.inf, (max_neighbors, 8), dtype=np.float32),
            "action_mask": spaces.MultiBinary(max_neighbors),
        })

    def _available_neighbors(self) -> List[Tuple[int, GraphEdge]]:
        return [
            (node, edge)
            for node, edge in self.adjacency.get(self.current_node, [])
            if self.dynamic_state.is_available(edge.edge_id)
        ][: self.max_neighbors]

    def _observation(self) -> Dict[str, np.ndarray]:
        from route_metrics import haversine_m

        neighbors = self._available_neighbors()
        goal = self.nodes[self.goal_node]
        current = self.nodes[self.current_node]
        rows = np.zeros((self.max_neighbors, 8), dtype=np.float32)
        mask = np.zeros(self.max_neighbors, dtype=np.int8)
        for index, (node, edge) in enumerate(neighbors):
            classified = classify_edge(edge, AGENT_PROFILES[self.agent_name])
            rows[index] = [
                edge.length_m / 1000,
                float(classified["incline_percent"] or 0),
                float(classified["width_m"] or 0),
                float(classified["stairs"]),
                float(classified["unknown_features"]),
                haversine_m(self.nodes[node], goal) / 1000,
                float(edge.features.get("dem", {}).get("available", False)),
                float(self.dynamic_state.is_available(edge.edge_id)),
            ]
            mask[index] = 1
        direction_lat = goal[0] - current[0]
        direction_lon = goal[1] - current[1]
        state = np.array([
            direction_lat,
            direction_lon,
            haversine_m(current, goal) / 1000,
            self.steps / 1000,
        ], dtype=np.float32)
        return {"state": state, "neighbors": rows, "action_mask": mask}

    def action_masks(self) -> np.ndarray:
        return self._observation()["action_mask"].astype(bool)

    def _sample_episode_nodes(self) -> Tuple[int, int]:
        """Create a connected, non-trivial pair within the training horizon."""

        candidate_nodes = list(self.adjacency)
        for _ in range(100):
            start = self.random_generator.choice(candidate_nodes)
            current = start
            visited = {start}
            walk_length = self.random_generator.randint(5, min(40, self.max_episode_steps - 1))
            for _ in range(walk_length):
                neighbors = self.adjacency.get(current, [])
                if not neighbors:
                    break
                current = self.random_generator.choice(neighbors)[0]
                visited.add(current)
            if current != start and len(visited) >= 5:
                return start, current
        return connected_start_goal(self.nodes, self.edges, self.random_generator)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.random_generator.seed(seed)
        options = options or {}
        if "start_node" in options and "goal_node" in options:
            self.current_node = int(options["start_node"])
            self.goal_node = int(options["goal_node"])
            if self.current_node not in self.nodes or self.goal_node not in self.nodes:
                raise ValueError("The requested PPO start or goal node is not in the graph.")
        else:
            self.current_node, self.goal_node = self._sample_episode_nodes()
        self.dynamic_state = DynamicEdgeState()
        self.steps = 0
        return self._observation(), {"current_node": self.current_node, "goal_node": self.goal_node}

    def step(self, action: int):
        previous_distance = self._distance_to_goal()
        self.steps += 1
        for edge_id in self.dynamic_schedule.get(self.steps, []):
            self.dynamic_state.block(edge_id, "dynamic_change")
        neighbors = self._available_neighbors()
        if action < 0 or action >= len(neighbors):
            return self._observation(), -5.0, False, self.steps >= self.max_episode_steps, {"invalid_action": True}
        next_node, edge = neighbors[action]
        classified = classify_edge(edge, AGENT_PROFILES[self.agent_name])
        self.current_node = next_node
        terminated = self.current_node == self.goal_node
        progress = previous_distance - self._distance_to_goal()
        reward = progress / 25.0 - classified["cost"] / 50.0
        if classified["critical_violations"]:
            reward -= 2.0
        if terminated:
            reward += 20.0
        return self._observation(), reward, terminated, self.steps >= self.max_episode_steps, {
            "edge_id": edge.edge_id,
            "dynamic_edge_available": self.dynamic_state.is_available(edge.edge_id),
            "distance_progress_m": progress,
        }

    def _distance_to_goal(self) -> float:
        from route_metrics import haversine_m

        return haversine_m(self.nodes[self.current_node], self.nodes[self.goal_node])


def train_ppo(
    env: AccessibleRoutingEnv,
    output_path: str,
    total_timesteps: int = 10_000,
) -> None:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as error:
        raise ImportError("Install sb3-contrib and stable-baselines3 to train PPO with action masking.") from error
    model = MaskablePPO("MultiInputPolicy", env, verbose=0, seed=0)
    model.learn(total_timesteps=total_timesteps)
    model.save(output_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dijkstra, A*, dynamic A*, or PPO routing experiments.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--area", required=True)
    parser.add_argument("--start-lat", required=True, type=float)
    parser.add_argument("--start-lon", required=True, type=float)
    parser.add_argument("--end-lat", required=True, type=float)
    parser.add_argument("--end-lon", required=True, type=float)
    parser.add_argument("--agent", choices=list(AGENT_PROFILES), default="wheelchair")
    parser.add_argument("--algorithm", choices=["dijkstra", "astar", "dynamic_astar", "ppo"], default="astar")
    parser.add_argument("--blocked-percent", type=float, default=0.0)
    parser.add_argument("--dynamic-block-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppo-model", default=None)
    parser.add_argument("--ppo-timesteps", type=int, default=10_000)
    parser.add_argument("--train-ppo", action="store_true")
    parser.add_argument("--model-output", default="models/accessible_routing_ppo")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    nodes, edges = load_graph(args.city, args.area)
    if args.train_ppo:
        environment = AccessibleRoutingEnv(nodes, edges, args.agent, seed=args.seed)
        train_ppo(environment, args.model_output, args.ppo_timesteps)
        print(json.dumps({
            "algorithm": "ppo",
            "trained": True,
            "model": args.model_output,
            "timesteps": args.ppo_timesteps,
        }, indent=2))
        return

    start = nearest_node(nodes, (args.start_lat, args.start_lon))
    goal = nearest_node(nodes, (args.end_lat, args.end_lon))
    blocked = block_random_edges(edges, args.blocked_percent, args.seed)
    if args.algorithm in {"dijkstra", "astar"}:
        result = shortest_path(nodes, edges, start, goal, args.agent, args.algorithm, blocked)
        metrics = calculate_route_metrics(result.edges, AGENT_PROFILES[args.agent])
        output = {
            "algorithm": args.algorithm,
            "agent": args.agent,
            "success": True,
            "metrics": metrics,
            "path_node_count": len(result.nodes),
            "blocked_edge_count": len(blocked.blocked_edges),
        }
    elif args.algorithm == "dynamic_astar":
        result = run_dynamic_astar(nodes, edges, start, goal, args.agent, blocked, args.dynamic_block_step, args.seed)
        metrics = calculate_route_metrics(result["edges"], AGENT_PROFILES[args.agent])
        output = {"algorithm": args.algorithm, "agent": args.agent, **{key: value for key, value in result.items() if key != "edges"}, "metrics": metrics}
    else:
        env = AccessibleRoutingEnv(nodes, edges, args.agent)
        if args.ppo_model is None:
            raise ValueError("--ppo-model is required for algorithm=ppo. Train a model with train_ppo() first.")
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(args.ppo_model, env=env)
        observation, info = env.reset(seed=args.seed)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not terminated and not truncated:
            action, _ = model.predict(
                observation,
                action_masks=env.action_masks(),
                deterministic=True,
            )
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
        output = {"algorithm": "ppo", "agent": args.agent, "success": terminated, "steps": steps, "total_reward": total_reward, "info": info}
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
