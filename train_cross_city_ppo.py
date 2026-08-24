"""Train one shared PPO policy per agent across multiple cities.

The resulting models can be evaluated on cities not used during training.
Training uses separate graph environments sequentially while retaining the same
policy, so the policy parameters are shared across all source cities.

Example:
    python train_cross_city_ppo.py \
        --train-cities berlin_de hamburg_de tallinn_ee \
        --timesteps-per-city 20000 \
        --output-dir models/cross_city
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from routing_experiments import AGENT_PROFILES, AccessibleRoutingEnv, load_graph


def train_agent(
    agent: str,
    city_ids: List[str],
    area_id: str,
    timesteps_per_city: int,
    output_dir: Path,
    seed: int,
    max_episode_steps: int,
) -> None:
    from sb3_contrib import MaskablePPO

    model = None
    for city_index, city_id in enumerate(city_ids):
        print(f"Loading {city_id} for {agent}...", flush=True)
        nodes, edges = load_graph(city_id, area_id)
        environment = AccessibleRoutingEnv(
            nodes,
            edges,
            agent,
            seed=seed + city_index,
            max_episode_steps=max_episode_steps,
        )
        if model is None:
            model = MaskablePPO(
                "MultiInputPolicy",
                environment,
                verbose=0,
                seed=seed,
            )
        else:
            model.set_env(environment)
        print(
            f"Training {agent} on {city_id}: "
            f"{timesteps_per_city:,} timesteps",
            flush=True,
        )
        model.learn(
            total_timesteps=timesteps_per_city,
            reset_num_timesteps=False,
        )

    if model is None:
        raise ValueError("At least one training city is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / agent
    model.save(str(output_path))
    print(f"Saved {output_path}.zip", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train shared PPO policies across multiple source cities."
    )
    parser.add_argument("--train-cities", nargs="+", required=True)
    parser.add_argument("--area", default="all")
    parser.add_argument("--timesteps-per-city", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=Path("models/cross_city"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument(
        "--agent",
        choices=[*AGENT_PROFILES, "all"],
        default="all",
    )
    args = parser.parse_args()
    if args.timesteps_per_city <= 0 or args.max_episode_steps < 10:
        raise ValueError("Timesteps must be positive and episode steps must be at least 10.")

    agents = AGENT_PROFILES if args.agent == "all" else {args.agent: AGENT_PROFILES[args.agent]}
    for agent in agents:
        train_agent(
            agent,
            args.train_cities,
            args.area,
            args.timesteps_per_city,
            args.output_dir,
            args.seed,
            args.max_episode_steps,
        )


if __name__ == "__main__":
    main()
