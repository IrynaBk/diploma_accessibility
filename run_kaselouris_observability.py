from route_metrics import load_graph
from routing_experiments import apply_observability_mask
from routing_implementations import find_connected_route_pair, build_kaselouris_graph
import kaselouris
from kaselouris import Config, classify_edges, route

nodes, edges = load_graph('lviv_ua', 'all')
valid = {n for e in edges for n in (e.start, e.end)}
safe = {n: c for n, c in nodes.items() if n in valid}
pair = find_connected_route_pair(safe, edges, (49.840, 24.030), (49.850, 24.045))
print('PAIR', pair)
cfg = Config(missing_policy='pass')
for level in [0.0, 0.25, 0.5, 0.75, 1.0]:
    G = build_kaselouris_graph(safe, apply_observability_mask(edges, level, seed=42))
    class_cfg = classify_edges(
        G,
        cfg=cfg,
        custom_criteria={
            'footway': lambda G, u, v, d, cfg: True if (set(kaselouris._hw(d)) & (kaselouris._PED_HIGHWAYS | {'steps'})) or d.get('sidewalk') is not None or not set(kaselouris._hw(d)) else True
        },
    )
    res = route(G, pair[0], pair[1], 'mobility', class_cfg, compute_baseline=False)
    rec = res.recommended
    print(level, res.tier, None if rec is None else (round(rec.cost, 3), round(rec.length, 3), rec.violations, len(rec.edges)))