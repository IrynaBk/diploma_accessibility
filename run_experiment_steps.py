import csv

from route_metrics import load_graph
from routing_experiments import apply_observability_mask
from routing_implementations import find_connected_route_pair, build_kaselouris_graph
import kaselouris
from kaselouris import Config, classify_edges, route


ROUTE_POINTS = [
    ((49.840, 24.030), (49.850, 24.045)),
    ((49.830, 24.010), (49.845, 24.055)),
    ((49.815, 24.020), (49.852, 24.060)),
]


def save_results(rows, output_path='kaselouris_observability_results.csv'):
    fieldnames = [
        'pair_index',
        'start_point',
        'goal_point',
        'start_node',
        'goal_node',
        'observability_level',
        'tier',
        'cost',
        'length',
        'violations',
        'edge_count',
    ]
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved CSV results to {output_path}')


def run_observability_experiment():
    print('Loading graph...')
    nodes, edges = load_graph('lviv_ua', 'all')

    valid = {n for e in edges for n in (e.start, e.end)}
    safe = {n: c for n, c in nodes.items() if n in valid}
    print(f'Valid nodes: {len(safe)}')

    cfg = Config(missing_policy='pass')
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []

    for pair_index, (start_point, goal_point) in enumerate(ROUTE_POINTS, start=1):
        print(f'\n=== Pair {pair_index}: {start_point} -> {goal_point} ===')
        pair = find_connected_route_pair(safe, edges, start_point, goal_point)
        print(f'Chosen pair: {pair}')

        for level in levels:
            print(f'--- observability level: {level} ---')
            masked_edges = apply_observability_mask(edges, level, seed=42 + pair_index)
            G = build_kaselouris_graph(safe, masked_edges)

            class_cfg = classify_edges(
                G,
                cfg=cfg,
                custom_criteria={
                    'footway': lambda G, u, v, d, cfg: True if (set(kaselouris._hw(d)) & (kaselouris._PED_HIGHWAYS | {'steps'})) or d.get('sidewalk') is not None or not set(kaselouris._hw(d)) else True
                },
            )

            res = route(G, pair[0], pair[1], 'mobility', class_cfg, compute_baseline=False)
            rec = res.recommended

            if rec is None:
                result_row = {
                    'pair_index': pair_index,
                    'start_point': start_point,
                    'goal_point': goal_point,
                    'start_node': pair[0],
                    'goal_node': pair[1],
                    'observability_level': level,
                    'tier': res.tier,
                    'cost': '',
                    'length': '',
                    'violations': '',
                    'edge_count': '',
                }
                print('tier:', res.tier)
                print('result: no route found')
            else:
                result_row = {
                    'pair_index': pair_index,
                    'start_point': start_point,
                    'goal_point': goal_point,
                    'start_node': pair[0],
                    'goal_node': pair[1],
                    'observability_level': level,
                    'tier': res.tier,
                    'cost': round(rec.cost, 3),
                    'length': round(rec.length, 3),
                    'violations': rec.violations,
                    'edge_count': len(rec.edges),
                }
                print('tier:', res.tier)
                print('cost:', round(rec.cost, 3))
                print('length:', round(rec.length, 3))
                print('violations:', rec.violations)
                print('edge_count:', len(rec.edges))

            rows.append(result_row)

    save_results(rows)


if __name__ == '__main__':
    run_observability_experiment()
