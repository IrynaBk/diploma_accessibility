"""
Accessibility routing model (after Kaselouris et al., 2025, J. Urban Mobility 8, 100164)
======================================================================================

Works on a NetworkX graph (OSMnx MultiDiGraph or any nx.Graph/DiGraph/MultiGraph).

Pipeline
--------
1. classify_edges(G, cfg)  -> writes boolean edge attributes  acc_<profile>
                              (edge in A  <=>  passes every criterion of the profile)
2. route(G, s, t, profile) -> tiered result:
      Model 1  shortest FULLY accessible path            (Eqs. 1-8)
      Model 2  shortest path with AT MOST ONE inaccessible edge (Eqs. 1-7, 10)
      Fallback unrestricted shortest path (assistance needed)
3. solve_ilp(...)          -> exact ILP of the paper's formulation (PuLP/CBC or Gurobi),
                              only for cross-checking; Dijkstra below is the fast path.

Model 2 is solved exactly with a layered graph: state = (node, #inaccessible edges used so far).
An accessible edge keeps the layer, an inaccessible edge moves layer k -> k+1 (allowed only if
k+1 <= max_violations). Dijkstra on this state space is polynomial and returns the same optimum
as the ILP, so neither Gurobi nor k-shortest-path enumeration is needed.
"""
from __future__ import annotations

import heapq
import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

# --------------------------------------------------------------------------------------
# Profiles / configuration
# --------------------------------------------------------------------------------------
ABLE, MOBILITY, VISUAL = "able", "mobility", "visual"
PROFILES = (ABLE, MOBILITY, VISUAL)


@dataclass
class Thresholds:
    """Greek standard FEK 6213/2022 as used in the paper (Table 5)."""
    min_width_m: float = 1.5
    max_horizontal_slope_pct: float = 12.0
    max_cross_slope_pct: float = 2.5      # Table 1 says 2 %, Table 5 says 2.5 % -> your call
    max_kerb_slope_pct: float = 5.0


@dataclass
class Config:
    thresholds: Thresholds = field(default_factory=Thresholds)

    # Which criteria each profile must satisfy (paper Table 2 + Sec. 4.4).
    # The paper does not list the exact rule for "able"; footway presence + width + obstacles
    # is the closest reading. Edit freely.
    profile_criteria: Dict[str, List[str]] = field(default_factory=lambda: {
        ABLE:     ["footway", "width", "obstacle"],
        MOBILITY: ["footway", "step_free", "width", "obstacle",
                   "horizontal_slope", "cross_slope", "kerb"],
        VISUAL:   ["footway", "width", "obstacle", "tactile"],
    })

    # What to do when an attribute is missing: "fail" (conservative) or "pass".
    missing_policy: str = "fail"
    # Per-criterion override. Absence of an 'obstacle' tag is normally read as "no obstacle".
    missing_by_criterion: Dict[str, str] = field(default_factory=lambda: {"obstacle": "pass"})

    # Walking speed per profile (m/s). The paper does not state c_ij; screenshots imply ~1.4.
    speed_mps: Dict[str, float] = field(default_factory=lambda: {ABLE: 1.4, MOBILITY: 1.4, VISUAL: 1.4})
    length_attr: str = "length"
    acc_prefix: str = "acc_"

    # Attribute names to read (adapt to your graph).
    width_attrs: Tuple[str, ...] = ("width", "est_width", "footway:width")
    incline_attrs: Tuple[str, ...] = ("incline",)          # OSM string: "5%", "3°"
    grade_attrs: Tuple[str, ...] = ("grade_abs", "grade")  # osmnx.elevation: fraction (0.05 = 5 %)
    cross_slope_attrs: Tuple[str, ...] = ("cross_slope", "incline:across")
    kerb_slope_attr: str = "kerb_slope"                    # node attribute, % (optional)
    obstacle_attr: str = "obstacle"                        # edge attribute, bool-like (optional)

    def missing(self, criterion: str) -> str:
        return self.missing_by_criterion.get(criterion, self.missing_policy)


# --------------------------------------------------------------------------------------
# Parsing helpers for OSM-style values
# --------------------------------------------------------------------------------------
def _first(v):
    """OSMnx merges parallel ways into lists; take the worst-case handled by callers."""
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def parse_width_m(v) -> Optional[float]:
    """'1.5', '1,5 m', '150 cm', '4 ft' -> metres. Lists -> minimum."""
    if v is None or _is_nan(v):
        return None
    vals = []
    for x in _first(v):
        if x is None or _is_nan(x):
            continue
        if isinstance(x, (int, float)):
            vals.append(float(x))
            continue
        m = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*(m|cm|mm|ft|')?\s*$", str(x).lower())
        if not m:
            continue
        n = float(m.group(1).replace(",", "."))
        unit = m.group(2) or "m"
        vals.append(n * {"m": 1, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "'": 0.3048}[unit])
    return min(vals) if vals else None


def parse_slope_pct(v) -> Optional[float]:
    """'5%', '-5 %', '3°', '5' (assumed %), 'up'/'down' (-> None). Returns |slope| in %."""
    if v is None or _is_nan(v):
        return None
    vals = []
    for x in _first(v):
        if x is None or _is_nan(x):
            continue
        if isinstance(x, (int, float)):
            vals.append(abs(float(x)))
            continue
        m = re.match(r"^\s*([+-]?[0-9]+(?:[.,][0-9]+)?)\s*(%|°|deg)?\s*$", str(x).lower())
        if not m:
            continue
        n = float(m.group(1).replace(",", "."))
        vals.append(abs(math.tan(math.radians(n)) * 100) if m.group(2) in ("°", "deg") else abs(n))
    return max(vals) if vals else None


def _yes(x) -> Optional[bool]:
    if x is None or _is_nan(x):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"yes", "true", "1", "y"}:
        return True
    if s in {"no", "false", "0", "n", "none"}:
        return False
    return None


def _tri_and(vals: Iterable[Optional[bool]]) -> Optional[bool]:
    vals = list(vals)
    if any(v is False for v in vals):
        return False
    if any(v is None for v in vals) or not vals:
        return None
    return True


# --------------------------------------------------------------------------------------
# Criteria: each returns True (pass) / False (fail) / None (no data)
# signature: (G, u, v, d, cfg) -> Optional[bool]
# --------------------------------------------------------------------------------------
_PED_HIGHWAYS = {"footway", "pedestrian", "path", "crossing", "living_street", "corridor", "track"}


def _hw(d) -> List[str]:
    return [str(x) for x in _first(d.get("highway"))] if d.get("highway") is not None else []


def _is_crossing(d) -> bool:
    if "crossing" in _hw(d):
        return True
    return any(str(x) == "crossing" for x in _first(d.get("footway")))


def crit_footway(G, u, v, d, cfg):
    """Pedestrian has a dedicated footway (or a pedestrian-only way); walking on the carriageway fails."""
    hw = set(_hw(d))
    if not hw:
        return None
    if hw & (_PED_HIGHWAYS | {"steps"}):
        return True
    side = d.get("sidewalk")
    if side is None:
        return False if hw & {"residential", "service", "unclassified", "tertiary", "secondary",
                              "primary", "trunk", "road", "living_street"} else None
    return not (set(map(str, _first(side))) <= {"no", "none"})


def crit_step_free(G, u, v, d, cfg):
    return "steps" not in _hw(d)


def crit_width(G, u, v, d, cfg):
    for a in cfg.width_attrs:
        w = parse_width_m(d.get(a))
        if w is not None:
            return w >= cfg.thresholds.min_width_m
    return None


def crit_obstacle(G, u, v, d, cfg):
    b = _yes(d.get(cfg.obstacle_attr))
    return None if b is None else (not b)


def crit_horizontal_slope(G, u, v, d, cfg):
    s = None
    for a in cfg.incline_attrs:
        s = parse_slope_pct(d.get(a))
        if s is not None:
            break
    if s is None:
        for a in cfg.grade_attrs:
            g = d.get(a)
            if g is not None and not _is_nan(g):
                s = abs(float(g)) * 100
                break
    return None if s is None else s <= cfg.thresholds.max_horizontal_slope_pct


def crit_cross_slope(G, u, v, d, cfg):
    for a in cfg.cross_slope_attrs:
        s = parse_slope_pct(d.get(a))
        if s is not None:
            return s <= cfg.thresholds.max_cross_slope_pct
    return None


def crit_kerb(G, u, v, d, cfg):
    """b_v / s_v: only crossing edges are checked; both endpoints need a lowered/flush kerb."""
    if not _is_crossing(d):
        return True
    res = []
    for n in (u, v):
        nd = G.nodes[n]
        k = nd.get("kerb", d.get("kerb"))
        if k is None:
            res.append(None)
            continue
        ok = str(k).lower() in {"lowered", "flush"}
        ks = parse_slope_pct(nd.get(cfg.kerb_slope_attr))
        if ok and ks is not None:
            ok = ks <= cfg.thresholds.max_kerb_slope_pct
        res.append(ok)
    return _tri_and(res)


def crit_tactile(G, u, v, d, cfg):
    """p_k: tactile paving on the edge; for crossings, also read the endpoint nodes."""
    e = _yes(d.get("tactile_paving"))
    if e is not None:
        return e
    if _is_crossing(d):
        return _tri_and(_yes(G.nodes[n].get("tactile_paving")) for n in (u, v))
    return None


CRITERIA: Dict[str, Callable] = {
    "footway": crit_footway,
    "step_free": crit_step_free,
    "width": crit_width,
    "obstacle": crit_obstacle,
    "horizontal_slope": crit_horizontal_slope,
    "cross_slope": crit_cross_slope,
    "kerb": crit_kerb,
    "tactile": crit_tactile,
}


# --------------------------------------------------------------------------------------
# Graph helpers (uniform handling of Multi/simple graphs)
# --------------------------------------------------------------------------------------
def iter_edges(G) -> Iterable[Tuple[Any, Any, Any, dict]]:
    if G.is_multigraph():
        yield from G.edges(keys=True, data=True)
    else:
        for u, v, d in G.edges(data=True):
            yield u, v, None, d


def out_edges(G, u) -> Iterable[Tuple[Any, Any, dict]]:
    """yield (v, key, data) for edges leaving u (works for undirected graphs too)."""
    if G.is_multigraph():
        for v, kd in G._adj[u].items():
            for k, d in kd.items():
                yield v, k, d
    else:
        for v, d in G._adj[u].items():
            yield v, None, d


def edge_data(G, u, v, k):
    return G[u][v][k] if G.is_multigraph() else G[u][v]


# --------------------------------------------------------------------------------------
# Step 1 - classify edges into A (accessible) per profile
# --------------------------------------------------------------------------------------
def classify_edges(G, cfg: Optional[Config] = None,
                   custom_criteria: Optional[Dict[str, Callable]] = None) -> Config:
    """
    Writes on every edge:
        acc_<profile>        : bool   (edge in A for that profile)
        acc_<profile>_fail   : list of failing criterion names (for diagnostics)
        acc_<profile>_unknown: list of criteria that had no data (resolved via cfg.missing_policy)
    Pass `custom_criteria={'name': fn(G,u,v,d,cfg)->bool|None}` to add/override criteria, and
    reference them in cfg.profile_criteria.
    """
    cfg = cfg or Config()
    crits = {**CRITERIA, **(custom_criteria or {})}
    for u, v, k, d in iter_edges(G):
        cache: Dict[str, Optional[bool]] = {}
        for prof, names in cfg.profile_criteria.items():
            fails, unknown = [], []
            for c in names:
                if c not in cache:
                    cache[c] = crits[c](G, u, v, d, cfg)
                r = cache[c]
                if r is None:
                    unknown.append(c)
                    r = cfg.missing(c) == "pass"
                if not r:
                    fails.append(c)
            d[f"{cfg.acc_prefix}{prof}"] = not fails
            d[f"{cfg.acc_prefix}{prof}_fail"] = fails
            d[f"{cfg.acc_prefix}{prof}_unknown"] = unknown
    return cfg


def accessibility_report(G, cfg: Optional[Config] = None) -> Dict[str, Dict[str, Any]]:
    """Share of accessible edges per profile + failure/missing-data counts per criterion.
    Compare with the paper (Penteli): able 82 %, mobility 7.6 %, visual 9.2 %."""
    cfg = cfg or Config()
    n = 0
    rep = {p: {"accessible": 0, "fail": {}, "unknown": {}} for p in cfg.profile_criteria}
    for _, _, _, d in iter_edges(G):
        n += 1
        for p in cfg.profile_criteria:
            rep[p]["accessible"] += bool(d.get(f"{cfg.acc_prefix}{p}"))
            for c in d.get(f"{cfg.acc_prefix}{p}_fail", []):
                rep[p]["fail"][c] = rep[p]["fail"].get(c, 0) + 1
            for c in d.get(f"{cfg.acc_prefix}{p}_unknown", []):
                rep[p]["unknown"][c] = rep[p]["unknown"].get(c, 0) + 1
    for p in rep:
        rep[p]["n_edges"] = n
        rep[p]["share_accessible"] = rep[p]["accessible"] / n if n else float("nan")
    return rep


# --------------------------------------------------------------------------------------
# Step 2 - routing
# --------------------------------------------------------------------------------------
@dataclass
class Route:
    nodes: List[Any]
    edges: List[Tuple[Any, Any, Any]]      # (u, v, key)
    cost: float                            # travel time [s]
    length: float                          # [m]
    violations: int                        # number of inaccessible edges traversed
    violation_edges: List[Tuple[Any, Any, Any]]


@dataclass
class RoutingResult:
    profile: str
    # 'accessible'   : a fully accessible path exists (alternative may also be offered)
    # 'alternative'  : no accessible path, but one with <= 1 inaccessible edge exists
    # 'inaccessible' : only fully unrestricted path exists -> assistance needed
    # 'none'         : origin and destination not connected
    tier: str
    accessible: Optional[Route] = None     # Model 1
    alternative: Optional[Route] = None    # Model 2, only if strictly faster (or accessible missing)
    fallback: Optional[Route] = None       # unrestricted, only when tiers 1 and 2 are empty
    baseline: Optional[Route] = None       # unrestricted shortest (optional, for evaluation)

    @property
    def recommended(self) -> Optional[Route]:
        return self.accessible or self.alternative or self.fallback


def make_cost_fn(cfg: Config, profile: str) -> Callable[[dict], float]:
    speed = cfg.speed_mps[profile]

    def cost(d: dict) -> float:
        return float(d.get(cfg.length_attr, 0.0)) / speed
    return cost


def _uncertainty_penalty(d: dict, profile: str, cfg: Config) -> float:
    """Penalize missing or uncertain accessibility information so partially observed edges are not treated as fully safe."""
    unknown_key = f"{cfg.acc_prefix}{profile}_unknown"
    fail_key = f"{cfg.acc_prefix}{profile}_fail"
    unknowns = d.get(unknown_key, [])
    fails = d.get(fail_key, [])
    if not unknowns and not fails:
        return 0.0
    unknown_count = len(unknowns)
    fail_count = len(fails)
    critical = {"width", "horizontal_slope", "cross_slope", "kerb", "obstacle"}
    critical_unknown = sum(1 for name in unknowns if name in critical)
    critical_fail = sum(1 for name in fails if name in critical)
    return 0.5 * unknown_count + 2.0 * critical_unknown + 10.0 * fail_count + 20.0 * critical_fail


def make_robust_cost_fn(cfg: Config, profile: str, uncertainty_weight: float = 1.0) -> Callable[[dict], float]:
    base_cost = make_cost_fn(cfg, profile)

    def cost(d: dict) -> float:
        return base_cost(d) + uncertainty_weight * _uncertainty_penalty(d, profile, cfg)
    return cost


def shortest_path(G, source, target, is_acc: Callable[[Any, Any, Any, dict], bool],
                  cost: Callable[[dict], float], length_attr: str = "length",
                  max_violations: Optional[int] = 0) -> Optional[Route]:
    """
    Layered Dijkstra.
      max_violations = 0     -> Model 1 (only accessible edges)
      max_violations = 1     -> Model 2 (at most one inaccessible edge)
      max_violations = None  -> unrestricted shortest path
    Ties are broken in favour of fewer violations.
    """
    if source not in G or target not in G:
        return None
    if source == target:
        return Route([source], [], 0.0, 0.0, 0, [])

    cap = max_violations
    start = (source, 0)
    dist = {start: 0.0}
    prev: Dict[Tuple[Any, int], Tuple[Tuple[Any, int], Any]] = {}
    tie = itertools.count()
    pq = [(0.0, 0, next(tie), source)]
    done = set()

    goal = None
    while pq:
        dcur, used, _, u = heapq.heappop(pq)
        st = (u, used)
        if st in done:
            continue
        done.add(st)
        if u == target:
            goal = st
            break
        for v, k, d in out_edges(G, u):
            acc = is_acc(u, v, k, d)
            if cap is None:          # unrestricted: single layer
                nused = 0
            elif acc:                # accessible edge: stay in the same layer
                nused = used
            elif used + 1 <= cap:    # inaccessible edge: move up one layer
                nused = used + 1
            else:
                continue
            ns = (v, nused)
            if ns in done:
                continue
            nd = dcur + cost(d)
            if nd < dist.get(ns, math.inf):
                dist[ns] = nd
                prev[ns] = (st, (u, v, k))
                heapq.heappush(pq, (nd, nused, next(tie), v))

    if goal is None:
        return None

    edges, st = [], goal
    while st != start:
        st, e = prev[st][0], prev[st][1]
        edges.append(e)
    edges.reverse()
    nodes = [source] + [e[1] for e in edges]
    viol = [e for e in edges if not is_acc(e[0], e[1], e[2], edge_data(G, *e))]
    length = sum(float(edge_data(G, *e).get(length_attr, 0.0)) for e in edges)
    return Route(nodes, edges, dist[goal], length, len(viol), viol)


def route(G, source, target, profile: str, cfg: Optional[Config] = None,
          compute_baseline: bool = False, eps: float = 1e-9,
          uncertainty_weight: float = 0.0) -> RoutingResult:
    """
    Tiered routing exactly as in the paper when uncertainty_weight=0.0.
    With a positive uncertainty_weight, edge cost is increased by a visibility-aware
    penalty so partially observed edges are treated as less trustworthy than fully observed ones.
    """
    cfg = cfg or Config()
    key = f"{cfg.acc_prefix}{profile}"
    is_acc = lambda u, v, k, d: bool(d.get(key, False))
    cost = make_robust_cost_fn(cfg, profile, uncertainty_weight) if uncertainty_weight > 0 else make_cost_fn(cfg, profile)
    kw = dict(is_acc=is_acc, cost=cost, length_attr=cfg.length_attr)

    acc = shortest_path(G, source, target, max_violations=0, **kw)
    alt = shortest_path(G, source, target, max_violations=1, **kw)
    # alt is only interesting if it really uses an inaccessible edge and is strictly faster
    if alt is not None and (alt.violations == 0 or (acc is not None and alt.cost >= acc.cost - eps)):
        alt = None

    fallback = None
    if acc is None and alt is None:
        fallback = shortest_path(G, source, target, max_violations=None, **kw)

    baseline = None
    if compute_baseline:
        baseline = fallback or shortest_path(G, source, target, max_violations=None, **kw)

    if acc is not None:
        tier = "accessible"
    elif alt is not None:
        tier = "alternative"
    elif fallback is not None:
        tier = "inaccessible"
    else:
        tier = "none"
    return RoutingResult(profile, tier, acc, alt, fallback, baseline)


def route_uncertainty_aware(G, source, target, profile: str, cfg: Optional[Config] = None,
                           compute_baseline: bool = False, eps: float = 1e-9,
                           uncertainty_weight: float = 1.0) -> RoutingResult:
    """Robust variant used for the improved observability-aware benchmark; preserves the same tier contract while penalizing missing edge information."""
    return route(G, source, target, profile, cfg=cfg, compute_baseline=compute_baseline, eps=eps, uncertainty_weight=uncertainty_weight)


# --------------------------------------------------------------------------------------
# Exact ILP of the paper (Eqs. 1-9, with Eq. 7 summed over K as intended) - for validation
# --------------------------------------------------------------------------------------
def solve_ilp(G, source, target, profile: str, cfg: Optional[Config] = None,
              max_violations: Optional[int] = 0, solver: str = "PULP_CBC_CMD") -> Optional[Route]:
    """
    max_violations=0 -> Eqs. 1-8 ; =1 -> Eqs. 1-7,10 ; None -> Eq. 8 removed.
    solver: any PuLP solver name, e.g. 'PULP_CBC_CMD' (default) or 'GUROBI' if installed.
    """
    import pulp

    cfg = cfg or Config()
    key = f"{cfg.acc_prefix}{profile}"
    cost = make_cost_fn(cfg, profile)

    E = [(u, v, k, d) for u, v, k, d in iter_edges(G)]
    if not G.is_directed():   # undirected: create both directions
        E = E + [(v, u, k, d) for u, v, k, d in E if u != v]
    prob = pulp.LpProblem("acc_route", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(E))}
    prob += pulp.lpSum(cost(E[i][3]) * x[i] for i in x)                                    # Eq. 1

    out_, in_ = {}, {}
    for i, (u, v, _, _) in enumerate(E):
        out_.setdefault(u, []).append(i)
        in_.setdefault(v, []).append(i)
    for n in G.nodes:
        o, ii = out_.get(n, []), in_.get(n, [])
        bal = pulp.lpSum(x[i] for i in o) - pulp.lpSum(x[i] for i in ii)
        rhs = 1 if n == source else (-1 if n == target else 0)                              # Eqs. 2-4, 6, 7
        if source != target:
            prob += bal == rhs
        else:
            prob += bal == 0
        prob += pulp.lpSum(x[i] for i in o) <= 1                                            # Eq. 5
    if max_violations is not None:
        bad = [i for i, (_, _, _, d) in enumerate(E) if not d.get(key, False)]
        prob += pulp.lpSum(x[i] for i in bad) <= max_violations                             # Eq. 8 / 10

    status = prob.solve(pulp.getSolver(solver, msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None
    chosen = [i for i in x if x[i].value() > 0.5]
    nxt = {E[i][0]: i for i in chosen}
    nodes, edges, cur = [source], [], source
    while cur != target and cur in nxt:
        i = nxt.pop(cur)
        edges.append((E[i][0], E[i][1], E[i][2]))
        cur = E[i][1]
        nodes.append(cur)
    viol = [(E[i][0], E[i][1], E[i][2]) for i in chosen if not E[i][3].get(key, False)]
    tot = sum(cost(E[i][3]) for i in chosen)
    length = sum(float(E[i][3].get(cfg.length_attr, 0.0)) for i in chosen)
    return Route(nodes, edges, tot, length, len(viol), viol)


# --------------------------------------------------------------------------------------
# Evaluation helper for your dataset
# --------------------------------------------------------------------------------------
def evaluate_od_pairs(G, od_pairs: Sequence[Tuple[Any, Any]], profile: str,
                      cfg: Optional[Config] = None) -> List[Dict[str, Any]]:
    """One row per O-D pair: tier used, times, detour ratio vs. unrestricted baseline, #violations."""
    cfg = cfg or Config()
    rows = []
    for s, t in od_pairs:
        r = route(G, s, t, profile, cfg, compute_baseline=True)
        b = r.baseline
        rec = r.recommended
        rows.append({
            "origin": s, "destination": t, "tier": r.tier,
            "baseline_time_s": b.cost if b else None,
            "accessible_time_s": r.accessible.cost if r.accessible else None,
            "alternative_time_s": r.alternative.cost if r.alternative else None,
            "recommended_time_s": rec.cost if rec else None,
            "recommended_violations": rec.violations if rec else None,
            "detour_ratio": (rec.cost / b.cost) if (rec and b and b.cost > 0) else None,
        })
    return rows


# --------------------------------------------------------------------------------------
# Usage
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # With OSMnx (needs: pip install osmnx):
    #   import osmnx as ox
    #   ox.settings.useful_tags_way += ["width", "incline", "tactile_paving", "kerb", "sidewalk", "footway"]
    #   ox.settings.useful_tags_node += ["kerb", "tactile_paving"]
    #   G = ox.graph_from_place("Penteli, Greece", network_type="walk")   # or your own graph
    #   cfg = classify_edges(G)
    #   print(accessibility_report(G, cfg))
    #   res = route(G, orig_node, dest_node, MOBILITY, cfg)
    #   print(res.tier, res.recommended.cost, res.recommended.violation_edges)
    import random

    random.seed(1)
    G = nx.MultiDiGraph()
    N = 6
    for i in range(N):
        for j in range(N):
            for di, dj in ((0, 1), (1, 0)):
                a, b = (i, j), (i + di, j + dj)
                if b[0] < N and b[1] < N:
                    for u, v in ((a, b), (b, a)):
                        G.add_edge(u, v, length=random.uniform(40, 120), highway="footway",
                                   width=random.choice(["1.0", "1.6", "2.0"]),
                                   incline=random.choice(["2%", "4%", "15%"]),
                                   cross_slope="1.5%", tactile_paving=random.choice(["yes", "no"]))
    cfg = classify_edges(G)
    for p, r in accessibility_report(G, cfg).items():
        print(p, f"{r['share_accessible']:.1%}", r["fail"])
    res = route(G, (0, 0), (N - 1, N - 1), MOBILITY, cfg)
    print(res.tier, res.recommended and round(res.recommended.cost),
          res.alternative and round(res.alternative.cost))