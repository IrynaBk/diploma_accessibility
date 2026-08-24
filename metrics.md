1. Agents and their preferences

I would start with 3 representative low-mobility agents. This is enough to demonstrate personalization without making the project unnecessarily large.

Agent	Main characteristics	Preferences
A1 — Wheelchair user	Strict accessibility requirements	Very strong avoidance of stairs, steep inclines, narrow paths, bad surfaces
A2 — Limited-mobility pedestrian	Can tolerate somewhat worse conditions	Moderate incline/surface constraints; stairs strongly discouraged
A3 — Elderly / low-endurance pedestrian	Accessibility important, but distance and effort also matter	Strong penalties for stairs/incline; prefers shorter and smoother routes
A1 — Wheelchair
{
    "name": "wheelchair",

    "max_incline_percent": 5,
    "preferred_incline_percent": 3,

    "min_width_m": 1.2,

    "avoid_stairs": True,

    "surface_penalties": {
        "smooth_paved": 0.0,
        "paved_stones": 0.5,
        "cobblestone": 2.0,
        "gravel": 3.0,
        "unpaved": 5.0
    },

    "weights": {
        "distance": 1.0,
        "incline": 3.0,
        "surface": 3.0,
        "width": 3.0,
        "stairs": 10.0,
        "unknown": 2.0
    }
}
A2 — Limited mobility
{
    "name": "limited_mobility",

    "max_incline_percent": 8,
    "preferred_incline_percent": 5,

    "min_width_m": 1.0,

    "avoid_stairs": True,

    "surface_penalties": {
        "smooth_paved": 0.0,
        "paved_stones": 0.3,
        "cobblestone": 1.0,
        "gravel": 2.0,
        "unpaved": 3.0
    },

    "weights": {
        "distance": 1.0,
        "incline": 2.0,
        "surface": 2.0,
        "width": 2.0,
        "stairs": 7.0,
        "unknown": 1.5
    }
}
A3 — Low endurance / elderly pedestrian
{
    "name": "low_endurance",

    "max_incline_percent": 10,
    "preferred_incline_percent": 5,

    "min_width_m": 0.9,

    "avoid_stairs": False,

    "surface_penalties": {
        "smooth_paved": 0.0,
        "paved_stones": 0.2,
        "cobblestone": 0.8,
        "gravel": 1.5,
        "unpaved": 2.0
    },

    "weights": {
        "distance": 2.0,
        "incline": 2.5,
        "surface": 1.5,
        "width": 1.0,
        "stairs": 4.0,
        "unknown": 1.0
    }
}

These should be treated as experimental agent profiles, not medical definitions. Their purpose is to create different routing preferences.

2. Dataset / environment metrics

These describe the cities and OSM data, not algorithm performance.

A. OSM mapping / observability
Overall observability

Percentage of accessibility-relevant attributes that are explicitly available:

O=
total accessibility attributes
known accessibility attributes
	​


Report:

mean
median
low/medium/high fraction
Feature-level coverage

For every attribute:

surface
smoothness
width
incline
wheelchair
kerb
ramp
sidewalk
tactile_paving

calculate:

Coverage(f)=
#edges
#edges with known f
	​


This is particularly important because you can say, for example:

Width information is available for only X% of edges, while surface information is available for Y%.

B. Urban environment metrics

For each city/area:

Network size
number of nodes
number of pedestrian-relevant edges
total network length
Road-type distribution

Percentage of:

footway
path
pedestrian
residential
service
primary
etc.
Surface distribution

Percentage of:

smooth paved
paving stones
cobblestone
gravel
unpaved
unknown
Incline
mean absolute incline
median
P90
maximum
fraction above 5%
fraction above 8%
fraction above 12%
Width
mean
median
P10
fraction below 1.2 m
fraction below 1.5 m
Stairs
number of stair edges
percentage of pedestrian network containing stairs

These give you the environmental diversity of the dataset.

3. Route-level metrics

These are probably the most important metrics for comparing A*, Dynamic A* and PPO.

1. Route length
L=
e∈route
∑
	​

length(e)

Lower is generally better.

2. Travel cost

Your agent-specific cost:

C(route)=
e
∑
	​

C(e,agent)

This is useful because a wheelchair user's optimal route and an elderly person's optimal route may legitimately be different.

3. Accessibility violations

Count how many times the route violates the agent's constraints.

Examples:

excessive incline
insufficient width
stairs
unacceptable surface

You can report:

ViolationRate=
route edges
violating edges
	​


This should be agent-specific.

4. Critical violations

Some violations should be treated differently.

For example, for the wheelchair agent:

stairs

could be a critical violation.

Report:

number of critical violations
whether route contains any critical violation

A route with one staircase may technically be short, but practically unusable.