"""
AccessibleRouteBench
--------------------

OSM -> normalized accessibility dataset -> MongoDB

Dependencies:
    pip install requests pymongo pandas numpy

Environment variables:
    MONGODB_URI=mongodb+srv://...
    MONGODB_DATABASE=accessible_routing

Example:
    python build_osm_dataset.py --city berlin
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()
# ============================================================
# CONFIG
# ============================================================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DATASET_VERSION = "0.1.0"

USER_AGENT = (
    "AccessibleRouteBench/0.1 "
    "(academic research)"
)

# Maximum number of objects we ideally want
# to insert for one area.
#
# This is NOT a hard API limit.
# It is only printed as a warning.
MAX_AREA_OBJECTS = 300_000


# Pedestrian-relevant highway classes.
# We intentionally retrieve all highway ways from Overpass
# and filter/classify here, rather than throwing information
# away during ingestion.
WALKABLE_HIGHWAYS = {
    "footway",
    "path",
    "pedestrian",
    "steps",
    "living_street",
    "residential",
    "service",
    "unclassified",
    "track",
    "cycleway",
    "tertiary",
    "tertiary_link",
    "secondary",
    "secondary_link",
    "primary",
    "primary_link",
}

# When configured areas overlap, the area with the higher priority owns the
# overlapping network. Historic areas are more specific than central areas.
AREA_PRIORITY = {
    "historic": 100,
    "hilly": 90,
    "waterfront": 80,
    "residential": 50,
    "suburban": 40,
    "dense_central": 10,
}

ACCESSIBILITY_TAGS = {
    "wheelchair",
    "surface",
    "smoothness",
    "width",
    "incline",
    "incline:across",
    "kerb",
    "sloped_curb",
    "ramp",
    "lit",
    "tactile_paving",
    "step_count",
    "step:height",
    "sidewalk",
    "sidewalk:left",
    "sidewalk:right",
    "sidewalk:surface",
    "sidewalk:left:surface",
    "sidewalk:right:surface",
    "sidewalk:width",
    "sidewalk:left:width",
    "sidewalk:right:width",
    "sidewalk:incline",
    "sidewalk:left:incline",
    "sidewalk:right:incline",
}


# ============================================================
# CITY CONFIG
# ============================================================

CITIES = {

    "berlin_de": {
        "name": "Berlin",
        "country": "Germany",
        "areas": [

            {
                "area_id": "central_dense",
                "type": "dense_central",
                "description": "Dense central urban area",
                "bbox": (52.495, 13.360, 52.535, 13.430),
            },

            {
                "area_id": "historic",
                "type": "historic",
                "description": "Historic urban environment",
                "bbox": (52.505, 13.390, 52.525, 13.425),
            },

            # {
            #     "area_id": "residential",
            #     "type": "residential",
            #     "description": "Dense residential neighborhood",
            #     "bbox": (52.525, 13.430, 52.565, 13.500),
            # },

            # {
            #     "area_id": "suburban",
            #     "type": "suburban",
            #     "description": "Lower-density suburban environment",
            #     "bbox": (52.480, 13.500, 52.540, 13.580),
            # },
        ],
    },


    "hamburg_de": {
        "name": "Hamburg",
        "country": "Germany",
        "areas": [

            {
                "area_id": "central_dense",
                "type": "dense_central",
                "description": "Dense central urban area",
                "bbox": (53.535, 9.960, 53.565, 10.030),
            },

            {
                "area_id": "historic",
                "type": "historic",
                "description": "Historic central area",
                "bbox": (53.540, 9.980, 53.560, 10.020),
            },

            # {
            #     "area_id": "residential",
            #     "type": "residential",
            #     "description": "Dense residential neighborhood",
            #     "bbox": (53.565, 9.950, 53.620, 10.020),
            # },

            # {
            #     "area_id": "suburban",
            #     "type": "suburban",
            #     "description": "Lower-density northern residential area",
            #     "bbox": (53.620, 9.950, 53.680, 10.050),
            # },
        ],
    },


    "tallinn_ee": {
        "name": "Tallinn",
        "country": "Estonia",
        "areas": [

            {
                "area_id": "historic",
                "type": "historic",
                "description": "Old Town historic environment",
                "bbox": (59.432, 24.720, 59.445, 24.755),
            },

            {
                "area_id": "central_dense",
                "type": "dense_central",
                "description": "Modern central urban area",
                "bbox": (59.425, 24.735, 59.450, 24.790),
            },

            # {
            #     "area_id": "residential",
            #     "type": "residential",
            #     "description": "Dense residential environment",
            #     "bbox": (59.410, 24.700, 59.450, 24.750),
            # },

            # {
            #     "area_id": "suburban",
            #     "type": "suburban",
            #     "description": "Peripheral residential area",
            #     "bbox": (59.400, 24.780, 59.470, 24.900),
            # },
        ],
    },


    "prague_cz": {
        "name": "Prague",
        "country": "Czechia",
        "areas": [

            {
                "area_id": "historic",
                "type": "historic",
                "description": "Historic central Prague",
                "bbox": (50.080, 14.400, 50.095, 14.435),
            },

            {
                "area_id": "dense_central",
                "type": "dense_central",
                "description": "Dense central urban area",
                "bbox": (50.060, 14.400, 50.105, 14.470),
            },

            # {
            #     "area_id": "hilly",
            #     "type": "hilly",
            #     "description": "Hilly urban environment",
            #     "bbox": (50.040, 14.370, 50.085, 14.430),
            # },

            # {
            #     "area_id": "residential",
            #     "type": "residential",
            #     "description": "Residential urban area",
            #     "bbox": (50.090, 14.450, 50.140, 14.520),
            # },
        ],
    },


    "lisbon_pt": {
        "name": "Lisbon",
        "country": "Portugal",
        "areas": [

            {
                "area_id": "historic",
                "type": "historic",
                "description": "Historic Alfama / Baixa environment",
                "bbox": (38.705, -9.145, 38.725, -9.125),
            },

            # {
            #     "area_id": "hilly",
            #     "type": "hilly",
            #     "description": "Steep historic/residential terrain",
            #     "bbox": (38.715, -9.170, 38.745, -9.135),
            # },

            {
                "area_id": "dense_central",
                "type": "dense_central",
                "description": "Dense central urban area",
                "bbox": (38.720, -9.170, 38.750, -9.120),
            },

            # {
            #     "area_id": "waterfront",
            #     "type": "waterfront",
            #     "description": "Relatively flat waterfront environment",
            #     "bbox": (38.690, -9.180, 38.720, -9.145),
            # },
        ],
    },
    "porto_pt": {
        "name": "Porto",
        "country": "Portugal",
        "areas": [

            {
                "area_id": "historic",
                "type": "historic",
                "description": "Historic Porto / Ribeira",
                "bbox": (41.135, -8.625, 41.150, -8.600),
            },

            # {
            #     "area_id": "hilly",
            #     "type": "hilly",
            #     "description": "Steep urban environment",
            #     "bbox": (41.145, -8.650, 41.175, -8.600),
            # },

            {
                "area_id": "dense_central",
                "type": "dense_central",
                "description": "Dense central urban area",
                "bbox": (41.145, -8.635, 41.175, -8.580),
            },

            # {
            #     "area_id": "waterfront",
            #     "type": "waterfront",
            #     "description": "Flatter waterfront area",
            #     "bbox": (41.125, -8.680, 41.160, -8.620),
            # },
        ],
    },


    "lviv_ua": {
        "name": "Lviv",
        "country": "Ukraine",
        "areas": [

            {
                "area_id": "historic",
                "type": "historic",
                "description": "Historic city center",
                "bbox": (49.835, 24.015, 49.850, 24.045),
            },

            {
                "area_id": "dense_central",
                "type": "dense_central",
                "description": "Dense central urban environment",
                "bbox": (49.825, 23.990, 49.865, 24.060),
            },

            # {
            #     "area_id": "residential",
            #     "type": "residential",
            #     "description": "Dense residential area",
            #     "bbox": (49.835, 23.950, 49.885, 24.020),
            # },

            # {
            #     "area_id": "suburban",
            #     "type": "suburban",
            #     "description": "Peripheral residential environment",
            #     "bbox": (49.800, 23.900, 49.850, 24.000),
            # },
        ],
    },


    "kyiv_ua": {
        "name": "Kyiv",
        "country": "Ukraine",
        "areas": [

            {
                "area_id": "historic",
                "type": "historic",
                "description": "Historic central Kyiv",
                "bbox": (50.445, 30.505, 50.465, 30.535),
            },

            {
                "area_id": "dense_central",
                "type": "dense_central",
                "description": "Dense central urban environment",
                "bbox": (50.430, 30.480, 50.470, 30.550),
            },

            # {
            #     "area_id": "residential",
            #     "type": "residential",
            #     "description": "Dense residential environment",
            #     "bbox": (50.400, 30.450, 50.450, 30.520),
            # },

            # {
            #     "area_id": "hilly",
            #     "type": "hilly",
            #     "description": "Hilly urban environment",
            #     "bbox": (50.430, 30.520, 50.490, 30.590),
            # },
        ],
    },
}


# ============================================================
# OVERPASS
# ============================================================

def build_overpass_query(
    bbox: Tuple[float, float, float, float]
) -> str:

    south, west, north, east = bbox

    query = f"""
    [out:json][timeout:300];

    (
      way["highway"](
        {south},
        {west},
        {north},
        {east}
      );
    );

    (
      ._;
      >;
    );

    out body;
    """

    return query


def fetch_overpass(
    bbox: Tuple[float, float, float, float],
    retries: int = 3,
) -> Dict[str, Any]:

    query = build_overpass_query(bbox)

    for attempt in range(retries):

        try:

            print(
                f"Overpass request: "
                f"attempt {attempt + 1}/{retries}"
            )

            response = requests.post(

                OVERPASS_URL,

                data={
                    "data": query
                },

                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },

                timeout=360,
            )

            response.raise_for_status()

            data = response.json()

            elements = data.get(
                "elements",
                []
            )

            print(
                f"Received "
                f"{len(elements):,} "
                f"OSM elements."
            )

            return data

        except Exception as exc:

            print(
                f"Overpass request failed: "
                f"{exc}"
            )

            if attempt == retries - 1:
                raise

            time.sleep(
                10 * (attempt + 1)
            )

    raise RuntimeError(
        "Overpass request failed."
    )


# ============================================================
# BASIC EXPLORATION
# ============================================================

def explore_raw_osm(
    data: Dict[str, Any]
) -> None:

    elements = data.get(
        "elements",
        []
    )

    type_counts = Counter(
        element.get("type")
        for element in elements
    )

    print("\n" + "=" * 70)
    print("RAW OSM STRUCTURE")
    print("=" * 70)

    print("\nElement counts:")

    for element_type, count in (
        type_counts.items()
    ):

        print(
            f"  {element_type:10s}: "
            f"{count:,}"
        )

    tag_key_counter = Counter()

    highway_counter = Counter()
    surface_counter = Counter()
    smoothness_counter = Counter()
    wheelchair_counter = Counter()

    incline_values = []
    width_values = []

    for element in elements:

        tags = element.get(
            "tags",
            {}
        )

        tag_key_counter.update(
            tags.keys()
        )

        if element.get("type") != "way":
            continue

        if "highway" in tags:
            highway_counter[
                tags["highway"]
            ] += 1

        if "surface" in tags:
            surface_counter[
                tags["surface"]
            ] += 1

        if "smoothness" in tags:
            smoothness_counter[
                tags["smoothness"]
            ] += 1

        if "wheelchair" in tags:
            wheelchair_counter[
                tags["wheelchair"]
            ] += 1

        if "incline" in tags:

            incline = parse_percent(
                tags["incline"]
            )

            if incline is not None:
                incline_values.append(
                    incline
                )

        if "width" in tags:

            width = parse_length_m(
                tags["width"]
            )

            if width is not None:
                width_values.append(
                    width
                )

    print(
        "\nMost common highway values:"
    )

    for value, count in (
        highway_counter.most_common(20)
    ):

        print(
            f"  {value:30s} "
            f"{count:,}"
        )

    print(
        "\nMost common surfaces:"
    )

    for value, count in (
        surface_counter.most_common(20)
    ):

        print(
            f"  {value:30s} "
            f"{count:,}"
        )

    print(
        "\nSmoothness:"
    )

    for value, count in (
        smoothness_counter.most_common(20)
    ):

        print(
            f"  {value:30s} "
            f"{count:,}"
        )

    print(
        "\nWheelchair:"
    )

    for value, count in (
        wheelchair_counter.most_common(20)
    ):

        print(
            f"  {value:30s} "
            f"{count:,}"
        )

    way_elements = [

        element

        for element in elements

        if (
            element.get("type") == "way"
            and
            "highway" in
            element.get("tags", {})
        )
    ]

    n = len(
        way_elements
    )

    print(
        "\nAccessibility tag coverage:"
    )

    for tag in sorted(
        ACCESSIBILITY_TAGS
    ):

        count = sum(

            tag in element.get(
                "tags",
                {}
            )

            for element in way_elements
        )

        percentage = (
            100 * count / n
            if n
            else 0
        )

        if count:

            print(
                f"  {tag:35s}"
                f"{count:8,d}"
                f" ({percentage:6.2f}%)"
            )

    if incline_values:

        print(
            "\nIncline statistics:"
        )

        print(
            pd.Series(
                incline_values
            ).describe()
        )

    if width_values:

        print(
            "\nWidth statistics:"
        )

        print(
            pd.Series(
                width_values
            ).describe()
        )

    print(
        "\nTop OSM tag keys:"
    )

    for key, count in (
        tag_key_counter.most_common(40)
    ):

        print(
            f"  {key:35s}"
            f" {count:,}"
        )


# ============================================================
# PARSING
# ============================================================

def parse_percent(
    value: Any
) -> Optional[float]:

    if value is None:
        return None

    value = str(
        value
    ).strip().lower()

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        value
    )

    if not match:
        return None

    try:
        return float(
            match.group()
        )

    except ValueError:
        return None


def parse_length_m(
    value: Any
) -> Optional[float]:

    if value is None:
        return None

    value = str(
        value
    ).strip().lower()

    match = re.search(
        r"\d+(?:\.\d+)?",
        value
    )

    if not match:
        return None

    number = float(
        match.group()
    )

    if "cm" in value:
        return number / 100

    if "mm" in value:
        return number / 1000

    return number


# ============================================================
# NORMALIZATION
# ============================================================

SURFACE_CLASSES = {

    "asphalt":
        "smooth_paved",

    "concrete":
        "smooth_paved",

    "paved":
        "smooth_paved",

    "paving_stones":
        "paved_stones",

    "sett":
        "paved_stones",

    "cobblestone":
        "cobblestone",

    "fine_gravel":
        "gravel",

    "gravel":
        "gravel",

    "ground":
        "unpaved",

    "dirt":
        "unpaved",

    "earth":
        "unpaved",

    "grass":
        "unpaved",

    "sand":
        "unpaved",

    "mud":
        "unpaved",

    "unpaved":
        "unpaved",
}


SMOOTHNESS_SCORE = {

    "excellent": 1.0,
    "good": 0.8,
    "intermediate": 0.6,

    "bad": 0.3,
    "very_bad": 0.1,

    "horrible": 0.0,
    "very_horrible": 0.0,

    "impassable": 0.0,
}


def normalize_surface(
    tags: Dict[str, Any]
) -> Optional[str]:

    value = tags.get(
        "surface"
    )

    if not value:
        return None

    value = str(
        value
    ).lower().strip()

    return SURFACE_CLASSES.get(
        value,
        "other"
    )


def normalize_wheelchair(
    tags: Dict[str, Any]
) -> Optional[str]:

    value = tags.get(
        "wheelchair"
    )

    if value is None:
        return None

    value = str(
        value
    ).lower().strip()

    if value in {
        "yes",
        "no",
        "limited",
        "designated",
    }:

        return value

    return "unknown"


def normalize_way(
    element: Dict[str, Any],
    node_lookup: Dict[
        int,
        Dict[str, Any]
    ],
) -> Optional[Dict[str, Any]]:

    if element.get("type") != "way":
        return None

    tags = element.get(
        "tags",
        {}
    )

    highway = tags.get(
        "highway"
    )

    if highway not in WALKABLE_HIGHWAYS:
        return None

    way_nodes = element.get(
        "nodes",
        []
    )

    if len(way_nodes) < 2:
        return None

    geometry = []

    valid_node_ids = []

    for node_id in way_nodes:

        node = node_lookup.get(
            node_id
        )

        if node is None:
            continue

        geometry.append(
            [
                node["lon"],
                node["lat"],
            ]
        )

        valid_node_ids.append(
            node_id
        )

    if len(geometry) < 2:
        return None

    incline = parse_percent(
        tags.get("incline")
    )

    width = parse_length_m(
        tags.get("width")
    )

    wheelchair = normalize_wheelchair(
        tags
    )

    smoothness = tags.get(
        "smoothness"
    )

    smoothness_score = (

        SMOOTHNESS_SCORE.get(
            str(
                smoothness
            ).lower()
        )

        if smoothness

        else None
    )

    surface = normalize_surface(
        tags
    )

    is_stairs = (
        highway == "steps"
    )

    return {

        "osm_way_id":
            element["id"],

        "highway":
            highway,

        "node_ids":
            valid_node_ids,

        "geometry":
            geometry,

        # Preserve original OSM information.
        "raw_tags":
            tags,

        "features": {

            "surface":
                surface,

            "smoothness":
                smoothness,

            "smoothness_score":
                smoothness_score,

            "width_m":
                width,

            "incline_percent":
                incline,

            "wheelchair":
                wheelchair,

            "stairs":
                is_stairs,

            "ramp":
                tags.get("ramp"),

            "kerb":
                tags.get("kerb"),

            "sloped_curb":
                tags.get(
                    "sloped_curb"
                ),

            "sidewalk":
                tags.get("sidewalk"),

            "lit":
                tags.get("lit"),

            "tactile_paving":
                tags.get(
                    "tactile_paving"
                ),
        },

        "data_quality": {

            "surface_known":
                surface is not None,

            "smoothness_known":
                smoothness is not None,

            "width_known":
                width is not None,

            "incline_known":
                incline is not None,

            "wheelchair_known":
                wheelchair is not None,

            "ramp_known":
                "ramp" in tags,

            "kerb_known":
                "kerb" in tags,
        },
    }


def bboxes_overlap(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> bool:

    return (
        max(first[0], second[0]) < min(first[2], second[2])
        and max(first[1], second[1]) < min(first[3], second[3])
    )


def bbox_intersection(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:

    intersection = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )

    return intersection if bboxes_overlap(intersection, intersection) else None


def edge_intersects_bbox(
    edge: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
) -> bool:
    """Return whether an edge's geometry can enter an exclusion bbox."""

    geometry = edge.get("geometry", [])
    if not geometry:
        return False

    edge_bbox = (
        min(point[1] for point in geometry),
        min(point[0] for point in geometry),
        max(point[1] for point in geometry),
        max(point[0] for point in geometry),
    )

    return bboxes_overlap(edge_bbox, bbox)


def get_exclusion_bboxes(
    city_config: Dict[str, Any],
    area: Dict[str, Any],
) -> List[Tuple[float, float, float, float]]:
    """Get overlapping areas that have priority over the current area."""

    current_priority = AREA_PRIORITY.get(area["type"], 0)
    exclusion_bboxes = []

    for other_area in city_config["areas"]:

        if other_area["area_id"] == area["area_id"]:
            continue

        other_priority = AREA_PRIORITY.get(other_area["type"], 0)

        if (
            other_priority > current_priority
            and bboxes_overlap(area["bbox"], other_area["bbox"])
        ):
            overlap = bbox_intersection(
                area["bbox"],
                other_area["bbox"],
            )
            if overlap is not None:
                exclusion_bboxes.append(overlap)

    return exclusion_bboxes


def report_area_overlaps(
    city_id: str,
    city_config: Dict[str, Any],
) -> None:

    areas = city_config["areas"]

    for index, first_area in enumerate(areas):

        for second_area in areas[index + 1:]:

            if not bboxes_overlap(
                first_area["bbox"],
                second_area["bbox"],
            ):
                continue

            first_priority = AREA_PRIORITY.get(first_area["type"], 0)
            second_priority = AREA_PRIORITY.get(second_area["type"], 0)

            if first_priority == second_priority:
                raise ValueError(
                    f"Overlapping areas have the same priority in {city_id}: "
                    f"{first_area['area_id']} and {second_area['area_id']}"
                )

            owner = (
                first_area
                if first_priority > second_priority
                else second_area
            )
            excluded = (
                second_area
                if owner is first_area
                else first_area
            )

            print(
                f"Area overlap: {owner['area_id']} owns the overlap; "
                f"{excluded['area_id']} will exclude it."
            )


# ============================================================
# MAPPING COMPLETENESS
# ============================================================

def calculate_mapping_completeness(
    edges: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not edges:
        return {}

    fields = [

        "surface",
        "smoothness",

        "width_m",
        "incline_percent",

        "wheelchair",

        "ramp",
        "kerb",
    ]

    result = {}

    for field in fields:

        known = 0

        for edge in edges:

            value = edge[
                "features"
            ].get(field)

            if value is not None:

                known += 1

        coverage = (
            known / len(edges)
        )

        result[field] = {

            "known":
                known,

            "total":
                len(edges),

            "coverage":
                coverage,
        }

    overall = np.mean([

        value["coverage"]

        for value in result.values()
    ])

    result["overall"] = float(
        overall
    )

    return result


# ============================================================
# AREA STATISTICS
# ============================================================
# ============================================================
# OBSERVABILITY
# ============================================================

# These are the features that an accessibility-aware routing
# agent may use when making routing decisions.
#
# IMPORTANT:
# Observability does NOT mean accessibility.
#
# It measures how much relevant information OSM provides
# for a particular way.
OBSERVABILITY_FEATURES = [
    "surface",
    "smoothness",
    "width_m",
    "incline_percent",
    "wheelchair",
    "stairs",
    "ramp",
    "kerb",
    "sloped_curb",
    "sidewalk",
    "lit",
    "tactile_paving",
]


def calculate_edge_observability(
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate how much accessibility-relevant information
    is available for an OSM way.

    A feature is considered observable when its value is
    explicitly available.

    False is a valid value and therefore counts as known.

    Examples:

        surface = "asphalt" -> known
        stairs = False      -> known
        width_m = None      -> unknown
        incline = None      -> unknown

    This is NOT an accessibility score.
    """

    feature_status = {}

    known_features = 0

    for feature in OBSERVABILITY_FEATURES:

        value = features.get(feature)

        known = value is not None

        feature_status[feature] = known

        if known:
            known_features += 1

    total_features = len(
        OBSERVABILITY_FEATURES
    )

    coverage = (
        known_features / total_features
        if total_features
        else 0.0
    )

    if coverage >= 0.67:
        level = "high"

    elif coverage >= 0.34:
        level = "medium"

    else:
        level = "low"

    return {
        "score": float(coverage),

        "level": level,

        "known_features": known_features,

        "total_features": total_features,

        "coverage": float(coverage),

        "features": feature_status,
    }

def calculate_city_statistics(
    edges: List[Dict[str, Any]]
) -> Dict[str, Any]:

    inclines = [
        abs(edge["features"]["incline_percent"])
        for edge in edges
        if edge["features"].get(
            "incline_percent"
        ) is not None
    ]

    widths = [
        edge["features"]["width_m"]
        for edge in edges
        if edge["features"].get(
            "width_m"
        ) is not None
    ]

    mapping = calculate_mapping_completeness(
        edges
    )

    observability_scores = []

    observability_levels = Counter()

    for edge in edges:

        observability = calculate_edge_observability(
            edge["features"]
        )

        score = observability["score"]

        observability_scores.append(score)

        observability_levels[
            observability["level"]
        ] += 1

    stats = {

        "num_edges": len(edges),

        "observability": {

            "mean_score":
                float(
                    np.mean(observability_scores)
                )
                if observability_scores
                else None,

            "median_score":
                float(
                    np.median(observability_scores)
                )
                if observability_scores
                else None,

            "low_fraction":
                float(
                    observability_levels["low"]
                    / len(edges)
                )
                if edges
                else None,

            "medium_fraction":
                float(
                    observability_levels["medium"]
                    / len(edges)
                )
                if edges
                else None,

            "high_fraction":
                float(
                    observability_levels["high"]
                    / len(edges)
                )
                if edges
                else None,
        },

        "mapping_completeness":
            mapping,

        "incline": {},

        "width": {},
    }

    if inclines:

        stats["incline"] = {

            "mean_abs":
                float(np.mean(inclines)),

            "median_abs":
                float(np.median(inclines)),

            "p90_abs":
                float(
                    np.percentile(
                        inclines,
                        90
                    )
                ),

            "max_abs":
                float(np.max(inclines)),
        }

    if widths:

        stats["width"] = {

            "mean":
                float(np.mean(widths)),

            "median":
                float(np.median(widths)),

            "p10":
                float(
                    np.percentile(
                        widths,
                        10
                    )
                ),
        }

    return stats


# ============================================================
# MONGODB
# ============================================================

def get_mongo():

    uri = os.environ.get(
        "MONGODB_URI"
    )

    database_name = os.environ.get(
        "MONGODB_DATABASE",
        "accessible_routing"
    )

    if not uri:

        raise RuntimeError(
            "Set MONGODB_URI environment variable."
        )

    client = MongoClient(
        uri
    )

    db = client[
        database_name
    ]

    return client, db


def insert_in_batches(
    collection,
    documents: List[Dict[str, Any]],
    batch_size: int = 1000,
):

    total = len(
        documents
    )

    for start in range(
        0,
        total,
        batch_size
    ):

        batch = documents[
            start:
            start + batch_size
        ]

        collection.insert_many(
            batch,
            ordered=False,
        )

        end = min(
            start + batch_size,
            total
        )

        print(
            f"Inserted "
            f"{end:,}/{total:,}"
        )


# ============================================================
# SAVE AREA
# ============================================================

def save_area(
    db,

    city_id: str,

    city_config:
        Dict[str, Any],

    area:
        Dict[str, Any],

    raw_data:
        Dict[str, Any],

    edges:
        List[Dict[str, Any]],

    node_lookup:
        Dict[int, Dict[str, Any]],
):

    timestamp = datetime.now(
        timezone.utc
    )

    area_stats = (
        calculate_city_statistics(
            edges
        )
    )

    # --------------------------------------------------------
    # AREA DOCUMENT
    # --------------------------------------------------------

    area_document = {

        "city_id":
            city_id,

        "name":
            city_config["name"],

        "country":
            city_config["country"],

        "area_id":
            area["area_id"],

        "area_type":
            area["type"],

        "terrain_type":
            area.get(
                "terrain",
                "unknown"
            ),

        "description":
            area.get(
                "description"
            ),

        "bbox": {

            "south":
                area["bbox"][0],

            "west":
                area["bbox"][1],

            "north":
                area["bbox"][2],

            "east":
                area["bbox"][3],
        },

        "dataset_version":
            DATASET_VERSION,

        "retrieved_at":
            timestamp,

        "statistics":
            area_stats,

        "osm_metadata": {

            "version":
                raw_data.get(
                    "version"
                ),

            "generator":
                raw_data.get(
                    "generator"
                ),

            "osm3s":
                raw_data.get(
                    "osm3s"
                ),
        },
    }

    # --------------------------------------------------------
    # Save area metadata
    #
    # Rebuild one area at a time so rerunning the importer does not
    # preserve stale or overlapping nodes and edges.
    # --------------------------------------------------------

    area_filter = {
        "city_id": city_id,
        "area_id": area["area_id"],
    }

    db.cities.delete_many(area_filter)
    db.nodes.delete_many(area_filter)
    db.edges.delete_many(area_filter)

    db.cities.insert_one(
        area_document
    )

    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    node_documents = []

    # Only save nodes that actually belong
    # to normalized pedestrian-relevant ways.
    used_node_ids = set()

    for edge in edges:

        used_node_ids.update(
            edge["node_ids"]
        )

    for node_id in used_node_ids:

        node = node_lookup.get(
            node_id
        )

        if node is None:
            continue

        document = {

            "city_id":
                city_id,

            "area_id":
                area["area_id"],

            "area_type":
                area["type"],

            "terrain_type":
                area.get(
                    "terrain",
                    "unknown"
                ),

            "dataset_version":
                DATASET_VERSION,

            "osm_node_id":
                node_id,

            "lat":
                node["lat"],

            "lon":
                node["lon"],

            # Preserve all node tags.
            "raw_tags":
                node.get(
                    "tags",
                    {}
                ),
        }

        node_documents.append(
            document
        )

    print(
        f"\nNodes to insert: "
        f"{len(node_documents):,}"
    )

    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    edge_documents = []

    for edge in edges:

        features = (
            edge["features"]
        )

        observability = calculate_edge_observability(
            features
        )


        document = {

            **edge,

            "city_id":
                city_id,

            "area_id":
                area["area_id"],

            "area_type":
                area["type"],

            "terrain_type":
                area.get(
                    "terrain",
                    "unknown"
                ),
            "observability":
                observability,
            "dataset_version":
                DATASET_VERSION,

            "retrieved_at":
                timestamp,
        }

        edge_documents.append(
            document
        )

    print(
        f"Edges to insert: "
        f"{len(edge_documents):,}"
    )

    total_objects = (
        len(node_documents)
        +
        len(edge_documents)
    )

    print(
        f"Total MongoDB objects: "
        f"{total_objects:,}"
    )

    if total_objects > MAX_AREA_OBJECTS:

        print(
            "\nWARNING:"
        )

        print(
            f"This area contains "
            f"{total_objects:,} objects, "
            f"which exceeds the recommended "
            f"target of "
            f"{MAX_AREA_OBJECTS:,}."
        )

        print(
            "Consider reducing the area bbox."
        )

    # --------------------------------------------------------
    # Insert
    # --------------------------------------------------------

    if node_documents:

        print(
            "\nInserting nodes..."
        )

        insert_in_batches(
            db.nodes,
            node_documents,
            batch_size=1000,
        )

    if edge_documents:

        print(
            "\nInserting edges..."
        )

        insert_in_batches(
            db.edges,
            edge_documents,
            batch_size=1000,
        )

    print(
        f"\nSaved area "
        f"{city_config['name']} / "
        f"{area['area_id']}: "
        f"{len(node_documents):,} nodes, "
        f"{len(edge_documents):,} edges."
    )


# ============================================================
# PROCESS AREA
# ============================================================

def process_area(
    city_id: str,
    area: Dict[str, Any],
):

    city_config = CITIES[
        city_id
    ]

    print("\n")
    print("=" * 80)

    print(
        f"PROCESSING: "
        f"{city_config['name']}, "
        f"{city_config['country']}"
    )

    print(
        f"AREA: "
        f"{area['area_id']}"
    )

    print(
        f"TYPE: "
        f"{area['type']}"
    )

    print(
        f"TERRAIN: "
        f"{area.get('terrain', 'unknown')}"
    )

    exclusion_bboxes = get_exclusion_bboxes(
        city_config,
        area,
    )

    if exclusion_bboxes:
        print(
            f"Excluding {len(exclusion_bboxes)} higher-priority "
            "overlapping area(s)."
        )

    print("=" * 80)

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    raw_data = fetch_overpass(
        area["bbox"]
    )

    # --------------------------------------------------------
    # Raw exploration
    # --------------------------------------------------------

    explore_raw_osm(
        raw_data
    )

    elements = raw_data.get(
        "elements",
        []
    )

    # --------------------------------------------------------
    # Node lookup
    # --------------------------------------------------------

    node_lookup = {}

    for element in elements:

        if element.get(
            "type"
        ) == "node":

            node_lookup[
                element["id"]
            ] = element

    print(
        f"\nRaw nodes: "
        f"{len(node_lookup):,}"
    )

    # --------------------------------------------------------
    # Normalize ways
    # --------------------------------------------------------

    edges = []

    for element in elements:

        edge = normalize_way(

            element,

            node_lookup,
        )

        if edge is not None:

            if any(
                edge_intersects_bbox(edge, exclusion_bbox)
                for exclusion_bbox in exclusion_bboxes
            ):
                continue

            edge["area_id"] = area["area_id"]

            edge["area_type"] = area["type"]

            edge["terrain_type"] = area.get(
                "terrain_type",
                "unknown"
            )

            edges.append(
                edge
            )

    print(
        "\nNormalized pedestrian-relevant "
        f"ways: {len(edges):,}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = (
        calculate_city_statistics(
            edges
        )
    )

    print(
        "\nAREA STATISTICS"
    )

    print(
        "=" * 80
    )

    print(
        json.dumps(
            stats,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # MongoDB
    # --------------------------------------------------------

    client, db = get_mongo()

    try:

        save_area(

            db=db,

            city_id=city_id,

            city_config=city_config,

            area=area,

            raw_data=raw_data,

            edges=edges,

            node_lookup=node_lookup,
        )

    finally:

        client.close()


# ============================================================
# PROCESS CITY
# ============================================================

def process_city(
    city_id: str,
    area_id: Optional[str] = None,
):

    city_config = CITIES[
        city_id
    ]

    report_area_overlaps(
        city_id,
        city_config,
    )

    areas = (
        city_config["areas"]
    )

    if area_id is not None:

        areas = [

            area

            for area in areas

            if area["area_id"]
            == area_id
        ]

        if not areas:

            available = [
                area["area_id"]
                for area
                in city_config["areas"]
            ]

            raise ValueError(
                f"Unknown area "
                f"'{area_id}' for "
                f"{city_id}. "
                f"Available: "
                f"{available}"
            )

    areas = sorted(
        areas,
        key=lambda area: AREA_PRIORITY.get(
            area["type"],
            0,
        ),
        reverse=True,
    )

    for i, area in enumerate(
        areas
    ):

        process_area(

            city_id=city_id,

            area=area,
        )

        # Be polite to Overpass
        # between area requests.

        if i < len(areas) - 1:

            print(
                "\nWaiting 5 seconds "
                "before next area..."
            )

            time.sleep(5)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build AccessibleRouteBench "
            "OSM dataset"
        )
    )

    parser.add_argument(

        "--city",

        choices=[
            *CITIES.keys(),
            "all",
        ],

        required=True,
    )

    parser.add_argument(

        "--area",

        default=None,

        help=(
            "Process only one area "
            "within the selected city."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Single city
    # --------------------------------------------------------

    if args.city != "all":

        process_city(

            city_id=args.city,

            area_id=args.area,
        )

        return

    # --------------------------------------------------------
    # All cities
    # --------------------------------------------------------

    if args.area is not None:

        raise ValueError(
            "--area cannot be used "
            "with --city all."
        )

    for city_index, city_id in enumerate(
        CITIES
    ):

        process_city(
            city_id
        )

        # Wait between cities.

        if city_index < len(CITIES) - 1:

            print(
                "\nWaiting 10 seconds "
                "before next city..."
            )

            time.sleep(10)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()