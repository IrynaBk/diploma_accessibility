"""
Export MongoDB area coverage as an interactive HTML map.

Usage:
    python export_coverage_map.py
    python export_coverage_map.py --city kyiv_ua --output kyiv_coverage.html

The script reads area documents from the ``cities`` collection. It does not
load nodes or edges, because each saved area document already contains its
bounding box and summary statistics.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient


LEAFLET_CSS_URL = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS_URL = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
AREA_COLORS = [
    "#0f766e",
    "#2563eb",
    "#c2410c",
    "#7c3aed",
    "#be123c",
    "#4d7c0f",
]
AREA_PRIORITY = {
    "historic": 100,
    "hilly": 90,
    "waterfront": 80,
    "residential": 50,
    "suburban": 40,
    "dense_central": 10,
}


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive HTML map from MongoDB area documents."
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Only export one city_id, for example kyiv_ua.",
    )
    parser.add_argument(
        "--output",
        default="coverage_map.html",
        help="HTML output path (default: coverage_map.html).",
    )
    return parser.parse_args()


def get_documents(city_id: Optional[str]) -> List[Dict[str, Any]]:
    load_dotenv()

    uri = os.environ.get("MONGODB_URI")
    database_name = os.environ.get("MONGODB_DATABASE", "accessible_routing")
    if not uri:
        raise RuntimeError("Set MONGODB_URI in the environment or .env file.")

    query = {"city_id": city_id} if city_id else {}
    client = MongoClient(uri, serverSelectionTimeoutMS=30_000)
    try:
        database = client[database_name]
        collection = database["cities"]
        documents = list(
            collection.find(
                query,
                {
                    "_id": 0,
                    "city_id": 1,
                    "name": 1,
                    "country": 1,
                    "area_id": 1,
                    "area_type": 1,
                    "terrain_type": 1,
                    "description": 1,
                    "bbox": 1,
                    "statistics": 1,
                    "dataset_version": 1,
                    "retrieved_at": 1,
                },
            )
        )

        edge_query = {"city_id": city_id} if city_id else {}
        dem_by_area: Dict[tuple[str, str], Dict[str, Any]] = {}
        for edge in database.edges.find(
            edge_query,
            {
                "city_id": 1,
                "area_id": 1,
                "features.dem": 1,
            },
        ):
            dem = edge.get("features", {}).get("dem", {})
            if not dem.get("available"):
                continue
            key = (edge.get("city_id", "unknown"), edge.get("area_id", "unknown"))
            area_dem = dem_by_area.setdefault(
                key,
                {"weighted_mean": 0.0, "weight": 0, "max": 0.0},
            )
            weight = int(dem.get("valid_segment_count", 1))
            area_dem["weighted_mean"] += float(
                dem.get("mean_abs_incline_percent", 0.0)
            ) * weight
            area_dem["weight"] += weight
            area_dem["max"] = max(
                area_dem["max"],
                float(dem.get("max_abs_incline_percent", 0.0)),
            )

        for document in documents:
            key = (document.get("city_id"), document.get("area_id"))
            area_dem = dem_by_area.get(key)
            if not area_dem or not area_dem["weight"]:
                continue
            statistics = document.setdefault("statistics", {})
            incline = statistics.setdefault("incline", {})
            incline.setdefault(
                "mean_abs",
                area_dem["weighted_mean"] / area_dem["weight"],
            )
            incline.setdefault("max_abs", area_dem["max"])
    finally:
        client.close()

    return documents


def normalize_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for document in documents:
        bbox = document.get("bbox", {})
        required_keys = ("south", "west", "north", "east")
        if not all(key in bbox for key in required_keys):
            print(
                "Skipping area without a complete bbox: "
                f"{document.get('city_id', 'unknown')}/"
                f"{document.get('area_id', 'unknown')}"
            )
            continue

        normalized.append(
            {
                "city_id": document.get("city_id", "unknown"),
                "name": document.get("name", document.get("city_id", "Unknown")),
                "country": document.get("country", ""),
                "area_id": document.get("area_id", "unknown"),
                "area_type": document.get("area_type", "unknown"),
                "terrain_type": document.get("terrain_type", "unknown"),
                "description": document.get("description", ""),
                "bbox": {key: float(bbox[key]) for key in required_keys},
                "statistics": document.get("statistics", {}),
                "dataset_version": document.get("dataset_version", ""),
                "retrieved_at": document.get("retrieved_at"),
            }
        )
    for area in normalized:
        area["exclusion_bboxes"] = []
        area_priority = AREA_PRIORITY.get(area["area_type"], 0)

        for other_area in normalized:
            if (
                other_area["city_id"] == area["city_id"]
                and other_area["area_id"] != area["area_id"]
                and AREA_PRIORITY.get(other_area["area_type"], 0) > area_priority
                and bboxes_overlap(
                    area["bbox"],
                    other_area["bbox"],
                )
            ):
                area["exclusion_bboxes"].append(
                    bbox_intersection(
                        area["bbox"],
                        other_area["bbox"],
                    )
                )

    return normalized


def bboxes_overlap(
    first: Dict[str, float],
    second: Dict[str, float],
    ) -> bool:
    return (
        max(first["south"], second["south"])
        < min(first["north"], second["north"])
        and max(first["west"], second["west"])
        < min(first["east"], second["east"])
    )


def bbox_intersection(
    first: Dict[str, float],
    second: Dict[str, float],
) -> Dict[str, float]:
    return {
        "south": max(first["south"], second["south"]),
        "west": max(first["west"], second["west"]),
        "north": min(first["north"], second["north"]),
        "east": min(first["east"], second["east"]),
    }


def json_for_script(value: Any) -> str:
    serialized = json.dumps(value, default=str, ensure_ascii=True)
    return serialized.replace("</", "<\\/")


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_html(areas: List[Dict[str, Any]]) -> str:
    map_data = []
    for index, area in enumerate(areas):
        statistics = area.get("statistics", {})
        observability = statistics.get("observability", {})
        map_data.append(
            {
                **area,
                "color": AREA_COLORS[index % len(AREA_COLORS)],
                "edge_count": statistics.get("num_edges", 0),
                "observability_score": observability.get("mean_score"),
            }
        )

    title = "Accessible routing coverage"
    if areas:
        title = f"Accessible routing coverage: {len(areas)} area(s)"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{LEAFLET_CSS_URL}">
  <style>
    :root {{ color-scheme: light; font-family: system-ui, -apple-system, sans-serif; }}
    body {{ margin: 0; color: #17202a; background: #eef2f3; }}
    header {{ padding: 18px 24px; background: #12343b; color: #f7f5ef; }}
    h1 {{ margin: 0 0 4px; font-size: 1.35rem; }}
    .subtitle {{ margin: 0; color: #c6d9d5; font-size: .9rem; }}
    #map {{ height: min(72vh, 720px); min-height: 440px; background: #dce8e7; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 10px 24px; padding: 14px 24px; background: #fff; border-bottom: 1px solid #d5dddd; }}
    .summary strong {{ color: #12343b; }}
    .table-wrap {{ overflow-x: auto; padding: 20px 24px 28px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; font-size: .9rem; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e1e7e6; white-space: nowrap; }}
    th {{ color: #526466; font-size: .76rem; letter-spacing: .04em; text-transform: uppercase; }}
    .swatch {{ display: inline-block; width: 11px; height: 11px; margin-right: 7px; border-radius: 50%; vertical-align: -1px; }}
    .empty {{ padding: 48px 24px; text-align: center; }}
    @media (max-width: 640px) {{ header, .summary, .table-wrap {{ padding-left: 14px; padding-right: 14px; }} #map {{ min-height: 380px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="subtitle">Bounding boxes from the MongoDB <code>cities</code> collection</p>
  </header>
  <div id="summary" class="summary"></div>
  <div id="map"></div>
  <section class="table-wrap">
    <table>
      <thead><tr><th>Area</th><th>City</th><th>Type</th><th>Edges</th><th>Mean observability</th><th>Dataset</th></tr></thead>
      <tbody id="area-table"></tbody>
    </table>
  </section>
  <script src="{LEAFLET_JS_URL}"></script>
  <script>
    const areas = {json_for_script(map_data)};
    const map = L.map('map', {{ zoomControl: true }});
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);
    const bounds = [];
    const table = document.getElementById('area-table');
    const cityAreas = new Map();

    function safe(value) {{
      const element = document.createElement('div');
      element.textContent = value ?? 'n/a';
      return element.innerHTML;
    }}

    areas.forEach((area) => {{
      const bbox = area.bbox;
      const rectangleBounds = [[bbox.south, bbox.west], [bbox.north, bbox.east]];
            if (!cityAreas.has(area.city_id)) {{
                cityAreas.set(area.city_id, {{ name: area.name, country: area.country, areas: [] }});
            }}
            cityAreas.get(area.city_id).areas.push(area);
      const outerRing = [
        [bbox.south, bbox.west], [bbox.south, bbox.east],
        [bbox.north, bbox.east], [bbox.north, bbox.west],
        [bbox.south, bbox.west]
      ];
      const holes = area.exclusion_bboxes.map((excluded) => [
        [excluded.south, excluded.west], [excluded.south, excluded.east],
        [excluded.north, excluded.east], [excluded.north, excluded.west],
        [excluded.south, excluded.west]
      ]);
      bounds.push(...rectangleBounds);
      const coverageLayer = holes.length
        ? L.polygon([outerRing, ...holes], {{ color: area.color, weight: 2, fillOpacity: 0.18 }})
        : L.rectangle(rectangleBounds, {{ color: area.color, weight: 2, fillOpacity: 0.18 }});
      coverageLayer
        .bindPopup(
          `<strong>${{safe(area.name)}} / ${{safe(area.area_id)}}</strong>` +
          `<br>Type: ${{safe(area.area_type)}}` +
          `<br>Terrain: ${{safe(area.terrain_type)}}` +
          `<br>Normalized ways: ${{safe(area.edge_count)}}` +
          `<br>Mean observability: ${{area.observability_score == null ? 'n/a' : (area.observability_score * 100).toFixed(1) + '%'}}`
        ).addTo(map);

      const row = document.createElement('tr');
      row.innerHTML = `<td><span class="swatch" style="background:${{area.color}}"></span>${{safe(area.area_id)}}</td>` +
        `<td>${{safe(area.name)}}${{area.country ? ', ' + safe(area.country) : ''}}</td>` +
        `<td>${{safe(area.area_type)}}</td>` +
        `<td>${{safe(area.edge_count)}}</td>` +
        `<td>${{area.observability_score == null ? 'n/a' : (area.observability_score * 100).toFixed(1) + '%'}}</td>` +
        `<td>${{safe(area.dataset_version)}}</td>`;
      table.appendChild(row);
    }});

        cityAreas.forEach((city) => {{
            const cityBounds = [];
            const cityEdgeCount = city.areas.reduce(
                (total, area) => total + (Number(area.statistics?.num_edges) || 0), 0
            );
            const weightedMean = (getValue) => {{
                const values = city.areas
                    .map((area) => {{
                        const value = getValue(area);
                        const edgeCount = Number(area.statistics?.num_edges) || 0;
                        return value == null || edgeCount === 0 ? null : {{ value: Number(value), edgeCount }};
                    }})
                    .filter(Boolean);
                if (!values.length) return null;
                const totalWeight = values.reduce((total, item) => total + item.edgeCount, 0);
                return values.reduce((total, item) => total + item.value * item.edgeCount, 0) / totalWeight;
            }};
            const cityObservability = weightedMean(
                (area) => area.statistics?.observability?.mean_score
            );
            const cityIncline = weightedMean(
                (area) => area.statistics?.incline?.mean_abs
            );
            const cityWidth = weightedMean(
                (area) => area.statistics?.width?.mean
            );
            const cityMaxIncline = Math.max(...city.areas.map(
                (area) => Number(area.statistics?.incline?.max_abs) || 0
            ));
            const cityMapping = weightedMean(
                (area) => area.statistics?.mapping_completeness?.overall
            );
            city.areas.forEach((area) => {{
                cityBounds.push([area.bbox.south, area.bbox.west]);
                cityBounds.push([area.bbox.north, area.bbox.east]);
            }});
            const cityBoundsObject = L.latLngBounds(cityBounds);
            const cityCenter = cityBoundsObject.getCenter();
            L.marker(cityCenter)
                .bindTooltip(`${{safe(city.name)}}`, {{ direction: 'top', offset: [0, -8] }})
                .bindPopup(
                    `<strong>${{safe(city.name)}}${{city.country ? ', ' + safe(city.country) : ''}}</strong>` +
                    `<hr style="margin:8px 0">` +
                    `<div><strong>Coverage</strong></div>` +
                    `<div>Areas: ${{city.areas.length}}</div>` +
                    `<div>Normalized ways: ${{cityEdgeCount.toLocaleString()}}</div>` +
                    `<div>Mean observability: ${{cityObservability == null ? 'n/a' : (cityObservability * 100).toFixed(1) + '%'}}</div>` +
                    `<div>Mapping completeness: ${{cityMapping == null ? 'n/a' : (cityMapping * 100).toFixed(1) + '%'}}</div>` +
                    `<div>Mean absolute incline: ${{cityIncline == null ? 'n/a' : cityIncline.toFixed(2) + '%'}}</div>` +
                    `<div>Maximum absolute incline: ${{cityMaxIncline === 0 ? 'n/a' : cityMaxIncline.toFixed(2) + '%'}}</div>` +
                    `<div>Mean width: ${{cityWidth == null ? 'n/a' : cityWidth.toFixed(2) + ' m'}}</div>`
                )
                .on('click', () => map.fitBounds(cityBoundsObject, {{ padding: [36, 36], maxZoom: 14 }}))
                .addTo(map);
        }});

    if (bounds.length) {{
      map.fitBounds(bounds, {{ padding: [24, 24] }});
    }} else {{
      map.setView([50.08, 14.43], 4);
      document.getElementById('map').insertAdjacentHTML('afterend', '<div class="empty">No area documents with valid bounding boxes were found.</div>');
    }}

    const cityCount = new Set(areas.map((area) => area.city_id)).size;
    const edgeCount = areas.reduce((total, area) => total + (Number(area.edge_count) || 0), 0);
    document.getElementById('summary').innerHTML =
      `<span><strong>${{areas.length}}</strong> covered areas</span>` +
      `<span><strong>${{cityCount}}</strong> cities</span>` +
      `<span><strong>${{edgeCount.toLocaleString()}}</strong> normalized ways</span>`;
  </script>
</body>
</html>
"""


def main() -> None:
    args = get_arguments()
    documents = get_documents(args.city)
    areas = normalize_documents(documents)
    output_path = Path(args.output)
    output_path.write_text(render_html(areas), encoding="utf-8")
    print(f"Fetched {len(documents):,} area document(s).")
    print(f"Wrote {len(areas):,} mapped area(s) to {output_path.resolve()}")


if __name__ == "__main__":
    main()
