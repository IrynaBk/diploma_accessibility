"""Add FABDEM incline estimates to stored OSM edges.

The script keeps the original OSM incline unchanged. DEM results are written
under ``features.dem`` and can be used when OSM incline is missing.

Examples:
    python enrich_edges_with_dem.py --city lviv_ua --area all
    python enrich_edges_with_dem.py --city all --area all
    python enrich_edges_with_dem.py --city lisbon_pt --area dense_central --dry-run

Dependencies:
    pip install fabdem rasterio pymongo python-dotenv
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import rasterio
import fabdem
import requests
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


EARTH_RADIUS_M = 6_371_000.0
DEFAULT_CACHE_DIR = Path(".fabdem_cache")
DEFAULT_SAMPLE_SPACING_M = 30.0
DEFAULT_MAX_GRADE_PERCENT = 50.0


Point = Tuple[float, float]  # longitude, latitude
BBox = Tuple[int, int]  # integer latitude, integer longitude


def tile_id(point: Point) -> BBox:
    return math.floor(point[1]), math.floor(point[0])


def tile_name(tile: BBox) -> str:
    latitude, longitude = tile
    lat_prefix = "N" if latitude >= 0 else "S"
    lon_prefix = "E" if longitude >= 0 else "W"
    return (
        f"FABDEM_V1-2_{lat_prefix}{abs(latitude):02d}_00_"
        f"{lon_prefix}{abs(longitude):03d}_00"
    )


def tile_path(tile: BBox, cache_dir: Path) -> Path:
    name = tile_name(tile)
    return cache_dir / f"{name}.tif"


def download_tile(tile: BBox, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = tile_path(tile, cache_dir)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    name = tile_name(tile)
    print(f"Downloading DEM tile {name}...", flush=True)
    latitude, longitude = tile
    original_session = fabdem.Session

    class CorrectedFabdemSession(original_session):
        @staticmethod
        def _correct_url(url: str) -> str:
            # FABDEM's current GeoJSON index has a W000 typo for archives
            # whose eastern boundary is 0 degrees longitude.
            return url.replace("W000_FABDEM_V1-2.zip", "E000_FABDEM_V1-2.zip")

        def get(self, url, *args, **kwargs):
            return super().get(self._correct_url(url), *args, **kwargs)

        def head(self, url, *args, **kwargs):
            return super().head(self._correct_url(url), *args, **kwargs)

    try:
        fabdem.Session = CorrectedFabdemSession
        fabdem.download(
            (longitude, latitude, longitude + 1, latitude + 1),
            output_path=output_path,
            show_progress=False,
            cache=cache_dir,
        )
    except requests.exceptions.HTTPError as error:
        raise RuntimeError(
            "FABDEM download failed. Check the public FABDEM archive "
            "availability and network connection."
        ) from error
    finally:
        fabdem.Session = original_session
    return output_path


def haversine_m(first: Point, second: Point) -> float:
    first_lon, first_lat = map(math.radians, first)
    second_lon, second_lat = map(math.radians, second)
    delta_lat = second_lat - first_lat
    delta_lon = second_lon - first_lon
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(first_lat)
        * math.cos(second_lat)
        * math.sin(delta_lon / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.asin(math.sqrt(value))


def sample_dem(
    geometry: Sequence[Point],
    cache_dir: Path,
) -> Tuple[List[Optional[float]], Set[BBox]]:
    if len(geometry) < 2:
        return [], set()

    tiles = {tile_id(point) for point in geometry}
    datasets = {
        tile: rasterio.open(download_tile(tile, cache_dir))
        for tile in tiles
    }
    try:
        elevations: List[Optional[float]] = []
        used_tiles: Set[BBox] = set()
        for point in geometry:
            tile = tile_id(point)
            dataset = datasets[tile]
            value = next(dataset.sample([(point[0], point[1])]))[0]
            if dataset.nodata is not None and math.isclose(value, dataset.nodata):
                elevations.append(None)
                continue
            if not math.isfinite(float(value)):
                elevations.append(None)
                continue
            elevations.append(round(float(value), 3))
            used_tiles.add(tile)
        return elevations, used_tiles
    finally:
        for dataset in datasets.values():
            dataset.close()


def calculate_dem_metrics(
    geometry: Sequence[Point],
    elevations: Sequence[Optional[float]],
    sample_spacing_m: float,
    max_grade_percent: float,
) -> Dict[str, Any]:
    valid_segments = []
    outlier_segment_count = 0
    for first, second, first_elevation, second_elevation in zip(
        geometry,
        geometry[1:],
        elevations,
        elevations[1:],
    ):
        if first_elevation is None or second_elevation is None:
            continue
        horizontal_m = haversine_m(first, second)
        if horizontal_m <= 0:
            continue
        elevation_change_m = second_elevation - first_elevation
        grade_percent = elevation_change_m / horizontal_m * 100
        if abs(grade_percent) > max_grade_percent:
            outlier_segment_count += 1
            continue
        valid_segments.append(
            {
                "horizontal_m": horizontal_m,
                "elevation_change_m": elevation_change_m,
                "grade_percent": grade_percent,
            }
        )

    if not valid_segments:
        return {
            "available": False,
            "reason": "no_valid_elevation_segments",
            "sample_count": len(elevations),
            "outlier_segment_count": outlier_segment_count,
            "max_grade_percent": max_grade_percent,
        }

    grades = [segment["grade_percent"] for segment in valid_segments]
    absolute_grades = [abs(grade) for grade in grades]
    uphill_gain = sum(
        max(0.0, segment["elevation_change_m"])
        for segment in valid_segments
    )
    downhill_loss = sum(
        max(0.0, -segment["elevation_change_m"])
        for segment in valid_segments
    )
    horizontal_length = sum(
        segment["horizontal_m"] for segment in valid_segments
    )
    weighted_mean_abs_grade = sum(
        abs(segment["grade_percent"]) * segment["horizontal_m"]
        for segment in valid_segments
    ) / horizontal_length

    return {
        "available": True,
        "source": "fabdem_v1_2",
        "sample_count": len(elevations),
        "valid_segment_count": len(valid_segments),
        "start_elevation_m": elevations[0],
        "end_elevation_m": elevations[-1],
        "min_elevation_m": min(
            value for value in elevations if value is not None
        ),
        "max_elevation_m": max(
            value for value in elevations if value is not None
        ),
        "elevation_gain_m": round(uphill_gain, 3),
        "elevation_loss_m": round(downhill_loss, 3),
        "net_elevation_change_m": round(
            elevations[-1] - elevations[0], 3
        ) if elevations[0] is not None and elevations[-1] is not None else None,
        "mean_abs_incline_percent": round(weighted_mean_abs_grade, 3),
        "max_abs_incline_percent": round(max(absolute_grades), 3),
        "p90_abs_incline_percent": round(
            sorted(absolute_grades)[
                min(len(absolute_grades) - 1, math.ceil(0.9 * len(absolute_grades)) - 1)
            ],
            3,
        ),
        "dem_incline_percent": round(
            (elevations[-1] - elevations[0]) / horizontal_length * 100,
            3,
        ) if elevations[0] is not None and elevations[-1] is not None else None,
        "sample_spacing_m": sample_spacing_m,
        "outlier_segment_count": outlier_segment_count,
        "max_grade_percent": max_grade_percent,
    }


def resample_geometry(
    geometry: Sequence[Point],
    sample_spacing_m: float,
) -> List[Point]:
    if len(geometry) < 2:
        return list(geometry)

    samples = [geometry[0]]
    for first, second in zip(geometry, geometry[1:]):
        segment_length = haversine_m(first, second)
        steps = max(1, math.ceil(segment_length / sample_spacing_m))
        for step in range(1, steps + 1):
            fraction = step / steps
            samples.append(
                (
                    first[0] + (second[0] - first[0]) * fraction,
                    first[1] + (second[1] - first[1]) * fraction,
                )
            )
    return samples


def enrich_edge(
    document: Dict[str, Any],
    cache_dir: Path,
    sample_spacing_m: float,
    max_grade_percent: float,
) -> Optional[Dict[str, Any]]:
    geometry = [
        (float(point[0]), float(point[1]))
        for point in document.get("geometry", [])
        if len(point) >= 2
    ]
    if len(geometry) < 2:
        return None

    sampled_geometry = resample_geometry(geometry, sample_spacing_m)
    elevations, used_tiles = sample_dem(sampled_geometry, cache_dir)
    metrics = calculate_dem_metrics(
        sampled_geometry,
        elevations,
        sample_spacing_m,
        max_grade_percent,
    )
    metrics["tile_count"] = len(used_tiles)
    return metrics


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich stored OSM edges with FABDEM incline data."
    )
    parser.add_argument("--city", required=True)
    parser.add_argument("--area", default="all")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sample-spacing-m", type=float, default=DEFAULT_SAMPLE_SPACING_M)
    parser.add_argument(
        "--max-grade-percent",
        type=float,
        default=DEFAULT_MAX_GRADE_PERCENT,
        help="Reject DEM segments above this absolute grade (default: 50%%).",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_arguments()
    if args.sample_spacing_m <= 0 or args.max_grade_percent <= 0:
        raise ValueError("Sampling spacing and maximum grade must be greater than zero.")

    load_dotenv()
    uri = os.environ.get("MONGODB_URI")
    database_name = os.environ.get("MONGODB_DATABASE", "accessible_routing")
    if not uri:
        raise RuntimeError("Set MONGODB_URI in the environment or .env file.")

    client = MongoClient(uri, serverSelectionTimeoutMS=30_000)
    try:
        database = client[database_name]
        city_ids = (
            sorted(database.cities.distinct("city_id"))
            if args.city == "all"
            else [args.city]
        )
        grand_total = 0
        grand_updated = 0
        grand_unavailable = 0
        print(
            f"Starting FABDEM enrichment: city={args.city}, area={args.area}, "
            f"max_grade={args.max_grade_percent}%",
            flush=True,
        )
        edge_projection = {
            "_id": 1,
            "osm_way_id": 1,
            "geometry": 1,
            "city_id": 1,
            "area_id": 1,
        }
        for city_id in city_ids:
            area_filter: Dict[str, Any] = {"city_id": city_id}
            if args.area != "all":
                area_filter["area_id"] = args.area

            total = 0
            updated = 0
            unavailable = 0
            operations: List[UpdateOne] = []
            for document in database.edges.find(area_filter, edge_projection):
                total += 1
                metrics = enrich_edge(
                    document,
                    args.cache_dir,
                    args.sample_spacing_m,
                    args.max_grade_percent,
                )
                if metrics is None:
                    unavailable += 1
                    continue
                if metrics["available"]:
                    updated += 1
                else:
                    unavailable += 1
                operations.append(
                    UpdateOne(
                        {"_id": document["_id"]},
                        {"$set": {"features.dem": metrics}},
                    )
                )
                if len(operations) >= args.batch_size and not args.dry_run:
                    database.edges.bulk_write(operations, ordered=False)
                    operations.clear()
                if total % args.batch_size == 0:
                    print(
                        f"{city_id}: inspected {total:,} edges...",
                        flush=True,
                    )

            if operations and not args.dry_run:
                database.edges.bulk_write(operations, ordered=False)

            grand_total += total
            grand_updated += updated
            grand_unavailable += unavailable
            mode = "(dry run)" if args.dry_run else ""
            print(
                f"{city_id}: processed {total:,} edges {mode}: "
                f"updated={updated:,}, unavailable={unavailable:,}.",
                flush=True,
            )

        mode = "(dry run)" if args.dry_run else ""
        print(
            f"Total: processed {grand_total:,} edges {mode}: "
            f"updated={grand_updated:,}, unavailable={grand_unavailable:,}.",
            flush=True,
        )
        if args.dry_run:
            print("No MongoDB documents were changed.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
