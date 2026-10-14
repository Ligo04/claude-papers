#!/usr/bin/env python3
"""Toy, dependency-free explanation of the paper's representation.

This is not the official model and does not render photorealistic GI.  It makes
three structural ideas executable: point-cloud scene inputs, an NNE-style
local pre-aggregation, and a fixed-K local vector relation for query decoding.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]
Feature = tuple[float, ...]


@dataclass(frozen=True)
class ScenePoint:
    position: Vec3
    normal: Vec3
    albedo: Vec3
    emissive: float


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vec3) -> float:
    return math.sqrt(max(dot(a, a), 1e-12))


def unit(a: Vec3) -> Vec3:
    length = norm(a)
    return (a[0] / length, a[1] / length, a[2] / length)


def distance2(a: Vec3, b: Vec3) -> float:
    delta = sub(a, b)
    return dot(delta, delta)


def point_feature(point: ScenePoint) -> Feature:
    """Raw attributes that play the role of F(p, n, c, e) in the paper."""
    return point.albedo + point.normal + (point.emissive,)


def add_features(features: Iterable[Feature]) -> Feature:
    values = list(features)
    if not values:
        return (0.0,) * 7
    width = len(values[0])
    return tuple(sum(row[index] for row in values) / len(values) for index in range(width))


def farthest_point_sampling(points: Sequence[ScenePoint], count: int) -> list[ScenePoint]:
    """Small deterministic FPS implementation for the NNE teaching example."""
    if count >= len(points):
        return list(points)
    selected = [points[0]]
    nearest = [distance2(point.position, selected[0].position) for point in points]
    while len(selected) < count:
        index = max(range(len(points)), key=nearest.__getitem__)
        selected.append(points[index])
        for point_index, point in enumerate(points):
            nearest[point_index] = min(
                nearest[point_index], distance2(point.position, points[index].position)
            )
    return selected


def knn(points: Sequence[ScenePoint], center: Vec3, count: int) -> list[ScenePoint]:
    return sorted(points, key=lambda point: distance2(point.position, center))[:count]


def nearest_neighbor_embedding(
    scene: Sequence[ScenePoint], center_count: int, neighborhood: int
) -> list[dict[str, object]]:
    """NNE-like stage: FPS centers summarize local raw point halos."""
    embeddings: list[dict[str, object]] = []
    for center in farthest_point_sampling(scene, center_count):
        neighbors = knn(scene, center.position, neighborhood)
        relative = [sub(point.position, center.position) for point in neighbors]
        relative_feature = tuple(
            sum(vector[axis] for vector in relative) / len(relative) for axis in range(3)
        )
        embedding = add_features(point_feature(point) for point in neighbors)
        embeddings.append(
            {
                "position": center.position,
                "feature": embedding + relative_feature,
                "neighbor_count": len(neighbors),
            }
        )
    return embeddings


def vector_cross_attention(
    query: ScenePoint, neighbors: Sequence[dict[str, object]]
) -> tuple[float, Feature]:
    """Channel-wise subtraction relation with relative geometry.

    The paper learns linear projections and a vector-valued relation.  The toy
    version keeps the same shape while using deterministic projections, so the
    computation can be inspected without PyTorch.
    """
    query_feature = point_feature(query)
    contributions: list[tuple[float, Feature]] = []
    for item in neighbors:
        position = tuple(item["position"])  # type: ignore[arg-type]
        feature = tuple(item["feature"])  # type: ignore[arg-type]
        relative = sub(query.position, position)
        relation = tuple(
            math.tanh(query_feature[index] - feature[index % len(feature)] + relative[index % 3])
            for index in range(len(query_feature))
        )
        # A smooth scalar only controls the toy aggregation; the real model
        # retains channel-wise relation values throughout its attention block.
        weight = 1.0 / (1.0 + norm(relative))
        contributions.append((weight, relation))
    normalizer = sum(weight for weight, _ in contributions) or 1.0
    aggregate = tuple(
        sum(weight * relation[index] for weight, relation in contributions) / normalizer
        for index in range(len(query_feature))
    )
    # A positive normal-light alignment is a transparent toy irradiance proxy.
    light_direction = unit((0.2, 0.4, 1.0))
    direct = max(0.0, dot(query.normal, light_direction)) * (0.25 + query.emissive)
    return direct, aggregate


def make_room_scene(width: int, height: int) -> list[ScenePoint]:
    scene: list[ScenePoint] = []
    for row in range(height):
        for column in range(width):
            x = column / max(width - 1, 1)
            z = row / max(height - 1, 1)
            albedo = (0.55 + 0.2 * x, 0.42 + 0.15 * z, 0.35 + 0.2 * (1 - x))
            scene.append(ScenePoint((x, 0.0, z), (0.0, 1.0, 0.0), albedo, 0.0))
    for row in range(height):
        for column in range(width // 3):
            x = column / max(width // 3 - 1, 1)
            z = row / max(height - 1, 1)
            scene.append(ScenePoint((x, 0.95, z), (0.0, -1.0, 0.0), (0.3, 0.35, 0.4), 0.0))
    # A small area-light strip is represented by the same point interface.
    for column in range(max(4, width // 3)):
        x = 0.25 + 0.5 * column / max(width // 3 - 1, 1)
        scene.append(ScenePoint((x, 0.9, 0.48), (0.0, -1.0, 0.0), (1.0, 0.85, 0.65), 4.0))
    return scene


def run_demo(scene_points: int, query_count: int, neighbor_count: int) -> dict[str, object]:
    scene = make_room_scene(max(8, int(math.sqrt(scene_points))), max(8, int(math.sqrt(scene_points))))
    scene = scene[:scene_points]
    embeddings = nearest_neighbor_embedding(
        scene, center_count=max(4, len(scene) // 4), neighborhood=min(8, len(scene))
    )
    queries = [scene[index % len(scene)] for index in range(query_count)]
    decoded = []
    for query in queries:
        neighbors = sorted(
            embeddings,
            key=lambda item: distance2(query.position, tuple(item["position"])),  # type: ignore[arg-type]
        )[:neighbor_count]
        irradiance, relation = vector_cross_attention(query, neighbors)
        decoded.append({"query": query.position, "toy_irradiance": irradiance, "relation": relation})

    m, n, c, k = 20_000, 512 * 512, 256, 32
    paper_metrics = {"ours_mse": 0.048, "ours_ssim": 0.912, "encoder_ms": 208, "decoder_ms_512": 368}
    return {
        "teaching_demo": True,
        "warning": "Toy structural computation; no official weights or photorealistic rendering.",
        "scene": {"input_points": len(scene), "embedding_points": len(embeddings), "query_count": query_count},
        "first_embedding": {"position": embeddings[0]["position"], "feature_width": len(embeddings[0]["feature"])},
        "decoded_queries": decoded[:3],
        "complexity_proxy": {
            "paper_scene_points_M": m,
            "paper_query_points_N": n,
            "paper_encoder_linear_CM": c * m,
            "paper_decoder_local_KN": k * n,
            "renderformer_encoder_quadratic_M2": m * m,
            "renderformer_decoder_MN_plus_N2": m * n + n * n,
        },
        "paper_metrics": paper_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-points", type=int, default=512)
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("demo_result.json"))
    args = parser.parse_args()
    result = run_demo(args.scene_points, args.queries, args.neighbors)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
