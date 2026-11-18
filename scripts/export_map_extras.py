"""Write what carla_canvas does not send about a map: its signal heads and parked vehicles.

The OmniDreams backend's rasters draw every signal head and every parked vehicle
(a box): neither comes
through carla_canvas. Traffic lights are actors, but their head boxes are not
in the protocol; parked vehicles are level geometry (environment objects), not
actors at all. This writes both once per map, from a running CARLA server:

    python scripts/export_map_extras.py --port 2000 --map Town10HD_Opt --out Town10HD_Opt.extras.json

and the renderer reads it (``renderer.options.map_extras``). Boxes are in
CARLA's world frame (left-handed), extents are half sizes, rotations
(pitch, yaw, roll) in degrees:

    {"map": ..., "parked_vehicles": [{"label", "name", "location", "extent", "rotation_pyr_deg"}],
     "traffic_light_heads": [{"light_id", "location", "extent", "rotation_pyr_deg"}]}

Run with a Python that has the CARLA client (``carla`` or typesafe_carla).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import typesafe_carla.carla as carla
    _ = carla.CityObjectLabel.Car
except Exception:  # typesafe_carla without the label enum: the plain client
    import carla  # type: ignore

LABELS = ("Car", "Truck", "Bus", "Motorcycle", "Bicycle")


def _box(bb) -> dict:
    return {"location": [bb.location.x, bb.location.y, bb.location.z],
            "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
            "rotation_pyr_deg": [bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--map", default=None, help="load this map first (default: the one loaded)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    client = carla.Client(a.host, a.port)
    client.set_timeout(300.0)
    world = client.get_world()
    if a.map and not world.get_map().name.endswith(a.map):
        world = client.load_world(a.map)
    print("server", client.get_server_version(), "map", world.get_map().name, flush=True)

    actor_ids = {actor.id for actor in world.get_actors()}
    parked = []
    for label in LABELS:
        n = 0
        for o in world.get_environment_objects(getattr(carla.CityObjectLabel, label)):
            if o.id in actor_ids:
                continue
            parked.append({"label": label, "name": o.name, "id": int(o.id), **_box(o.bounding_box)})
            n += 1
        print(f"parked {label}: {n}", flush=True)

    heads = []
    for light in world.get_actors().filter("traffic.traffic_light"):
        for bb in light.get_light_boxes():
            heads.append({"light_id": int(light.id), **_box(bb)})
    print(f"traffic light boxes: {len(heads)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "map": world.get_map().name, "server": client.get_server_version(),
        "frame": "CARLA world (left-handed, x forward, y right, z up); extents are half sizes",
        "parked_vehicles": parked, "traffic_light_heads": heads}, indent=1))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
