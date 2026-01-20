import ai2thor.controller
import json
import pprint
import sys
import os

"""
AI2-THOR scene graph extractor
Usage:
    python3 extract_scene_graph.py [scene_name]
Saves scene metadata JSON to `scene_graph_<scene>.json` and prints a short summary.
"""

scene = "FloorPlan425"
output_path = f"scene_graph_{scene}.json"

controller = ai2thor.controller.Controller(
    scene=scene,
    gridSize=0.25,
    width=300,
    height=300,
    rotateStepDegrees=45,
    # Disable snapToGrid to avoid conflicts when using non-90-degree rotate steps
    snapToGrid=False,
    visibilityDistance=5,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
)

print(f"Initializing scene: {scene}")
event = controller.step(action="Initialize", scene=scene)
metadata = controller.last_event.metadata

print("\nTop-level metadata keys:")
for k in sorted(metadata.keys()):
    print(f" - {k}")

# Save full metadata to disk
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\nFull metadata saved to: {os.path.abspath(output_path)}")

# Print summary about objects
objects = metadata.get("objects", [])
print(f"\nNumber of objects in metadata['objects']: {len(objects)}")

# Collect object types and counts
type_counts = {}
for o in objects:
    t = o.get("objectType") or o.get("name") or "Unknown"
    type_counts[t] = type_counts.get(t, 0) + 1

print("\nObject type counts (top 20):")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:20]:
    print(f" - {t}: {c}")

# Print detailed keys for the first few objects
print("\nSample object metadata fields (first 5 objects):")
for i, o in enumerate(objects[:5]):
    print(f"\nObject #{i+1} id={o.get('objectId')} name={o.get('name')}")
    keys = sorted(list(o.keys()))
    print("  fields: ", keys)
    # print some common fields if present
    for field in ["objectId", "name", "objectType", "position", "rotation", "boundingBox", "visible", "pickupable", "openable", "receptacle", "receptacleObjectIds", "parentReceptacles"]:
        if field in o:
            print(f"    {field}: {pprint.pformat(o[field])}")

# Optional: pretty-print entire metadata (commented to avoid huge terminal output)
# pprint.pprint(metadata)

controller.stop()
print("\nDone.")
