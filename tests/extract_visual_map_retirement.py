"""Compile production lifecycle methods without the ROS/PCL front end.

The harness supplies small owning point/feature/map types; the deletion,
rejection, and sweep method bodies come verbatim from the production sources.
This tests lifetime behavior, not compilation of the complete ROS node.
"""

import argparse
from pathlib import Path


def method(source: Path, signature: str) -> str:
    text = source.read_text(encoding="utf-8")
    start = text.index(signature + "(")
    # These out-of-class definitions use unindented top-level closing braces.
    end = text.index("\n}\n", start) + 3
    line = text.count("\n", 0, start) + 1
    return f'#line {line} "{source.as_posix()}"\n' + text[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    vio = args.source_root / "src/vio_multi_cam.cpp"
    point = args.source_root / "src/visual_point_multi_cam.cpp"
    definitions = [method(vio, "void VIOManager::" + name) for name in (
        "queueReferenceRetirement", "queuePointRetirement",
        "flushVisualMapRetirements", "queueStaleVisualMapPoints",
        "recordManagedReferenceRejection",
    )]
    definitions += [method(point, "void VisualPoint::deleteFeatureRef"),
                    method(point, "bool VisualPoint::hasUsableReference")]
    args.output.write_text("\n".join(definitions), encoding="utf-8")


if __name__ == "__main__":
    main()
