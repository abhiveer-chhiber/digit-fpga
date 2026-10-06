"""Package the learning website for static hosting."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "build" / "site"

VIEWER_FILES = (
    "interface/learn.html",
    "interface/learn.css",
    "interface/learn.js",
    "interface/network_math.js",
)


def build():
    snapshot = ROOT / "site_assets/network-view.json"
    data = json.loads(snapshot.read_text())

    if data["schema"] != 1:
        raise ValueError("Unsupported model export schema")

    copies = [(ROOT / name, Path(name)) for name in VIEWER_FILES]
    copies.append((
        snapshot,
        Path("experiments/results/network-view.json"),
    ))

    sample_paths = set()
    for sample in data["samples"]:
        relative = Path(sample["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:2] != ("data", "raw")
            or relative.suffix != ".png"
        ):
            raise ValueError(f"Invalid drawing path: {relative}")
        sample_paths.add(relative)

    copies.extend((ROOT / path, path) for path in sorted(sample_paths))

    # Check every input before replacing the previous build.
    for source, _ in copies:
        if not source.is_file():
            raise FileNotFoundError(f"Missing website input: {source}")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    for source, relative in copies:
        destination = OUTPUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    (OUTPUT / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="0; url=interface/learn.html">
    <title>Digit / FPGA — Learning Explorer</title>
</head>
<body>
    <p><a href="interface/learn.html">Open the learning explorer</a></p>
</body>
</html>
""",
        encoding="utf-8",
    )
    (OUTPUT / ".nojekyll").touch()

    total_bytes = sum(
        path.stat().st_size
        for path in OUTPUT.rglob("*")
        if path.is_file()
    )

    print("Built website: build/site")
    print(f"Included {len(data['snapshots'])} recorded model checkpoints")
    print(f"Included {len(sample_paths)} original validation drawings")
    print(f"Total size: {total_bytes / 1_000_000:.1f} MB")


if __name__ == "__main__":
    build()
