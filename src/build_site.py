"""Package the project website for static hosting."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "build" / "site"

VIEWER_FILES = (
    "interface/learn.html",
    "interface/navigation.js",
    "interface/project.html",
    "interface/overview.html",
    "interface/methods.html",
    "interface/results.html",
    "interface/learn.css",
    "interface/learn.js",
    "interface/network_math.js",
    "interface/drawing_input.js",
)



def fixed_point_rows(report):
    if report["schema"] != 1 or report["accumulator_bits"] != 32:
        raise ValueError("Unsupported fixed-point report")
    expected = [(16, 10), (12, 6), (10, 4), (8, 2)]
    formats = report["formats"]
    if [(r["bits"], r["fractional_bits"]) for r in formats] != expected:
        raise ValueError("Unexpected fixed-point formats")

    rows = []
    for result in formats:
        train, val = result["train"], result["validation"]
        if train["samples"] != 80 or val["samples"] != 20:
            raise ValueError("Unexpected fixed-point sample counts")
        saturations = (
            train["total_saturation_events"] + val["total_saturation_events"]
        )
        rows.append(
            "<tr>"
            f"<th scope='row'>{result['bits']} / {result['fractional_bits']}</th>"
            f"<td>{int(train['fixed_correct'])} / 80</td>"
            f"<td>{int(val['fixed_correct'])} / 20</td>"
            f"<td>{int(val['agreement_with_float'])} / 20</td>"
            f"<td>{float(val['maximum_logit_error']):.6f}</td>"
            f"<td>{int(saturations)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


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

    report_path = ROOT / "docs/fixed-point-v1.json"
    report = json.loads(report_path.read_text())
    precision_rows = fixed_point_rows(report)
    results_source = (ROOT / "interface/results.html").read_text()
    if results_source.count("<!-- FIXED_POINT_ROWS -->") != 1:
        raise ValueError("Expected one fixed-point table marker")
    results_html = results_source.replace(
        "<!-- FIXED_POINT_ROWS -->", precision_rows
    )
    copies.append((report_path, Path("docs/fixed-point-v1.json")))

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

    (OUTPUT / "interface/results.html").write_text(
        results_html, encoding="utf-8"
    )

    (OUTPUT / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="0; url=interface/learn.html">
    <title>Digit / FPGA — Project Display</title>
</head>
<body>
    <p><a href="interface/learn.html">Open the project display</a></p>
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
