import csv
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw"
OUTPUT_DIR = ROOT / "data/processed"
SIZES = (8, 16)
INK_THRESHOLD = 32
CONTENT_FRACTION = 0.75


def load_ink(path):
    with Image.open(path) as source:
        if source.format != "PNG" or source.size != (128, 128):
            raise ValueError(f"Expected a 128x128 PNG: {path}")

        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("L")

    return ImageOps.invert(image)


def center_digit(ink):
    mask = ink.point(lambda value: 255 if value > INK_THRESHOLD else 0)
    bounds = mask.getbbox()

    if bounds is None:
        raise ValueError("Blank sample")

    left, top, right, bottom = bounds
    edge_touching = (
        left == 0
        or top == 0
        or right == ink.width
        or bottom == ink.height
    )

    cropped = ink.crop(bounds)
    longest_side = max(cropped.size)
    square_side = int(longest_side / CONTENT_FRACTION + 0.5)
    square = Image.new("L", (square_side, square_side), 0)

    x = (square_side - cropped.width) // 2
    y = (square_side - cropped.height) // 2
    square.paste(cropped, (x, y))

    return square, bounds, edge_touching


def main():
    files = sorted(RAW_DIR.glob("*/*/*.png"))
    if not files:
        raise SystemExit("No raw samples found")

    prepared = []
    for path in files:
        relative = path.relative_to(RAW_DIR)
        writer, label, filename = relative.parts

        if label not in {str(digit) for digit in range(10)}:
            raise ValueError(f"Invalid digit directory: {path}")

        ink = load_ink(path)
        try:
            centered, bounds, edge_touching = center_digit(ink)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error

        prepared.append(
            (relative, writer, label, filename, centered, bounds, edge_touching)
        )

    rows = []
    for relative, writer, label, filename, centered, bounds, edge in prepared:
        for size in SIZES:
            output = OUTPUT_DIR / f"{size}x{size}" / relative
            output.parent.mkdir(parents=True, exist_ok=True)

            processed = centered.resize(
                (size, size), resample=Image.Resampling.BOX
            )
            processed.save(output)

            rows.append({
                "raw_path": (RAW_DIR / relative).relative_to(ROOT).as_posix(),
                "processed_path": output.relative_to(ROOT).as_posix(),
                "writer": writer,
                "label": label,
                "sample": filename,
                "resolution": size,
                "edge_touching": int(edge),
                "left": bounds[0],
                "top": bounds[1],
                "right": bounds[2],
                "bottom": bounds[3],
            })

    manifest = OUTPUT_DIR / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    edge_count = sum(item[-1] for item in prepared)
    print(f"Processed {len(prepared)} originals")
    print(f"Created {len(rows)} images across resolutions {SIZES}")
    print(f"Edge-touching originals: {edge_count}")
    print(f"Manifest: {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
