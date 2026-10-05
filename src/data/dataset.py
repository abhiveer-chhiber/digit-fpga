import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = ROOT / "data/splits/writer01-v1.json"


def load_dataset(resolution, subset):
    if resolution not in (8, 16):
        raise ValueError("Resolution must be 8 or 16")
    if subset not in ("train", "validation"):
        raise ValueError("Subset must be train or validation")

    split = json.loads(SPLIT_PATH.read_text())
    train_paths = set(split["train"])
    validation_paths = set(split["validation"])

    if train_paths & validation_paths:
        raise ValueError("Training and validation overlap")

    paths = split[subset]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError(f"Empty subset or duplicate samples: {subset}")

    inputs = []
    labels = []

    for raw_path in paths:
        relative = Path(raw_path).relative_to("data/raw")
        processed_path = (
            ROOT / "data/processed" / f"{resolution}x{resolution}" / relative
        )

        if not (ROOT / raw_path).is_file():
            raise FileNotFoundError(raw_path)

        with Image.open(processed_path) as image:
            if image.mode != "L" or image.size != (resolution, resolution):
                raise ValueError(f"Unexpected image format: {processed_path}")
            pixels = np.asarray(image, dtype=np.float32) / 255.0

        inputs.append(pixels.reshape(-1))
        labels.append(int(relative.parent.name))

    return (
        np.stack(inputs),
        np.asarray(labels, dtype=np.int64),
        paths,
    )


def main():
    for resolution in (8, 16):
        for subset in ("train", "validation"):
            inputs, labels, paths = load_dataset(resolution, subset)
            assert np.isfinite(inputs).all()
            assert np.all((inputs >= 0) & (inputs <= 1))
            assert np.all(inputs.max(axis=1) > 0)
            assert len(inputs) == len(labels) == len(paths)

            counts = np.bincount(labels, minlength=10)
            expected = 8 if subset == "train" else 2
            assert np.all(counts == expected)

            print(
                f"{resolution}x{resolution} {subset}: "
                f"inputs={inputs.shape}, labels={labels.shape}, "
                f"{expected} per digit"
            )

    for subset in ("train", "validation"):
        _, labels8, paths8 = load_dataset(8, subset)
        _, labels16, paths16 = load_dataset(16, subset)
        assert paths8 == paths16
        assert np.array_equal(labels8, labels16)

    print("Both resolutions use identical sample assignments")


if __name__ == "__main__":
    main()
