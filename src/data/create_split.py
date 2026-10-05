import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/writer01"
OUTPUT = ROOT / "data/splits/writer01-v1.json"
SEED = 42


def main():
    if OUTPUT.exists():
        raise SystemExit(
            "Split already exists. Keep it unchanged for comparable experiments."
        )

    rng = random.Random(SEED)
    training = []
    validation = []

    for digit in range(10):
        samples = sorted((RAW_DIR / str(digit)).glob("*.png"))
        if len(samples) != 10:
            raise ValueError(
                f"Digit {digit}: expected 10 samples, found {len(samples)}"
            )

        rng.shuffle(samples)
        training.extend(path.relative_to(ROOT).as_posix() for path in samples[:8])
        validation.extend(path.relative_to(ROOT).as_posix() for path in samples[8:])

    assert len(training) == 80
    assert len(validation) == 20
    assert not set(training) & set(validation)
    assert len(set(training + validation)) == 100

    split = {
        "name": "writer01-v1",
        "seed": SEED,
        "evaluation": "same-writer validation",
        "train": sorted(training),
        "validation": sorted(validation),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(split, indent=2) + "\n")

    print(f"Training: {len(training)} samples, 8 per digit")
    print(f"Validation: {len(validation)} samples, 2 per digit")
    print("No overlap between training and validation")
    print(f"Saved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
