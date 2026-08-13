# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2", "torch>=2.13,<3"]
# ///
"""Build the deterministic local question-quality checkpoint.

How to run:
    uv run scripts/build_quality_checkpoint.py
    uv run scripts/build_quality_checkpoint.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

CHECKPOINT = (
    Path(__file__).parents[1]
    / "skills"
    / "producing-exam-prep-packages"
    / "assets"
    / "question_quality_classifier.pt"
)
PORTABLE_EXPORT = CHECKPOINT.with_suffix(".json")
EXPECTED_WEIGHTS = torch.tensor([0.9, 1.0, 0.7, 0.9, 0.9], dtype=torch.float32)
EXPECTED_BIAS = torch.tensor(-2.0, dtype=torch.float32)


class QualityModel(torch.nn.Module):
    """Fixed lightweight classifier used for delivery provenance."""

    def __init__(self) -> None:
        """Register immutable release parameters."""
        super().__init__()
        self.register_buffer("weights", EXPECTED_WEIGHTS)
        self.register_buffer("bias", EXPECTED_BIAS)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Score normalized language and completeness features."""
        return torch.sigmoid(torch.dot(features, self.weights) + self.bias)


def build() -> str:
    """Write fixed classifier parameters and return the checkpoint hash."""
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    model = torch.jit.script(QualityModel())
    torch.jit.save(model, str(CHECKPOINT))
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    PORTABLE_EXPORT.write_text(
        json.dumps(
            {
                "bias": float(EXPECTED_BIAS.item()),
                "format": "exam-prep-pytorch-linear-v1",
                "source_checkpoint_sha256": digest,
                "weights": [float(value) for value in EXPECTED_WEIGHTS.tolist()],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return digest


def check() -> str:
    """Verify the bundled checkpoint contains the expected fixed parameters."""
    model = torch.jit.load(str(CHECKPOINT), map_location="cpu")
    probe = torch.ones(5, dtype=torch.float32)
    expected = torch.sigmoid(torch.dot(probe, EXPECTED_WEIGHTS) + EXPECTED_BIAS)
    if not torch.equal(model(probe), expected):
        msg = "checkpoint parameters do not match the release definition"
        raise RuntimeError(msg)
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    portable = json.loads(PORTABLE_EXPORT.read_text(encoding="utf-8"))
    if portable.get("source_checkpoint_sha256") != digest:
        msg = "portable export does not match the release checkpoint"
        raise RuntimeError(msg)
    if portable.get("weights") != [float(value) for value in EXPECTED_WEIGHTS.tolist()]:
        msg = "portable export weights do not match the release definition"
        raise RuntimeError(msg)
    if portable.get("bias") != float(EXPECTED_BIAS.item()):
        msg = "portable export bias does not match the release definition"
        raise RuntimeError(msg)
    return digest


def main() -> None:
    """Build or verify the checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    digest = check() if args.check else build()
    print(digest)


if __name__ == "__main__":
    main()
