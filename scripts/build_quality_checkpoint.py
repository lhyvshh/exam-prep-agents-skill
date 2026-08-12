# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2", "torch>=2.2,<3"]
# ///
"""Build the deterministic local question-quality checkpoint.

How to run:
    uv run scripts/build_quality_checkpoint.py
    uv run scripts/build_quality_checkpoint.py --check
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

CHECKPOINT = (
    Path(__file__).parents[1]
    / "skills"
    / "producing-exam-prep-packages"
    / "assets"
    / "question_quality_classifier.pt"
)
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
    return hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()


def check() -> str:
    """Verify the bundled checkpoint contains the expected fixed parameters."""
    model = torch.jit.load(str(CHECKPOINT), map_location="cpu")
    probe = torch.ones(5, dtype=torch.float32)
    expected = torch.sigmoid(torch.dot(probe, EXPECTED_WEIGHTS) + EXPECTED_BIAS)
    if not torch.equal(model(probe), expected):
        msg = "checkpoint parameters do not match the release definition"
        raise RuntimeError(msg)
    return hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()


def main() -> None:
    """Build or verify the checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    digest = check() if args.check else build()
    print(digest)


if __name__ == "__main__":
    main()
