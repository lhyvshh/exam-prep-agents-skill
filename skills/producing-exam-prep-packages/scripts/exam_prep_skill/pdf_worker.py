"""Isolated PyMuPDF process used to contain native extension failures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf as fitz


def main() -> None:
    """Write extracted page records to standard output."""
    path = Path(sys.argv[1])
    document = fitz.open(path)
    try:
        pages: list[dict[str, int | str]] = [
            {"physical_page": index + 1, "text": page.get_text("text").strip()}
            for index, page in enumerate(document)
        ]
    finally:
        document.close()
    sys.stdout.write(json.dumps(pages))


if __name__ == "__main__":
    main()
