import json
from pathlib import Path


def create_manifest(manifest_entries: list[dict], output_dir: Path) -> Path:
    """Write the dubbing manifest to output_dir/manifest.json."""
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, ensure_ascii=False, indent=2)
    return manifest_path
