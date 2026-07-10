#!/usr/bin/env python3
"""Attach pre-rendered static-geometry videos to an Echo-Memory metadata CSV."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--geometry_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--column", default="geometry_memory")
    parser.add_argument("--template", default="{video_name}/Vid_masktarget.mp4")
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    geometry_root = Path(args.geometry_root).resolve()
    output_path = Path(args.output).resolve()
    table = pd.read_csv(metadata_path)
    if "video_name" not in table.columns:
        raise ValueError("metadata must contain a video_name column")

    relative_paths = []
    missing = []
    for index, row in table.iterrows():
        values = {key: row[key] for key in table.columns}
        relative = args.template.format(**values)
        path = geometry_root / relative
        if not path.is_file():
            missing.append((index, str(path)))
        relative_paths.append(relative)

    if missing and not args.allow_missing:
        examples = "\n".join(f"  row {idx}: {path}" for idx, path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} geometry videos are missing under {geometry_root}. Examples:\n{examples}"
        )

    table[args.column] = relative_paths
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(
        f"Wrote {len(table)} rows to {output_path}; "
        f"geometry_column={args.column}, missing={len(missing)}"
    )


if __name__ == "__main__":
    main()

