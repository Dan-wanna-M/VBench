#!/usr/bin/env python3
"""Standalone script to run VBench text_alignment evaluation.

Meant to be called via subprocess using the VBench venv Python, e.g.:
    third_party/VBench/.venv/bin/python third_party/VBench/run_text_alignment.py \
        --videos_path /path/to/videos \
        --output_path /path/to/output \
        --prompt_file /path/to/prompt_map.json   # optional
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VBench text_alignment evaluation")
    parser.add_argument("--videos_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--prompt_file",
        type=Path,
        default=None,
        help="JSON file mapping video filename -> prompt (used to build sorted prompt list)",
    )
    args = parser.parse_args()

    # Import VBenchCompetition (available because VBench is installed in this venv)
    vbench_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(vbench_root))
    sys.path.insert(0, str(vbench_root / "competitions"))
    from competitions import VBenchCompetition

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_path.mkdir(parents=True, exist_ok=True)

    prompt_list: list[str] = []
    if args.prompt_file is not None:
        with args.prompt_file.open("r", encoding="utf-8") as f:
            prompt_map: dict[str, str] = json.load(f)
        sorted_videos = sorted(
            str(args.videos_path / fname)
            for fname in os.listdir(args.videos_path)
            if fname.lower().endswith(".mp4")
        )
        prompt_list = [prompt_map[Path(v).name] for v in sorted_videos]

    bench = VBenchCompetition(device, None, str(args.output_path))
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    bench.evaluate(
        videos_path=str(args.videos_path),
        name=f"results_{current_time}",
        prompt_list=prompt_list,
        dimension_list=["text_alignment"],
    )
    print(f"[text_alignment] Saved to {args.output_path}")


if __name__ == "__main__":
    main()
