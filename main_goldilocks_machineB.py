"""
Machine B launcher for Goldilocks campaign phases.

Usage:
    uv run main_goldilocks_machineB.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

DEFAULT_ARTIFACT_ROOT = Path("D:/YOLOv11_ARTIFACTS")


def main() -> int:
    # Auto-load artifact root so users can run one command without setting env manually.
    os.environ.setdefault("YOLO_STAGE1_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))
    artifact_root = Path(os.environ["YOLO_STAGE1_ARTIFACT_ROOT"]).expanduser()
    artifact_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Machine B: D(k-sweep) + C + combine")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Artifact root: {artifact_root}")
    print("=" * 60)

    from scripts.run_goldilocks_campaign import main as campaign_main

    failures: list[str] = []
    original_argv = sys.argv[:]

    phases = [
        ("peak", "D", "D peak k-sweep (33 windows)"),
        ("peak", "C", "C peak (11 windows)"),
        ("combine", "", "combine validation"),
    ]

    for phase, signal, label in phases:
        print(f"\n{'#' * 60}")
        print(f"  {label}")
        print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#' * 60}\n")

        argv = [original_argv[0], "--phase", phase, "--device", "0"]
        if signal:
            argv.extend(["--signal", signal])
        sys.argv = argv

        try:
            campaign_main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
            if code not in (0, None):
                failures.append(label)
                print(f"  FAILED: exit code {code}")
        except Exception as exc:  # pragma: no cover - runtime safety
            failures.append(label)
            print(f"  FAILED: {exc}")
            traceback.print_exc()
        finally:
            sys.argv = original_argv

    print(f"\n{'=' * 60}")
    print("  Machine B complete")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    if failures:
        print(f"  Failed phases: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
