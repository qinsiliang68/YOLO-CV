from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render thesis_format.tex from JSON config and template.")
    parser.add_argument(
        "--config",
        default=r"C:\GitHub\YOLO-CV\essay\docs\thesis_format.json",
        help="Path to thesis format JSON config.",
    )
    parser.add_argument(
        "--template",
        default=r"C:\GitHub\YOLO-CV\essay\docs\thesis_format.template.tex",
        help="Path to thesis format template.",
    )
    parser.add_argument(
        "--output",
        default=r"C:\GitHub\YOLO-CV\essay\docs\thesis_format.tex",
        help="Rendered output path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return payload


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    template_path = Path(args.template).resolve()
    output_path = Path(args.output).resolve()

    config = load_json(config_path)
    template = template_path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in config:
            raise SystemExit(f"Missing thesis format key: {key}")
        return str(config[key])

    rendered = re.sub(r"\{\{\{([a-zA-Z0-9_]+)\}\}\}", replace, template)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"[done] rendered thesis format -> {output_path}")


if __name__ == "__main__":
    main()
