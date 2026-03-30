from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply configurable regex replacement rules to one or more text files."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file with targets and regex rules.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned replacements without writing files.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Default file encoding when a target does not specify one.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return payload


def compile_flags(flag_names: list[str]) -> int:
    value = 0
    for name in flag_names:
        key = str(name).strip().upper()
        if not key:
            continue
        if not hasattr(re, key):
            raise SystemExit(f"Unsupported regex flag: {name}")
        value |= getattr(re, key)
    return value


def resolve_targets(config: dict, base_dir: Path) -> list[Path]:
    raw_targets = config.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise SystemExit("Config field 'targets' must be a non-empty array.")
    targets: list[Path] = []
    for item in raw_targets:
        if not isinstance(item, str) or not item.strip():
            raise SystemExit("Each target must be a non-empty string path.")
        path = Path(item)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        targets.append(path)
    return targets


def resolve_rules(config: dict) -> list[dict]:
    raw_rules = config.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise SystemExit("Config field 'rules' must be a non-empty array.")
    rules: list[dict] = []
    for idx, item in enumerate(raw_rules, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"Rule #{idx} must be an object.")
        pattern = item.get("pattern")
        replacement = item.get("replacement", "")
        if not isinstance(pattern, str) or pattern == "":
            raise SystemExit(f"Rule #{idx} must define a non-empty string 'pattern'.")
        if not isinstance(replacement, str):
            raise SystemExit(f"Rule #{idx} field 'replacement' must be a string.")
        rules.append(
            {
                "name": str(item.get("name") or f"rule_{idx}"),
                "pattern": pattern,
                "replacement": replacement,
                "flags": compile_flags(item.get("flags") or []),
                "count": int(item.get("count", 0) or 0),
            }
        )
    return rules


def apply_rules(text: str, rules: list[dict]) -> tuple[str, list[tuple[str, int]]]:
    updated = text
    summary: list[tuple[str, int]] = []
    for rule in rules:
        compiled = re.compile(rule["pattern"], rule["flags"])
        updated, replacements = compiled.subn(rule["replacement"], updated, count=rule["count"])
        summary.append((rule["name"], replacements))
    return updated, summary


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    base_dir = config_path.parent
    targets = resolve_targets(config, base_dir)
    rules = resolve_rules(config)

    total_replacements = 0

    for target in targets:
        if not target.exists():
            raise SystemExit(f"Target file not found: {target}")
        encoding = str(config.get("encoding") or args.encoding)
        original = target.read_text(encoding=encoding)
        updated, summary = apply_rules(original, rules)
        changed = updated != original
        file_replacements = sum(count for _, count in summary)
        total_replacements += file_replacements

        print(f"[file] {target}")
        for name, count in summary:
            print(f"  - {name}: {count}")

        if args.dry_run:
            print(f"  => dry-run changed={changed} replacements={file_replacements}")
            continue

        if changed:
            target.write_text(updated, encoding=encoding)
            print(f"  => written replacements={file_replacements}")
        else:
            print("  => no change")

    print(f"[summary] total replacements = {total_replacements}")


if __name__ == "__main__":
    main()
