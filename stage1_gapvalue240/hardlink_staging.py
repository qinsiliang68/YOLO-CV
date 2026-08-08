from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from .errors import ValidationError
from .util import atomic_write_json, sha256_file, stable_hash


REPLAY_PREFIX = "replay__"
BASE_CACHE_SCHEMA = "stage1_gapvalue240.hardlink_base_cache.v1"
REQUIRED_COLUMNS = {"canonical_image_relpath", "Filename"}
REPLAY_IDENTITY_COLUMNS = {
    "selection_rank",
    "sample_id",
    "y_true",
    "replay_role",
    "source_canonical_image_relpath",
    "source_filename",
    "staged_filename",
}


@dataclass(frozen=True)
class StagingExpectedCounts:
    train_defect: int = 60_000
    train_normal: int = 60_000
    val_defect: int = 12_000
    val_normal: int = 12_000

    def as_dict(self) -> dict[str, int]:
        return {
            "train_defect": self.train_defect,
            "train_normal": self.train_normal,
            "val_defect": self.val_defect,
            "val_normal": self.val_normal,
        }


@dataclass(frozen=True)
class BaseCache:
    staging_root: Path
    dataset_dir: Path
    metadata_path: Path
    snapshot_id: str
    reused: bool


@dataclass(frozen=True)
class StagedReplay:
    dataset_dir: Path
    replay_rows: int
    active_journal: Path


MANIFEST_LAYOUT = {
    "train_defect": ("train", "target_defect"),
    "train_normal": ("train", "no_target"),
    "val_defect": ("val", "target_defect"),
    "val_normal": ("val", "no_target"),
}


def _nearest_existing(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise ValidationError(f"Cannot identify filesystem volume for: {path}")
    return candidate


def _volume_identity(path: str | Path) -> str:
    resolved = Path(path).resolve(strict=False)
    if os.name == "nt":
        identity = (resolved.drive or resolved.anchor).casefold()
        if identity:
            return f"windows:{identity}"
    return f"device:{os.stat(_nearest_existing(resolved)).st_dev}"


def ensure_same_volume(dataset_root: str | Path, staging_root: str | Path) -> None:
    dataset_root = Path(dataset_root).resolve()
    staging_root = Path(staging_root).resolve(strict=False)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    source_volume = _volume_identity(dataset_root)
    staging_volume = _volume_identity(staging_root)
    if source_volume != staging_volume:
        raise ValidationError(
            "dataset_root and staging_root must be on the same volume for hardlink-only staging: "
            f"dataset={source_volume}, staging={staging_volume}"
        )


def _read_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValidationError(f"Manifest missing columns {sorted(missing)}: {path}")
        for row in reader:
            yield row


def _safe_source(dataset_root: Path, row: Mapping[str, str], manifest: Path) -> tuple[Path, str, str]:
    relpath = str(row["canonical_image_relpath"])
    filename = str(row["Filename"])
    if not relpath or Path(relpath).is_absolute() or ".." in Path(relpath).parts:
        raise ValidationError(f"Unsafe canonical_image_relpath in {manifest}: {relpath!r}")
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValidationError(f"Unsafe Filename in {manifest}: {filename!r}")
    source = (dataset_root / Path(relpath)).resolve()
    try:
        source.relative_to(dataset_root)
    except ValueError as exc:
        raise ValidationError(f"Image resolves outside dataset_root: {relpath}") from exc
    if not source.is_file():
        raise FileNotFoundError(source)
    return source, relpath, filename


def _update_identity_digest(digest: "hashlib._Hash", relpath: str, filename: str) -> None:
    digest.update(relpath.encode("utf-8"))
    digest.update(b"\0")
    digest.update(filename.encode("utf-8"))
    digest.update(b"\n")


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        raise ValidationError(
            f"Hardlink creation failed; copy fallback is forbidden: {source} -> {destination}: {exc}"
        ) from exc


def _hardlink_probe(dataset_root: Path, staging_root: Path, manifest: Path) -> None:
    try:
        first = next(_read_rows(manifest))
    except StopIteration as exc:
        raise ValidationError(f"Cannot hardlink-probe with empty manifest: {manifest}") from exc
    source, _, _ = _safe_source(dataset_root, first, manifest)
    staging_root.mkdir(parents=True, exist_ok=True)
    probe = staging_root / f".hardlink_probe_{os.getpid()}_{uuid.uuid4().hex}"
    try:
        os.link(source, probe)
        if not os.path.samefile(source, probe):
            raise ValidationError(f"Hardlink probe did not preserve file identity: {source} -> {probe}")
    except OSError as exc:
        raise ValidationError(
            f"hardlink probe failed; dataset and staging filesystem must support hardlinks: {exc}"
        ) from exc
    finally:
        if probe.exists():
            probe.unlink()


def _write_probe(root: Path, label: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".write_probe_{label}_{os.getpid()}_{uuid.uuid4().hex}"
    try:
        with probe.open("xb") as handle:
            handle.write(b"stage1-gapvalue240-write-probe")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ValidationError(f"{label} root is not writable: {root}: {exc}") from exc
    finally:
        probe.unlink(missing_ok=True)


def storage_preflight(
    *,
    dataset_root: str | Path,
    staging_root: str | Path,
    output_root: str | Path,
    hardlink_probe_manifest: str | Path,
    expected_staging_files: int,
    maximum_staging_files: int,
    minimum_staging_free_bytes: int,
    minimum_output_free_bytes: int,
) -> dict[str, object]:
    """Fail before cache construction when disk/file-entry limits are unsafe."""

    dataset = Path(dataset_root).resolve()
    staging = Path(staging_root).resolve(strict=False)
    output = Path(output_root).resolve(strict=False)
    manifest = Path(hardlink_probe_manifest).resolve()
    if expected_staging_files > maximum_staging_files:
        raise ValidationError(
            "Expected staging file count exceeds configured file ceiling: "
            f"{expected_staging_files} > {maximum_staging_files}"
        )
    ensure_same_volume(dataset, staging)
    _hardlink_probe(dataset, staging, manifest)
    _write_probe(staging, "staging")
    _write_probe(output, "output")
    staging_usage = shutil.disk_usage(_nearest_existing(staging))
    output_usage = shutil.disk_usage(_nearest_existing(output))
    if staging_usage.free < minimum_staging_free_bytes:
        raise ValidationError(
            "Insufficient staging free space: "
            f"{staging_usage.free} < {minimum_staging_free_bytes} bytes"
        )
    if output_usage.free < minimum_output_free_bytes:
        raise ValidationError(
            "Insufficient output free space: "
            f"{output_usage.free} < {minimum_output_free_bytes} bytes"
        )
    return {
        "status": "PASS",
        "dataset_volume": _volume_identity(dataset),
        "staging_volume": _volume_identity(staging),
        "expected_staging_files": int(expected_staging_files),
        "maximum_staging_files": int(maximum_staging_files),
        "staging_free_bytes": int(staging_usage.free),
        "output_free_bytes": int(output_usage.free),
        "minimum_staging_free_bytes": int(minimum_staging_free_bytes),
        "minimum_output_free_bytes": int(minimum_output_free_bytes),
    }


def _process_is_current(lock_data: Mapping[str, object]) -> bool:
    if lock_data.get("hostname") != socket.gethostname():
        return True
    try:
        import psutil

        process = psutil.Process(int(lock_data["pid"]))
        expected = float(lock_data.get("process_create_time", -1))
        return abs(process.create_time() - expected) < 1.0
    except Exception:
        return False


class ExclusiveStagingLock:
    def __init__(self, staging_root: str | Path):
        self.staging_root = Path(staging_root).resolve()
        self.path = self.staging_root / ".staging.lock"
        self.token = uuid.uuid4().hex

    def __enter__(self) -> "ExclusiveStagingLock":
        self.staging_root.mkdir(parents=True, exist_ok=True)
        try:
            import psutil

            created = psutil.Process(os.getpid()).create_time()
        except Exception:
            created = time.time()
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "process_create_time": created,
            "token": self.token,
            "created_at": time.time(),
        }
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                try:
                    current = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    current = {"hostname": "unknown"}
                if _process_is_current(current):
                    raise ValidationError(f"Staging root is locked by another worker: {self.path}")
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            return self
        raise ValidationError(f"Unable to acquire staging lock: {self.path}")

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if current.get("token") == self.token:
                self.path.unlink()
        except FileNotFoundError:
            pass


def _manifest_specs(manifests: Mapping[str, str | Path]) -> tuple[dict[str, Path], dict[str, str], str]:
    missing = set(MANIFEST_LAYOUT) - set(manifests)
    unknown = set(manifests) - set(MANIFEST_LAYOUT)
    if missing or unknown:
        raise ValidationError(f"Invalid base manifest map: missing={sorted(missing)}, unknown={sorted(unknown)}")
    paths = {name: Path(path).resolve() for name, path in manifests.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    snapshot_id = stable_hash({"schema": BASE_CACHE_SCHEMA, "manifest_sha256": hashes})
    return paths, hashes, snapshot_id


def _load_matching_cache(
    cache_root: Path,
    snapshot_id: str,
    expected: StagingExpectedCounts,
    dataset_root: Path,
) -> BaseCache | None:
    metadata_path = cache_root / "BASE_CACHE.json"
    if not cache_root.exists():
        return None
    if not metadata_path.is_file():
        raise ValidationError(f"Base cache exists without metadata: {cache_root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != BASE_CACHE_SCHEMA:
        raise ValidationError(f"Unsupported base cache schema: {metadata.get('schema_version')}")
    if metadata.get("snapshot_id") != snapshot_id:
        raise ValidationError(
            f"Base cache snapshot mismatch; cleanup must be explicit: "
            f"expected={snapshot_id}, found={metadata.get('snapshot_id')}"
        )
    if metadata.get("expected_counts") != expected.as_dict():
        raise ValidationError("Base cache expected-count contract mismatch")
    if Path(str(metadata.get("dataset_root", ""))).resolve() != dataset_root:
        raise ValidationError("Base cache dataset_root changed; cleanup/rebuild must be explicit")
    dataset_dir = cache_root / "dataset"
    for split, class_name in MANIFEST_LAYOUT.values():
        if not (dataset_dir / split / class_name).is_dir():
            raise ValidationError(f"Base cache class directory is missing: {dataset_dir / split / class_name}")
    return BaseCache(cache_root.parent, dataset_dir, metadata_path, snapshot_id, reused=True)


def prepare_base_cache(
    dataset_root: str | Path,
    staging_root: str | Path,
    manifests: Mapping[str, str | Path],
    *,
    expected_counts: StagingExpectedCounts | None = None,
) -> BaseCache:
    """Build the shared 120k train + 24k val cache using hardlinks only."""

    dataset_root = Path(dataset_root).resolve()
    staging_root = Path(staging_root).resolve(strict=False)
    expected_counts = expected_counts or StagingExpectedCounts()
    ensure_same_volume(dataset_root, staging_root)
    paths, hashes, snapshot_id = _manifest_specs(manifests)
    _hardlink_probe(dataset_root, staging_root, paths["train_defect"])
    cache_root = staging_root / "base_cache"
    inprogress = staging_root / "base_cache.inprogress"

    with ExclusiveStagingLock(staging_root):
        existing = _load_matching_cache(cache_root, snapshot_id, expected_counts, dataset_root)
        if existing:
            return existing
        if inprogress.exists():
            shutil.rmtree(inprogress)
        dataset_dir = inprogress / "dataset"
        metadata_manifests: dict[str, dict[str, object]] = {}
        try:
            for name, (split, class_name) in MANIFEST_LAYOUT.items():
                destination_dir = dataset_dir / split / class_name
                destination_dir.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                seen: set[str] = set()
                count = 0
                for row in _read_rows(paths[name]):
                    source, relpath, filename = _safe_source(dataset_root, row, paths[name])
                    if filename.startswith(REPLAY_PREFIX):
                        raise ValidationError(
                            f"Base manifest uses reserved replay prefix in {paths[name]}: {filename}"
                        )
                    if filename in seen:
                        raise ValidationError(f"Duplicate Filename in {paths[name]}: {filename}")
                    seen.add(filename)
                    _hardlink(source, destination_dir / filename)
                    _update_identity_digest(digest, relpath, filename)
                    count += 1
                expected = expected_counts.as_dict()[name]
                if count != expected:
                    raise ValidationError(f"Base manifest count mismatch for {name}: {count} != {expected}")
                metadata_manifests[name] = {
                    "path": str(paths[name]),
                    "sha256": hashes[name],
                    "rows": count,
                    "base_identity_digest": digest.hexdigest().upper(),
                }
            metadata = {
                "schema_version": BASE_CACHE_SCHEMA,
                "snapshot_id": snapshot_id,
                "link_mode": "hardlink_only",
                "dataset_root": str(dataset_root),
                "expected_counts": expected_counts.as_dict(),
                "total_rows": sum(expected_counts.as_dict().values()),
                "manifests": metadata_manifests,
            }
            atomic_write_json(inprogress / "BASE_CACHE.json", metadata)
            inprogress.rename(cache_root)
        except Exception:
            if inprogress.exists():
                shutil.rmtree(inprogress)
            raise
    return BaseCache(staging_root, cache_root / "dataset", cache_root / "BASE_CACHE.json", snapshot_id, reused=False)


def _cleanup_replay_links(dataset_dir: Path) -> int:
    removed = 0
    for class_name in ("target_defect", "no_target"):
        class_dir = dataset_dir / "train" / class_name
        if not class_dir.is_dir():
            continue
        for path in class_dir.glob(f"{REPLAY_PREFIX}*"):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def _cleanup_ultralytics_cache(dataset_dir: Path) -> None:
    for path in (dataset_dir / "train.cache", dataset_dir / "val.cache"):
        if path.exists():
            if not path.is_file():
                raise ValidationError(f"Refusing to remove non-file Ultralytics cache path: {path}")
            path.unlink()


def _split_combined_manifest(
    path: Path,
    dataset_root: Path,
    expected_base: Mapping[str, object],
) -> list[tuple[Path, str]]:
    digest = hashlib.sha256()
    base_count = 0
    replay: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for row in _read_rows(path):
        source, relpath, filename = _safe_source(dataset_root, row, path)
        if filename in seen:
            raise ValidationError(f"Duplicate Filename in combined manifest {path}: {filename}")
        seen.add(filename)
        if filename.startswith(REPLAY_PREFIX):
            replay.append((source, filename))
        else:
            _update_identity_digest(digest, relpath, filename)
            base_count += 1
    if base_count != int(expected_base["rows"]):
        raise ValidationError(f"Combined manifest base count mismatch: {path}: {base_count} != {expected_base['rows']}")
    actual_digest = digest.hexdigest().upper()
    if actual_digest != expected_base["base_identity_digest"]:
        raise ValidationError(f"Combined manifest base identities differ from frozen base cache: {path}")
    return replay


@contextmanager
def staged_replay_session(
    cache: BaseCache,
    combined_defect_manifest: str | Path,
    combined_normal_manifest: str | Path,
    *,
    run_slot: str,
    expected_replay_rows: int,
) -> Iterator[StagedReplay]:
    """Temporarily add one run's replay links to the shared cache under an exclusive lock."""

    metadata = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
    dataset_root = Path(metadata["dataset_root"]).resolve()
    if not dataset_root.is_dir():
        raise ValidationError(f"Base-cache dataset_root is unavailable: {dataset_root}")

    active_journal = cache.staging_root / "ACTIVE_REPLAY.json"
    with ExclusiveStagingLock(cache.staging_root):
        _cleanup_replay_links(cache.dataset_dir)
        _cleanup_ultralytics_cache(cache.dataset_dir)
        defect_replay = _split_combined_manifest(
            Path(combined_defect_manifest).resolve(),
            dataset_root,
            metadata["manifests"]["train_defect"],
        )
        normal_replay = _split_combined_manifest(
            Path(combined_normal_manifest).resolve(),
            dataset_root,
            metadata["manifests"]["train_normal"],
        )
        replay_rows = len(defect_replay) + len(normal_replay)
        if replay_rows != expected_replay_rows:
            raise ValidationError(f"Replay row count mismatch for {run_slot}: {replay_rows} != {expected_replay_rows}")
        created: list[Path] = []
        try:
            for source, filename in defect_replay:
                destination = cache.dataset_dir / "train/target_defect" / filename
                _hardlink(source, destination)
                created.append(destination)
            for source, filename in normal_replay:
                destination = cache.dataset_dir / "train/no_target" / filename
                _hardlink(source, destination)
                created.append(destination)
            atomic_write_json(
                active_journal,
                {
                    "run_slot": run_slot,
                    "snapshot_id": cache.snapshot_id,
                    "replay_rows": replay_rows,
                    "created_links": [path.relative_to(cache.dataset_dir).as_posix() for path in created],
                    "pid": os.getpid(),
                },
                overwrite=True,
            )
            yield StagedReplay(cache.dataset_dir, replay_rows, active_journal)
        finally:
            _cleanup_replay_links(cache.dataset_dir)
            _cleanup_ultralytics_cache(cache.dataset_dir)
            if active_journal.exists():
                active_journal.unlink()


def _read_replay_identities(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REPLAY_IDENTITY_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValidationError(f"Replay identity manifest missing columns {sorted(missing)}: {path}")
        return [dict(row) for row in reader]


@contextmanager
def staged_identity_replay_session(
    cache: BaseCache,
    replay_identity_manifest: str | Path,
    *,
    run_slot: str,
    expected_replay_rows: int,
) -> Iterator[StagedReplay]:
    """Stage only compact registered replay identities, without 120k-row combined manifests."""

    identity_path = Path(replay_identity_manifest).resolve()
    rows = _read_replay_identities(identity_path)
    if len(rows) != expected_replay_rows:
        raise ValidationError(
            f"Replay identity row count mismatch for {run_slot}: {len(rows)} != {expected_replay_rows}"
        )
    metadata = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
    dataset_root = Path(metadata["dataset_root"]).resolve()
    if not dataset_root.is_dir():
        raise ValidationError(f"Base-cache dataset_root is unavailable: {dataset_root}")

    frozen_pairs: dict[int, set[tuple[str, str]]] = {}
    for y_true, manifest_key in ((0, "train_normal"), (1, "train_defect")):
        manifest = Path(str(metadata["manifests"][manifest_key]["path"])).resolve()
        frozen_pairs[y_true] = {
            (str(row["canonical_image_relpath"]), str(row["Filename"]))
            for row in _read_rows(manifest)
        }

    resolved: list[tuple[Path, Path]] = []
    seen_sources: set[tuple[int, str]] = set()
    seen_destinations: set[Path] = set()
    expected_prefix = f"replay__{run_slot}__"
    for row in rows:
        try:
            y_true = int(row["y_true"])
            rank = int(row["selection_rank"])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid replay identity numeric field in {identity_path}") from exc
        if y_true not in (0, 1) or rank <= 0:
            raise ValidationError(f"Invalid replay identity label/rank in {identity_path}")
        role = str(row["replay_role"])
        expected_role = "normal_replay" if y_true == 0 else "defect_guard"
        if role != expected_role:
            raise ValidationError(f"Replay identity role conflicts with label: {row['sample_id']}")
        relpath = str(row["source_canonical_image_relpath"])
        source_filename = str(row["source_filename"])
        if str(row["sample_id"]) != relpath:
            raise ValidationError(f"Replay identity sample_id differs from canonical source: {row['sample_id']}")
        if (relpath, source_filename) not in frozen_pairs[y_true]:
            raise ValidationError(f"Replay identity is absent from frozen base manifest: {relpath}")
        source, verified_relpath, verified_filename = _safe_source(
            dataset_root,
            {"canonical_image_relpath": relpath, "Filename": source_filename},
            identity_path,
        )
        if verified_relpath != relpath or verified_filename != source_filename:
            raise ValidationError(f"Replay source identity changed: {relpath}")
        source_key = (y_true, relpath)
        if source_key in seen_sources:
            raise ValidationError(f"Duplicate replay source identity: {relpath}")
        seen_sources.add(source_key)
        staged_filename = str(row["staged_filename"])
        if (
            not staged_filename.startswith(expected_prefix)
            or Path(staged_filename).name != staged_filename
            or not staged_filename.startswith(REPLAY_PREFIX)
        ):
            raise ValidationError(f"Invalid staged replay filename: {staged_filename}")
        class_name = "no_target" if y_true == 0 else "target_defect"
        destination = cache.dataset_dir / "train" / class_name / staged_filename
        if destination in seen_destinations:
            raise ValidationError(f"Duplicate replay destination: {destination}")
        seen_destinations.add(destination)
        resolved.append((source, destination))

    active_journal = cache.staging_root / "ACTIVE_REPLAY.json"
    with ExclusiveStagingLock(cache.staging_root):
        _cleanup_replay_links(cache.dataset_dir)
        _cleanup_ultralytics_cache(cache.dataset_dir)
        created: list[Path] = []
        try:
            for source, destination in resolved:
                _hardlink(source, destination)
                created.append(destination)
            atomic_write_json(
                active_journal,
                {
                    "run_slot": run_slot,
                    "snapshot_id": cache.snapshot_id,
                    "replay_rows": len(created),
                    "replay_identity_manifest": str(identity_path),
                    "replay_identity_sha256": sha256_file(identity_path),
                    "created_links": [path.relative_to(cache.dataset_dir).as_posix() for path in created],
                    "pid": os.getpid(),
                },
                overwrite=True,
            )
            yield StagedReplay(cache.dataset_dir, len(created), active_journal)
        finally:
            _cleanup_replay_links(cache.dataset_dir)
            _cleanup_ultralytics_cache(cache.dataset_dir)
            active_journal.unlink(missing_ok=True)
