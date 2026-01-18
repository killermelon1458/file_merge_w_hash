#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

BUF_SIZE = 1024 * 1024  # 1 MiB


def try_import_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


@dataclass(frozen=True)
class FileSig:
    size: int
    mtime_ns: int


class HashCache:
    """SQLite-backed cache: path -> (size, mtime_ns, sha256).

    Supports read-only mode (for dry-run) and can be configured to skip writes.
    """

    def __init__(self, db_path: Path, *, readonly: bool = False, allow_writes: bool = True):
        self.db_path = db_path
        self.readonly = readonly
        self.allow_writes = allow_writes and (not readonly)

        # If readonly but cache doesn't exist, fall back to in-memory cache.
        if readonly and not db_path.exists():
            self.conn = sqlite3.connect(":memory:")
            self._init()
            return

        if not readonly:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(db_path))
            self._init()
            return

        # Read-only open
        uri = f"file:{db_path.as_posix()}?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True)

    def _init(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS filehash (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, path: Path, sig: FileSig) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute(
            """SELECT sha256, size, mtime_ns FROM filehash WHERE path = ?""",
            (str(path),),
        )
        row = cur.fetchone()
        if not row:
            return None
        sha256, size, mtime_ns = row
        if int(size) == sig.size and int(mtime_ns) == sig.mtime_ns:
            return str(sha256)
        return None

    def put(self, path: Path, sig: FileSig, sha256: str) -> None:
        if not self.allow_writes:
            return
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO filehash(path, size, mtime_ns, sha256)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size=excluded.size,
                mtime_ns=excluded.mtime_ns,
                sha256=excluded.sha256
            """,
            (str(path), sig.size, sig.mtime_ns, sha256),
        )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


class ProgressTracker:
    """Tracks progress and allows resume from checkpoint. Only records SUCCESSFUL work."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.processed_files: set[str] = set()
        self.stats = {
            "copied": 0,
            "renamed": 0,
            "skipped_dup": 0,
            "symlinks_copied": 0,
            "symlinks_skipped": 0,
            "deleted_sources": 0,
            "errors": 0,
        }
        self.load()

    def load(self) -> None:
        try:
            if self.checkpoint_path.exists():
                with self.checkpoint_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.processed_files = set(data.get("processed_files", []))
                self.stats = data.get("stats", self.stats)
        except Exception:
            pass

    def save(self) -> None:
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.checkpoint_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "processed_files": sorted(self.processed_files),
                        "stats": self.stats,
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp, self.checkpoint_path)
        except Exception:
            pass

    def clear(self) -> None:
        try:
            self.checkpoint_path.unlink()
        except Exception:
            pass
        self.processed_files = set()

    def is_processed(self, key: str) -> bool:
        return key in self.processed_files

    def mark_processed(self, key: str) -> None:
        self.processed_files.add(key)

    def increment(self, stat: str, n: int = 1) -> None:
        self.stats[stat] = int(self.stats.get(stat, 0)) + n


def get_disk_space(path: Path) -> int:
    usage = shutil.disk_usage(str(path))
    return usage.free


def gib(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def check_disk_space(dst_root: Path, bytes_to_copy: int, margin: float) -> Tuple[bool, str, int, int]:
    free = get_disk_space(dst_root)
    required = int(bytes_to_copy * (1.0 + margin))
    ok = free >= required
    msg = f"Free: {gib(free):.2f} GiB, need: {gib(required):.2f} GiB (margin {margin*100:.0f}%) -> {'OK' if ok else 'NOT OK'}"
    return ok, msg, free, required


def file_sig(p: Path) -> FileSig:
    st = p.stat()
    return FileSig(size=st.st_size, mtime_ns=st.st_mtime_ns)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(BUF_SIZE)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cached_hash(p: Path, cache: HashCache, *, write_cache: bool = True) -> str:
    """Return SHA-256 for path, using cache when valid.

    If write_cache is False, the cache is used for reads but will not be updated.
    """
    sig = file_sig(p)
    got = cache.get(p, sig)
    if got:
        return got
    digest = sha256_file(p)
    if write_cache:
        cache.put(p, sig, digest)
    return digest


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    with src.open("rb") as rf, tmp.open("wb") as wf:
        shutil.copyfileobj(rf, wf, length=BUF_SIZE)
    shutil.copystat(src, tmp, follow_symlinks=False)
    os.replace(tmp, dst)


def add_suffix(path: Path, n: int) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem} ({n}){path.suffix}")
    return path.with_name(f"{path.name} ({n})")


def should_exclude(rel_path: Path, exclude_names: set[str], exclude_globs: list[str]) -> bool:
    name = rel_path.name
    if name in exclude_names:
        return True
    rel_str = str(rel_path).replace("\\", "/")
    for pat in exclude_globs:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_str, pat):
            return True
    return False


def iter_items(root: Path, *, ignore_symlinks: bool) -> tuple[str, Path, Path]:
    # yields (kind, path, relpath)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirp = Path(dirpath)
        rel_dir = dirp.relative_to(root)
        # dirs
        for d in list(dirnames):
            p = dirp / d
            rel = (rel_dir / d)
            try:
                if p.is_symlink():
                    if ignore_symlinks:
                        continue
                    yield ("symlink", p, rel)
                else:
                    yield ("dir", p, rel)
            except Exception:
                continue
        # files
        for fn in filenames:
            p = dirp / fn
            rel = (rel_dir / fn)
            try:
                if p.is_symlink():
                    if ignore_symlinks:
                        continue
                    yield ("symlink", p, rel)
                else:
                    yield ("file", p, rel)
            except Exception:
                continue


def format_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def ensure_symlink(dst: Path, target: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    os.symlink(target, tmp)
    os.replace(tmp, dst)


# =========================
# Importable API (Option A)
# =========================

@dataclass
class MergeRunConfig:
    """Programmatic configuration for a merge run (Option A integration)."""

    src_roots: list[Path]
    dst_root: Path
    log_path: Path
    cache_path: Path
    checkpoint_path: Optional[Path] = None

    move: bool = False
    log_dups: bool = False
    space_margin: float = 0.1
    force_space: bool = False
    max_errors: int = 100
    ignore_symlinks: bool = False
    exclude_names: list[str] = None  # type: ignore
    exclude_globs: list[str] = None  # type: ignore
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.exclude_names is None:
            self.exclude_names = []
        if self.exclude_globs is None:
            self.exclude_globs = []


@dataclass
class MergeRunResult:
    exit_code: int
    log_path: Path
    checkpoint_path: Path
    stats: dict


class DryProgressTracker(ProgressTracker):
    """Progress tracker that never reads/writes checkpoints (for dry-run)."""

    def load(self) -> None:  # type: ignore[override]
        self.processed_files = set()

    def save(self) -> None:  # type: ignore[override]
        return

    def clear(self) -> None:  # type: ignore[override]
        self.processed_files = set()


def run_merge(cfg: MergeRunConfig) -> MergeRunResult:
    """Run merge programmatically (importable API).

    This is the preferred entrypoint for backup_runner.py (Option A).
    """
    args = argparse.Namespace()
    args.src = [str(p) for p in cfg.src_roots]
    args.dst = str(cfg.dst_root)
    args.log = str(cfg.log_path)
    args.cache = str(cfg.cache_path)
    args.checkpoint = str(cfg.checkpoint_path) if cfg.checkpoint_path else None

    args.move = cfg.move
    args.log_dups = cfg.log_dups
    args.space_margin = cfg.space_margin
    args.force_space = cfg.force_space
    args.max_errors = cfg.max_errors
    args.ignore_symlinks = cfg.ignore_symlinks
    args.exclude_name = cfg.exclude_names
    args.exclude_glob = cfg.exclude_globs
    args.dry_run = cfg.dry_run

    return _run_with_args(args)


def _run_with_args(args: argparse.Namespace) -> MergeRunResult:
    """Internal runner shared by CLI and programmatic API."""
    src_roots = [Path(s).resolve() for s in args.src]
    dst_root = Path(args.dst).resolve()
    log_path = Path(args.log).resolve()
    cache_path = Path(args.cache).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else cache_path.with_suffix(".checkpoint")

    # Dry-run: read-only cache if possible, and do not write checkpoints.
    cache = HashCache(
        cache_path,
        readonly=bool(getattr(args, "dry_run", False)),
        allow_writes=not bool(getattr(args, "dry_run", False)),
    )
    tracker: ProgressTracker
    if getattr(args, "dry_run", False):
        tracker = DryProgressTracker(checkpoint_path)
    else:
        tracker = ProgressTracker(checkpoint_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    tqdm = try_import_tqdm()

    def progress(nbytes: int) -> None:
        if tqdm_bar is not None:
            tqdm_bar.update(nbytes)

    # Header
    log(f"=== merge_duplicates.py run @ {format_ts(time.time())} ===")
    log("Version: importable API enabled")
    log(f"Src roots: {', '.join(str(p) for p in src_roots)}")
    log(f"Dst root : {dst_root}")
    log(f"Dry run  : {bool(getattr(args, 'dry_run', False))}")
    log(f"Move     : {args.move}")
    log(f"Cache    : {cache_path} (readonly={cache.readonly})")
    log(f"Checkpoint: {checkpoint_path} (enabled={not getattr(args, 'dry_run', False)})")

    exclude_names = set(args.exclude_name or [])
    exclude_globs = list(args.exclude_glob or [])

    # Build list of all items
    all_items: list[tuple[str, Path, Path, Path]] = []
    total_bytes_all = 0
    total_bytes_to_process = 0
    per_src_remaining_bytes: dict[str, int] = {str(r): 0 for r in src_roots}

    for src_root in src_roots:
        for kind, src_path, rel in iter_items(src_root, ignore_symlinks=args.ignore_symlinks):
            if should_exclude(rel, exclude_names, exclude_globs):
                continue
            all_items.append((kind, src_root, src_path, rel))
            if kind == "file":
                try:
                    sz = src_path.stat().st_size
                except Exception:
                    sz = 0
                total_bytes_all += sz
                file_key = f"{src_root}::{rel}"
                if not tracker.is_processed(file_key):
                    total_bytes_to_process += sz
                    per_src_remaining_bytes[str(src_root)] += sz

    print(f"Found {len(all_items)} items ({gib(total_bytes_all):.2f} GiB total across all sources)")
    if tracker.processed_files:
        print(f"Resuming: {len(tracker.processed_files)} files already processed")
        print(f"Remaining: {gib(total_bytes_to_process):.2f} GiB to process")

    # Disk space check
    print("Checking disk space...")
    ok, msg, free_bytes, required_with_margin = check_disk_space(dst_root, total_bytes_to_process, args.space_margin)
    log(f"Disk space check: {msg}")
    print(msg)

    smallest_possible = max(per_src_remaining_bytes.values()) if per_src_remaining_bytes else 0
    largest_possible = total_bytes_to_process
    bounds = (
        f"Bounds for remaining run: smallest possible add = {gib(smallest_possible):.2f} GiB "
        f"(if everything overlaps); largest possible add = {gib(largest_possible):.2f} GiB "
        f"(if nothing overlaps). Free now: {gib(free_bytes):.2f} GiB. "
        f"Required w/ margin: {gib(required_with_margin):.2f} GiB."
    )
    log(bounds)
    print(bounds)

    dry_run = bool(getattr(args, "dry_run", False))
    if (not ok) and (not args.force_space) and (not dry_run):
        cache.close()
        fatal_msg = "Insufficient disk space (use --force-space to override, NOT recommended)."
        log(f"ABORTED: {fatal_msg}")
        print(f"\nERROR: {fatal_msg}")
        raise SystemExit(3)

    # Progress bar
    tqdm_bar = None
    if tqdm and total_bytes_to_process > 0:
        tqdm_bar = tqdm(total=total_bytes_to_process, unit="B", unit_scale=True, unit_divisor=1024)

    checkpoint_counter = 0
    write_cache = not dry_run

    try:
        for kind, src_root, src_path, rel in all_items:
            if kind == "dir":
                continue

            file_key = f"{src_root}::{rel}"
            if tracker.is_processed(file_key):
                continue

            dst_path = dst_root / rel

            try:
                # --- Symlinks ---
                if kind == "symlink":
                    if args.ignore_symlinks:
                        tracker.increment("symlinks_skipped")
                        tracker.mark_processed(file_key)
                        continue

                    target = os.readlink(src_path)

                    if (dst_path.exists() or dst_path.is_symlink()) and dst_path.is_symlink():
                        try:
                            if os.readlink(dst_path) == target:
                                tracker.increment("skipped_dup")
                                if args.log_dups:
                                    log(f"[DUP_SYMLINK] {src_path} == {dst_path} (target={target})")
                                if args.move and not dry_run:
                                    src_path.unlink()
                                    tracker.increment("deleted_sources")
                                tracker.mark_processed(file_key)
                                continue
                        except Exception:
                            pass

                    chosen = dst_path
                    if chosen.exists() or chosen.is_symlink():
                        n = 1
                        while True:
                            alt = add_suffix(dst_path, n)
                            if (not alt.exists()) and (not alt.is_symlink()):
                                chosen = alt
                                break
                            if alt.is_symlink():
                                try:
                                    if os.readlink(alt) == target:
                                        tracker.increment("skipped_dup")
                                        if args.log_dups:
                                            log(f"[DUP_SYMLINK] {src_path} == {alt} (target={target})")
                                        if args.move and not dry_run:
                                            src_path.unlink()
                                            tracker.increment("deleted_sources")
                                        tracker.mark_processed(file_key)
                                        chosen = None  # type: ignore
                                        break
                                except Exception:
                                    pass
                            n += 1
                        if chosen is None:  # type: ignore
                            continue

                    if dry_run:
                        log(f"[DRYRUN_SYMLINK] {src_path} -> {chosen} (target={target})")
                    else:
                        ensure_symlink(chosen, target)
                    tracker.increment("symlinks_copied" if chosen == dst_path else "renamed")

                    if args.move and not dry_run:
                        src_path.unlink()
                        tracker.increment("deleted_sources")

                    tracker.mark_processed(file_key)
                    checkpoint_counter += 1
                    if checkpoint_counter % 50 == 0 and not dry_run:
                        tracker.save()
                    continue

                # --- Files ---
                src_size = src_path.stat().st_size
                if not dst_path.exists():
                    if dry_run:
                        log(f"[DRYRUN_COPY] {src_path} -> {dst_path}")
                        tracker.increment("copied")
                    else:
                        atomic_copy(src_path, dst_path)
                        dst_hash = cached_hash(dst_path, cache, write_cache=write_cache)
                        tracker.increment("copied")

                        if args.move:
                            src_h = cached_hash(src_path, cache, write_cache=write_cache)
                            if src_h != dst_hash:
                                raise RuntimeError("Move verify failed: src/dst hash mismatch")
                            src_path.unlink()
                            tracker.increment("deleted_sources")

                    tracker.mark_processed(file_key)
                    progress(src_size)
                    checkpoint_counter += 1
                    if checkpoint_counter % 50 == 0 and not dry_run:
                        tracker.save()
                    continue

                try:
                    dst_size = dst_path.stat().st_size
                except FileNotFoundError:
                    if dry_run:
                        log(f"[DRYRUN_COPY] {src_path} -> {dst_path}")
                        tracker.increment("copied")
                    else:
                        atomic_copy(src_path, dst_path)
                        cached_hash(dst_path, cache, write_cache=write_cache)
                        tracker.increment("copied")

                    if args.move and (not dry_run):
                        src_h = cached_hash(src_path, cache, write_cache=write_cache)
                        dst_h = cached_hash(dst_path, cache, write_cache=write_cache)
                        if src_h != dst_h:
                            raise RuntimeError("Move verify failed: src/dst hash mismatch")
                        src_path.unlink()
                        tracker.increment("deleted_sources")

                    tracker.mark_processed(file_key)
                    progress(src_size)
                    checkpoint_counter += 1
                    if checkpoint_counter % 50 == 0 and not dry_run:
                        tracker.save()
                    continue

                if src_size != dst_size:
                    n = 1
                    while True:
                        alt = add_suffix(dst_path, n)
                        if not alt.exists():
                            chosen = alt
                            break
                        n += 1

                    if dry_run:
                        log(f"[DRYRUN_RENAME_COPY] {src_path} -> {chosen} (dst exists with different size)")
                        tracker.increment("renamed")
                    else:
                        atomic_copy(src_path, chosen)
                        cached_hash(chosen, cache, write_cache=write_cache)
                        tracker.increment("renamed")

                        if args.move:
                            src_h = cached_hash(src_path, cache, write_cache=write_cache)
                            ch_h = cached_hash(chosen, cache, write_cache=write_cache)
                            if src_h != ch_h:
                                raise RuntimeError("Move verify failed: src/alt hash mismatch")
                            src_path.unlink()
                            tracker.increment("deleted_sources")

                    tracker.mark_processed(file_key)
                    progress(src_size)
                    checkpoint_counter += 1
                    if checkpoint_counter % 50 == 0 and not dry_run:
                        tracker.save()
                    continue

                src_h = cached_hash(src_path, cache, write_cache=write_cache)
                dst_h = cached_hash(dst_path, cache, write_cache=write_cache)

                if src_h == dst_h:
                    tracker.increment("skipped_dup")
                    if args.log_dups:
                        log(f"[DUP] {src_path} == {dst_path}")
                    if args.move and not dry_run:
                        src_path.unlink()
                        tracker.increment("deleted_sources")
                    tracker.mark_processed(file_key)
                    progress(src_size)
                    checkpoint_counter += 1
                    if checkpoint_counter % 50 == 0 and not dry_run:
                        tracker.save()
                    continue

                n = 1
                while True:
                    alt = add_suffix(dst_path, n)
                    if not alt.exists():
                        chosen = alt
                        break
                    n += 1

                if dry_run:
                    log(f"[DRYRUN_RENAME_COPY] {src_path} -> {chosen} (same size, different hash)")
                    tracker.increment("renamed")
                else:
                    atomic_copy(src_path, chosen)
                    alt_h = cached_hash(chosen, cache, write_cache=write_cache)
                    tracker.increment("renamed")

                    if args.move:
                        if src_h != alt_h:
                            raise RuntimeError("Move verify failed: src/alt hash mismatch")
                        src_path.unlink()
                        tracker.increment("deleted_sources")

                tracker.mark_processed(file_key)
                progress(src_size)
                checkpoint_counter += 1
                if checkpoint_counter % 50 == 0 and not dry_run:
                    tracker.save()

            except Exception as e:
                tracker.increment("errors")
                log(f"[ERROR] {src_path} -> {dst_path} : {type(e).__name__}: {e}")
                if args.max_errors > 0 and tracker.stats["errors"] >= args.max_errors:
                    log(f"ABORTED: Maximum error count ({args.max_errors}) reached")
                    print(f"\nERROR: Maximum error count ({args.max_errors}) reached. Aborting.")
                    raise KeyboardInterrupt()

    finally:
        if tqdm_bar is not None:
            tqdm_bar.close()
        cache.close()

    summary = (
        f"Completed. Copied: {tracker.stats['copied']}, Renamed: {tracker.stats['renamed']}, "
        f"Skipped dups: {tracker.stats['skipped_dup']}, Symlinks copied: {tracker.stats['symlinks_copied']}, "
        f"Errors: {tracker.stats['errors']}"
    )
    log(summary)
    print(summary)
    print(f"Log: {log_path}")

    if tracker.stats["errors"] == 0 and (not dry_run):
        tracker.clear()
        print("Checkpoint cleared (run completed successfully)")

    exit_code = 0 if tracker.stats["errors"] == 0 else 2
    if tracker.stats["errors"] != 0 and (not dry_run):
        print(f"Checkpoint saved at: {checkpoint_path} (rerun same command to retry errors)")

    return MergeRunResult(exit_code=exit_code, log_path=log_path, checkpoint_path=checkpoint_path, stats=tracker.stats)


# ==========
# CLI entry
# ==========

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="Source backup root (repeatable)")
    ap.add_argument("--dst", required=True, help="Destination master root")
    ap.add_argument("--log", required=True, help="Log file path")
    ap.add_argument("--cache", required=True, help="SQLite cache file path")
    ap.add_argument("--checkpoint", help="Checkpoint file for resume capability (default: <cache>.checkpoint)")

    ap.add_argument("--move", action="store_true", help="After verified copy/link, delete the source entry.")
    ap.add_argument("--log-dups", action="store_true", help="Also log straight duplicates. Default: only renames/errors.")
    ap.add_argument("--space-margin", type=float, default=0.1, help="Disk space safety margin (default 0.1 = 10%)")
    ap.add_argument("--force-space", action="store_true", help="Proceed even if disk space check fails (NOT recommended).")

    ap.add_argument("--max-errors", type=int, default=100, help="Max errors before abort (0=unlimited).")

    ap.add_argument("--ignore-symlinks", action="store_true", help="Skip symlinks entirely (default is to copy them).")
    ap.add_argument("--exclude-name", action="append", default=[], help="Exact filename to exclude (repeatable).")
    ap.add_argument("--exclude-glob", action="append", default=[], help="Glob to exclude (repeatable).")

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only: do not write destination, do not delete sources, do not write cache/checkpoint.",
    )

    args = ap.parse_args(argv)
    result = _run_with_args(args)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
