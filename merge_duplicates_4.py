#!/usr/bin/env python3
"""
merge_duplicates.py

Merges one or more source directory trees into a single destination "master copy"
directory.

Core behaviors:
- If destination path doesn't exist: copy file into place
- If destination exists:
  - If identical (size+sha256): treat as duplicate (skip)
  - Else: copy as "<name> (n).ext" (rename-on-conflict)
- Optional move mode deletes source only after verification
- Symlinks: copy by default, or ignore with --ignore-symlinks
- Resume support via checkpoint (only marks SUCCESSFUL items)
- Hash caching via SQLite (path + size + mtime_ns => sha256) to speed up repeats
- Works as both a CLI and an importable engine for backup_runner.py (Option A)

Output improvements (per your latest asks):
- Timestamps on all non-progress prints
- "Bounds (guesses)" computed using only folder sizes:
    min_guess = max(0, max(source_remaining) - destination_current_size)
    max_guess = sum(source_remaining)
  (This is a GUESS; does not account for duplicates/overlap precisely.)
- Progress bar retained (tqdm if installed; otherwise a simple terminal progress line)
"""

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
from typing import Optional, Tuple, List, Dict, Set


BUF_SIZE = 1024 * 1024  # 1 MiB


# -------------------------
# Optional tqdm
# -------------------------
def try_import_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


TQDM = try_import_tqdm()


# -------------------------
# Timestamped printing / logging helpers
# -------------------------
def format_ts(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def print_ts(msg: str) -> None:
    # Timestamp every non-progress line; keep progress bar clean.
    print(f"[{format_ts()}] {msg}", flush=True)


def safe_write_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")

    return ":".join(parts)


# -------------------------
# Hash cache
# -------------------------
@dataclass(frozen=True)
class FileSig:
    size: int
    mtime_ns: int


class HashCache:
    """SQLite-backed cache: path -> (size, mtime_ns, sha256).

    If readonly=True and DB doesn't exist, uses in-memory DB.
    """

    def __init__(self, db_path: Path, *, readonly: bool = False, allow_writes: bool = True):
        self.db_path = db_path
        self.readonly = readonly
        self.allow_writes = allow_writes and (not readonly)

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
        cur.execute("SELECT sha256, size, mtime_ns FROM filehash WHERE path = ?", (str(path),))
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


def file_sig(p: Path) -> FileSig:
    st = p.stat()
    # st_mtime_ns exists on modern Python; fall back if needed
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    return FileSig(size=st.st_size, mtime_ns=mtime_ns)


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
    sig = file_sig(p)
    got = cache.get(p, sig)
    if got:
        return got
    digest = sha256_file(p)
    if write_cache:
        cache.put(p, sig, digest)
    return digest


# -------------------------
# Checkpoint / resume tracker
# -------------------------
class ProgressTracker:
    """Tracks progress and allows resume from checkpoint. Only records SUCCESSFUL work."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.processed_files: Set[str] = set()
        self.stats = {
            "copied": 0,
            "copied_bytes": 0,
            "renamed": 0,
            "renamed_bytes": 0,
            "skipped_dup": 0,
            "skipped_bytes": 0,
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
            # If corrupted, start fresh (do not crash)
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


class DryProgressTracker(ProgressTracker):
    """Never reads/writes checkpoints (for dry-run)."""

    def load(self) -> None:  # type: ignore[override]
        self.processed_files = set()

    def save(self) -> None:  # type: ignore[override]
        return

    def clear(self) -> None:  # type: ignore[override]
        self.processed_files = set()


# -------------------------
# File operations
# -------------------------
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


def ensure_symlink(dst: Path, target: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    os.symlink(target, tmp)
    os.replace(tmp, dst)


# -------------------------
# Excludes + iterators
# -------------------------
def should_exclude(rel_path: Path, exclude_names: Set[str], exclude_globs: List[str]) -> bool:
    name = rel_path.name
    if name in exclude_names:
        return True
    rel_str = str(rel_path).replace("\\", "/")
    for pat in exclude_globs:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_str, pat):
            return True
    return False


def iter_items(root: Path, exclude_names: Set[str], exclude_globs: List[str], *, ignore_symlinks: bool):
    """Yields tuples: (kind, abs_path, rel_path) where kind is 'file' or 'symlink'."""
    # os.walk is generally faster than rglob on Windows for large trees
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirp = Path(dirpath)
        rel_dir = dirp.relative_to(root)

        # Filter directory names in-place (prevents descending into excluded dirs)
        keep_dirnames = []
        for d in dirnames:
            rel = rel_dir / d
            if should_exclude(rel, exclude_names, exclude_globs):
                continue
            keep_dirnames.append(d)
        dirnames[:] = keep_dirnames

        # Files
        for fn in filenames:
            rel = rel_dir / fn
            if should_exclude(rel, exclude_names, exclude_globs):
                continue
            p = dirp / fn
            try:
                if p.is_symlink():
                    if ignore_symlinks:
                        continue
                    yield ("symlink", p, rel)
                else:
                    yield ("file", p, rel)
            except Exception:
                continue


# -------------------------
# Disk space + size helpers
# -------------------------
def get_disk_space_free(path: Path) -> int:
    return shutil.disk_usage(str(path)).free


def gib(nbytes: int) -> float:
    return nbytes / (1024 ** 3)

def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def check_disk_space(dst_root: Path, bytes_to_copy: int, margin: float) -> Tuple[bool, str, int, int]:
    free = get_disk_space_free(dst_root)
    required = int(bytes_to_copy * (1.0 + margin))
    ok = free >= required
    msg = (
        f"Free: {gib(free):.2f} GiB, need: {gib(required):.2f} GiB "
        f"(margin {margin*100:.0f}%) -> {'OK' if ok else 'NOT OK'}"
    )
    return ok, msg, free, required


def tree_size_bytes(root: Path) -> int:
    """Compute total size of all regular files under root (best-effort; skips unreadables)."""
    if not root.exists():
        return 0
    total = 0
    stack = [root]
    while stack:
        p = stack.pop()
        try:
            with os.scandir(p) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


# =========================
# Importable API (Option A)
# =========================
@dataclass
class MergeRunConfig:
    src_roots: List[Path]
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
    exclude_names: List[str] = None  # type: ignore
    exclude_globs: List[str] = None  # type: ignore
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
    stats: Dict[str, int]
    duration_seconds: float


def run_merge(cfg: MergeRunConfig) -> MergeRunResult:
    """Preferred entrypoint for backup_runner.py."""
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


# -------------------------
# Core engine (shared by CLI + API)
# -------------------------
def _run_with_args(args: argparse.Namespace) -> MergeRunResult:
    start_time = time.time()

    src_roots = [Path(s).resolve() for s in args.src]
    dst_root = Path(args.dst).resolve()
    log_path = Path(args.log).resolve()
    cache_path = Path(args.cache).resolve()
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if getattr(args, "checkpoint", None)
        else cache_path.with_suffix(".checkpoint")
    )

    dst_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Excludes
    exclude_names: Set[str] = set(getattr(args, "exclude_name", []) or []) | {
        ".DS_Store",
        "Thumbs.db",
    }
    exclude_globs: List[str] = list(getattr(args, "exclude_glob", []) or []) + [
        "*.tmp",
        "*.swp",
        "*.swpx",
        "*.swx",
    ]

    # Dry-run: read-only cache if possible, and do not write checkpoints.
    dry_run = bool(getattr(args, "dry_run", False))
    cache = HashCache(cache_path, readonly=dry_run, allow_writes=not dry_run)
    tracker: ProgressTracker = DryProgressTracker(checkpoint_path) if dry_run else ProgressTracker(checkpoint_path)

    def log(line: str) -> None:
        safe_write_line(log_path, f"[{format_ts()}] {line}")

    # Header
    log("=== Merge run start ===")
    log(f"Sources: {', '.join(str(s) for s in src_roots)}")
    log(f"Destination: {dst_root}")
    log(f"Mode: {'MOVE' if args.move else 'COPY'}")
    log(f"Symlinks: {'SKIP' if args.ignore_symlinks else 'COPY'}")
    log(f"Dry run: {dry_run}")
    log(f"Cache DB: {cache_path}")
    log(f"Checkpoint: {checkpoint_path}")
    log(f"Max errors: {'unlimited' if args.max_errors == 0 else args.max_errors}")
    log(f"Excludes (names): {sorted(exclude_names)}")
    if exclude_globs:
        log(f"Excludes (globs): {exclude_globs}")

    print_ts("Scanning source directories...")

    # Scan sources
    all_items: List[Tuple[str, Path, Path, Path]] = []  # (kind, src_root, src_path, rel)
    total_bytes_all = 0
    total_bytes_to_process = 0

    per_src_total_bytes: Dict[str, int] = {str(s): 0 for s in src_roots}
    per_src_remaining_bytes: Dict[str, int] = {str(s): 0 for s in src_roots}

    for src_root in src_roots:
        for kind, src_path, rel in iter_items(
            src_root,
            exclude_names,
            exclude_globs,
            ignore_symlinks=bool(args.ignore_symlinks),
        ):
            all_items.append((kind, src_root, src_path, rel))
            if kind == "file":
                try:
                    sz = src_path.stat().st_size
                except Exception:
                    sz = 0
                total_bytes_all += sz
                per_src_total_bytes[str(src_root)] += sz

                file_key = f"{src_root}::{rel}"
                if not tracker.is_processed(file_key):
                    total_bytes_to_process += sz
                    per_src_remaining_bytes[str(src_root)] += sz

    print_ts(f"Found {len(all_items)} items ({gib(total_bytes_all):.2f} GiB total across all sources)")
    if tracker.processed_files:
        print_ts(f"Resuming: {len(tracker.processed_files)} files already processed")
        print_ts(f"Remaining: {gib(total_bytes_to_process):.2f} GiB to process")

    # Disk space check
    print_ts("Checking disk space...")
    ok, msg, free_bytes, required_with_margin = check_disk_space(dst_root, total_bytes_to_process, float(args.space_margin))
    log(f"Disk space check: {msg}")
    print_ts(msg)

    # Bounds (GUESS) based on folder sizes
    # min_guess = max(0, max(src_remaining) - dest_current_size)
    # max_guess = sum(src_remaining)
    dest_size = tree_size_bytes(dst_root)
    src_remaining_sum = total_bytes_to_process
    src_remaining_max = max(per_src_remaining_bytes.values()) if per_src_remaining_bytes else 0
    min_guess = max(0, src_remaining_max - dest_size)
    max_guess = src_remaining_sum

    bounds_msg = (
        "Bounds (guesses based on folder sizes): "
        f"min possible add ≈ {gib(min_guess):.2f} GiB; "
        f"max possible add ≈ {gib(max_guess):.2f} GiB. "
        f"(Dest size={gib(dest_size):.2f} GiB, Src remaining max={gib(src_remaining_max):.2f} GiB, "
        f"Src remaining sum={gib(src_remaining_sum):.2f} GiB)"
    )
    log(bounds_msg)
    print_ts(bounds_msg)

    if not ok:
        print_ts("Destination may be too small.")
        if not bool(args.force_space):
            print_ts("ABORTING (use --force-space to run anyway).")
            log("ABORTED: insufficient disk space (no --force-space).")
            cache.close()
            # Return result in a consistent shape
            return MergeRunResult(
                exit_code=1,
                log_path=log_path,
                checkpoint_path=checkpoint_path,
                stats=tracker.stats,
                duration_seconds=time.time() - start_time,
            )
        else:
            print_ts("WARNING: Continuing due to --force-space. You may run out of disk mid-run.")
            log("WARNING: continuing despite disk space check failure due to --force-space.")

    # Progress bar
    if TQDM:
        pbar = TQDM(total=total_bytes_to_process, unit="B", unit_scale=True, desc="Merging", smoothing=0.05)

        def progress(n: int) -> None:
            pbar.update(n)
    else:
        done = 0
        last_print = time.time()

        def progress(n: int) -> None:
            nonlocal done, last_print
            done += n
            now = time.time()
            if now - last_print >= 0.2:
                pct = (done / total_bytes_to_process * 100) if total_bytes_to_process else 100.0
                sys.stdout.write(f"\rProgress: {pct:6.2f}%  ({done}/{total_bytes_to_process} bytes)")
                sys.stdout.flush()
                last_print = now

    checkpoint_counter = 0

    try:
        for kind, src_root, src_path, rel in all_items:
            file_key = f"{src_root}::{rel}"

            # Skip already-successful work
            if tracker.is_processed(file_key):
                continue

            # --- Symlink handling ---
            if kind == "symlink":
                # (If ignore_symlinks=True, iter_items won’t yield symlinks; this is just defensive.)
                if bool(args.ignore_symlinks):
                    tracker.increment("symlinks_skipped")
                    log(f"[SKIP_SYMLINK] {src_path} (rel={rel})")
                    tracker.mark_processed(file_key)
                else:
                    try:
                        target = os.readlink(src_path)
                        dst_link = dst_root / rel

                        # If exists and identical, skip; else pick alt name
                        if dst_link.exists() or dst_link.is_symlink():
                            if dst_link.is_symlink():
                                try:
                                    if os.readlink(dst_link) == target:
                                        tracker.increment("skipped_dup")
                                        tracker.stats["skipped_bytes"] += 0


                                        if bool(args.log_dups):
                                            log(f"[DUP_SYMLINK] {src_path} == {dst_link} (target={target})")
                                        if bool(args.move) and (not dry_run):
                                            src_path.unlink()
                                            tracker.increment("deleted_sources")
                                        tracker.mark_processed(file_key)
                                        continue
                                except Exception:
                                    pass

                            n = 1
                            chosen: Optional[Path] = None
                            while True:
                                alt = add_suffix(dst_link, n)
                                if not alt.exists() and not alt.is_symlink():
                                    chosen = alt
                                    break
                                if alt.is_symlink():
                                    try:
                                        if os.readlink(alt) == target:
                                            tracker.increment("skipped_dup")
                                            tracker.stats["skipped_bytes"] += 0

                                            if bool(args.log_dups):
                                                log(f"[DUP_SYMLINK] {src_path} == {alt} (target={target})")
                                            if bool(args.move) and (not dry_run):
                                                src_path.unlink()
                                                tracker.increment("deleted_sources")
                                            chosen = None
                                            break
                                    except Exception:
                                        pass
                                n += 1

                            if chosen is not None:
                                if not dry_run:
                                    ensure_symlink(chosen, target)
                                tracker.increment("renamed")
                                tracker.stats["renamed_bytes"] += src_size

                                tracker.increment("symlinks_copied")
                                log(f"[RENAME_SYMLINK] {src_path} -> {chosen} (target={target})")
                                if bool(args.move) and (not dry_run):
                                    src_path.unlink()
                                    tracker.increment("deleted_sources")
                        else:
                            if not dry_run:
                                ensure_symlink(dst_link, target)
                            tracker.increment("symlinks_copied")
                            log(f"[SYMLINK] {src_path} -> {dst_link} (target={target})")
                            if bool(args.move) and (not dry_run):
                                src_path.unlink()
                                tracker.increment("deleted_sources")

                        tracker.mark_processed(file_key)

                    except Exception as e:
                        tracker.increment("errors")
                        log(f"[ERROR_SYMLINK] {src_path} : {type(e).__name__}: {e}")
                        # do NOT mark processed; it will retry next run

                checkpoint_counter += 1
                if (not dry_run) and (checkpoint_counter % 50 == 0):
                    tracker.save()
                continue

            # --- File handling ---
            try:
                src_size = src_path.stat().st_size
            except Exception:
                src_size = 0

            dst_path = dst_root / rel

            try:
                # Case 1: destination missing => copy
                if not dst_path.exists():
                    if not dry_run:
                        atomic_copy(src_path, dst_path)
                        # hash destination to populate cache
                        _ = cached_hash(dst_path, cache, write_cache=True)
                    tracker.increment("copied")
                    tracker.stats["copied_bytes"] += src_size

                    if bool(args.move) and (not dry_run):
                        # Verify move
                        src_h = cached_hash(src_path, cache, write_cache=True)
                        dst_h = cached_hash(dst_path, cache, write_cache=True)
                        if src_h != dst_h:
                            raise RuntimeError("Move verify failed: src/dst hash mismatch")
                        src_path.unlink()
                        tracker.increment("deleted_sources")

                    tracker.mark_processed(file_key)
                    progress(src_size)

                    checkpoint_counter += 1
                    if (not dry_run) and (checkpoint_counter % 50 == 0):
                        tracker.save()
                    continue

                # Case 2: destination exists
                try:
                    dst_size = dst_path.stat().st_size
                except FileNotFoundError:
                    if not dry_run:
                        atomic_copy(src_path, dst_path)
                        _ = cached_hash(dst_path, cache, write_cache=True)
                    tracker.increment("copied")
                    tracker.stats["copied_bytes"] += src_size

                    if bool(args.move) and (not dry_run):
                        src_h = cached_hash(src_path, cache, write_cache=True)
                        dst_h = cached_hash(dst_path, cache, write_cache=True)
                        if src_h != dst_h:
                            raise RuntimeError("Move verify failed: src/dst hash mismatch")
                        src_path.unlink()
                        tracker.increment("deleted_sources")

                    tracker.mark_processed(file_key)
                    progress(src_size)

                    checkpoint_counter += 1
                    if (not dry_run) and (checkpoint_counter % 50 == 0):
                        tracker.save()
                    continue

                # Same size => hash compare for duplicate
                if dst_size == src_size:
                    src_h = cached_hash(src_path, cache, write_cache=(not dry_run))
                    dst_h = cached_hash(dst_path, cache, write_cache=(not dry_run))
                    if src_h == dst_h:
                        tracker.increment("skipped_dup")
                        tracker.stats["skipped_bytes"] += src_size

                        if bool(args.log_dups):
                            log(f"[DUP] {src_path} == {dst_path} (sha256 {src_h})")
                        if bool(args.move) and (not dry_run):
                            src_path.unlink()
                            tracker.increment("deleted_sources")

                        tracker.mark_processed(file_key)
                        progress(src_size)

                        checkpoint_counter += 1
                        if (not dry_run) and (checkpoint_counter % 50 == 0):
                            tracker.save()
                        continue

                # Conflict: find unique "(n)" name, but dedupe against existing alts too
                src_h = cached_hash(src_path, cache, write_cache=(not dry_run))
                n = 1
                chosen: Optional[Path] = None
                while True:
                    alt = add_suffix(dst_path, n)

                    if not alt.exists():
                        chosen = alt
                        break

                    try:
                        if alt.stat().st_size == src_size:
                            alt_h = cached_hash(alt, cache, write_cache=(not dry_run))
                            if alt_h == src_h:
                                tracker.increment("skipped_dup")
                                tracker.stats["skipped_bytes"] += src_size

                                if bool(args.log_dups):
                                    log(f"[DUP] {src_path} == {alt} (sha256 {src_h})")
                                if bool(args.move) and (not dry_run):
                                    src_path.unlink()
                                    tracker.increment("deleted_sources")
                                chosen = None
                                break
                    except Exception:
                        pass

                    n += 1

                if chosen is not None:
                    if not dry_run:
                        atomic_copy(src_path, chosen)
                        dst_h = cached_hash(chosen, cache, write_cache=True)
                    else:
                        dst_h = src_h  # meaningless in dry-run; keeps logic consistent
                    tracker.increment("renamed")
                    tracker.stats["renamed_bytes"] += src_size

                    log(f"[RENAME] {src_path} -> {chosen} (conflict with {dst_path}, src_sha256={src_h})")

                    if bool(args.move) and (not dry_run):
                        if src_h != dst_h:
                            raise RuntimeError("Move verify failed: src/dst hash mismatch")
                        src_path.unlink()
                        tracker.increment("deleted_sources")

                tracker.mark_processed(file_key)
                progress(src_size)

                checkpoint_counter += 1
                if (not dry_run) and (checkpoint_counter % 50 == 0):
                    tracker.save()

            except Exception as e:
                tracker.increment("errors")
                log(f"[ERROR] {src_path} -> {dst_path} : {type(e).__name__}: {e}")
                # do NOT mark processed; it will retry next run

                if int(args.max_errors) > 0 and int(tracker.stats["errors"]) >= int(args.max_errors):
                    log(f"ABORTED: Maximum error count ({args.max_errors}) reached")
                    print_ts(f"ERROR: Maximum error count ({args.max_errors}) reached. Aborting.")
                    raise KeyboardInterrupt()

    except KeyboardInterrupt:
        print_ts("Interrupted! Progress has been saved.")
        log("INTERRUPTED by user/abort")
        if not dry_run:
            tracker.save()
        exit_code = 130
    finally:
        if TQDM:
            try:
                pbar.close()
            except Exception:
                pass
        else:
            sys.stdout.write("\n")
        if not dry_run:
            tracker.save()
        cache.close()

    duration = time.time() - start_time

    # Summary
    summary_line = (
        f"Completed. Copied: {tracker.stats['copied']}, Renamed: {tracker.stats['renamed']}, "
        f"Skipped dups: {tracker.stats['skipped_dup']}, Symlinks copied: {tracker.stats['symlinks_copied']}, "
        f"Errors: {tracker.stats['errors']}"
    )
    log("--- Summary ---")
    log(summary_line)
    log(f"Duration: {format_duration(duration)}")

    log(f"Log file: {log_path}")
    log(f"Checkpoint: {checkpoint_path}")
    log(
    "Bytes summary: "
    f"copied={format_bytes(tracker.stats['copied_bytes'])}, "
    f"renamed={format_bytes(tracker.stats['renamed_bytes'])}, "
    f"skipped={format_bytes(tracker.stats['skipped_bytes'])}"
    )
    print_ts(
    "Bytes summary: "
    f"copied={format_bytes(tracker.stats['copied_bytes'])}, "
    f"renamed={format_bytes(tracker.stats['renamed_bytes'])}, "
    f"skipped={format_bytes(tracker.stats['skipped_bytes'])}"
    )

    print_ts(summary_line)
    print_ts(f"Log: {log_path}")
    print_ts(f"Duration: {format_duration(duration)}")


    # Clear checkpoint if everything succeeded (and not dry-run)
    if tracker.stats["errors"] == 0 and not dry_run:
        tracker.clear()
        print_ts("Checkpoint cleared (run completed successfully)")
        exit_code = 0
    else:
        # If dry-run, never clear; if errors, preserve
        if dry_run:
            print_ts(f"Dry-run complete (checkpoint not modified).")
            exit_code = 0 if tracker.stats["errors"] == 0 else 2
        else:
            if tracker.stats["errors"] == 0:
                exit_code = 0
            else:
                print_ts(f"Checkpoint saved at: {checkpoint_path} (rerun same command to retry errors)")
                exit_code = 2

    return MergeRunResult(
        exit_code=exit_code,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
        stats=dict(tracker.stats),
        duration_seconds=duration,
    )


# -------------------------
# CLI
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Merge duplicate-aware backups into a master destination folder.")
    ap.add_argument("--src", action="append", required=True, help="Source root (repeatable)")
    ap.add_argument("--dst", required=True, help="Destination root")
    ap.add_argument("--log", required=True, help="Log file path")
    ap.add_argument("--cache", required=True, help="SQLite cache file path")
    ap.add_argument("--checkpoint", help="Checkpoint file path (default: <cache>.checkpoint)")

    ap.add_argument("--move", action="store_true", help="Delete source after verified copy/link.")
    ap.add_argument("--log-dups", action="store_true", help="Log duplicates (default logs only renames/errors).")
    ap.add_argument("--space-margin", type=float, default=0.1, help="Disk space safety margin (default 0.1 = 10%).")
    ap.add_argument("--force-space", action="store_true", help="Proceed even if disk space check fails.")
    ap.add_argument("--max-errors", type=int, default=100, help="Max errors before abort (0 = unlimited).")
    ap.add_argument("--ignore-symlinks", action="store_true", help="Skip symlinks (default is to copy).")

    ap.add_argument("--exclude-name", action="append", default=[], help="Exact name to exclude (repeatable).")
    ap.add_argument("--exclude-glob", action="append", default=[], help="Glob to exclude (repeatable).")

    ap.add_argument("--dry-run", action="store_true", help="Dry-run (no writes; cache opened read-only if possible).")

    args = ap.parse_args()

    cfg = MergeRunConfig(
        src_roots=[Path(s) for s in args.src],
        dst_root=Path(args.dst),
        log_path=Path(args.log),
        cache_path=Path(args.cache),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        move=bool(args.move),
        log_dups=bool(args.log_dups),
        space_margin=float(args.space_margin),
        force_space=bool(args.force_space),
        max_errors=int(args.max_errors),
        ignore_symlinks=bool(args.ignore_symlinks),
        exclude_names=list(args.exclude_name or []),
        exclude_globs=list(args.exclude_glob or []),
        dry_run=bool(args.dry_run),
    )

    result = run_merge(cfg)
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
