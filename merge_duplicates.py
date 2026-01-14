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


TQDM = try_import_tqdm()


@dataclass(frozen=True)
class FileSig:
    size: int
    mtime_ns: int


class HashCache:
    """SQLite-backed cache: path -> (size, mtime_ns, sha256)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init()

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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_size ON filehash(size)")
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
        if self.checkpoint_path.exists():
            try:
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
                        "processed_files": list(self.processed_files),
                        "stats": self.stats,
                    },
                    f,
                )
            os.replace(tmp, self.checkpoint_path)
        except Exception:
            pass

    def mark_processed(self, file_key: str) -> None:
        self.processed_files.add(file_key)

    def is_processed(self, file_key: str) -> bool:
        return file_key in self.processed_files

    def increment(self, stat: str) -> None:
        if stat in self.stats:
            self.stats[stat] += 1

    def clear(self) -> None:
        try:
            self.checkpoint_path.unlink()
        except Exception:
            pass


def get_disk_space(path: Path) -> Tuple[int, int]:
    stat = shutil.disk_usage(path)
    return stat.free, stat.total


def gib(n: int) -> float:
    return n / (1024**3)


def check_disk_space(dst_root: Path, required_bytes: int, safety_margin: float = 0.1) -> Tuple[bool, str, int, int]:
    """
    Returns (ok, message, free_bytes, required_with_margin).
    """
    free_bytes, _total_bytes = get_disk_space(dst_root)
    required_with_margin = int(required_bytes * (1 + safety_margin))

    if free_bytes < required_with_margin:
        return (
            False,
            f"Insufficient space: {gib(free_bytes):.2f} GiB free, {gib(required_with_margin):.2f} GiB required (with {safety_margin*100:.0f}% margin)",
            free_bytes,
            required_with_margin,
        )

    return (
        True,
        f"Space check OK: {gib(free_bytes):.2f} GiB free, {gib(required_with_margin):.2f} GiB required",
        free_bytes,
        required_with_margin,
    )


def file_sig(p: Path) -> FileSig:
    st = p.stat()
    return FileSig(size=st.st_size, mtime_ns=getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(BUF_SIZE)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cached_hash(p: Path, cache: HashCache) -> str:
    sig = file_sig(p)
    got = cache.get(p, sig)
    if got:
        return got
    digest = sha256_file(p)
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


def iter_items(root: Path, exclude_names: set[str], exclude_globs: list[str]):
    for p in root.rglob("*"):
        try:
            rel = p.relative_to(root)
            if should_exclude(rel, exclude_names, exclude_globs):
                continue
            if p.is_symlink():
                yield ("symlink", p, rel)
                continue
            if p.is_file():
                yield ("file", p, rel)
        except Exception:
            continue


def format_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_symlink(dst: Path, target: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    os.symlink(target, tmp)
    os.replace(tmp, dst)


def main() -> int:
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

    args = ap.parse_args()

    src_roots = [Path(s).resolve() for s in args.src]
    dst_root = Path(args.dst).resolve()
    log_path = Path(args.log).resolve()
    cache_path = Path(args.cache).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else cache_path.with_suffix(".checkpoint")

    dst_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Default excludes: nextcloud-ish + OS noise
    exclude_names = set(args.exclude_name) | {
        "nextcloud.log",
        "owncloud.db",
        ".DS_Store",
        "Thumbs.db",
    }
    exclude_globs = list(args.exclude_glob) + [
        "*.tmp",
        "*.swp",
        "*.swpx",
        "*.swx",
    ]

    cache = HashCache(cache_path)
    tracker = ProgressTracker(checkpoint_path)

    def log(line: str) -> None:
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(line.rstrip("\n") + "\n")

    log(f"\n=== Merge run {format_ts()} ===")
    log(f"Sources: {', '.join(str(s) for s in src_roots)}")
    log(f"Destination: {dst_root}")
    log(f"Mode: {'MOVE' if args.move else 'COPY'}")
    log(f"Symlinks: {'SKIP' if args.ignore_symlinks else 'COPY'}")
    log(f"Cache DB: {cache_path}")
    log(f"Checkpoint: {checkpoint_path}")
    log(f"Max errors: {'unlimited' if args.max_errors == 0 else args.max_errors}")
    log(f"Excludes (names): {sorted(exclude_names)}")
    if exclude_globs:
        log(f"Excludes (globs): {exclude_globs}")

    # Scan sources
    print("Scanning source directories...")
    all_items: list[Tuple[str, Path, Path, Path]] = []  # (kind, src_root, src_path, rel)
    total_bytes_all = 0
    total_bytes_to_process = 0

    # Per-source totals for “best-case vs worst-case” bounds
    per_src_total_bytes = {str(s): 0 for s in src_roots}
    per_src_remaining_bytes = {str(s): 0 for s in src_roots}

    for src_root in src_roots:
        for kind, src_path, rel in iter_items(src_root, exclude_names, exclude_globs):
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

    print(f"Found {len(all_items)} items ({gib(total_bytes_all):.2f} GiB total across all sources)")
    if tracker.processed_files:
        print(f"Resuming: {len(tracker.processed_files)} files already processed")
        print(f"Remaining: {gib(total_bytes_to_process):.2f} GiB to process")

    # Disk space check + bounds
    print("Checking disk space...")
    ok, msg, free_bytes, required_with_margin = check_disk_space(dst_root, total_bytes_to_process, args.space_margin)
    log(f"Disk space check: {msg}")
    print(msg)

    # Bounds: smallest possible final size vs largest possible final size (remaining run)
    # Smallest possible = largest single backup remaining size (if everything overlaps)
    # Largest possible  = sum of all remaining sizes (if nothing overlaps)
    smallest_possible = max(per_src_remaining_bytes.values()) if per_src_remaining_bytes else 0
    largest_possible = total_bytes_to_process

    if not ok:
        print("\nDestination may be too small.")
        print(f"  Smallest possible final (best-case overlap): {gib(smallest_possible):.2f} GiB")
        print(f"  Largest possible final  (worst-case overlap): {gib(largest_possible):.2f} GiB")
        print(f"  Free space now: {gib(free_bytes):.2f} GiB")
        log(
            f"Space bounds (remaining): best_case={smallest_possible} bytes, worst_case={largest_possible} bytes, free={free_bytes} bytes"
        )

        if not args.force_space:
            print("\nABORTING (use --force-space to run anyway).")
            log("ABORTED: insufficient disk space (no --force-space).")
            cache.close()
            return 1
        else:
            print("\nWARNING: Continuing due to --force-space. You may run out of disk mid-run.")
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

            # Symlink handling
            if kind == "symlink":
                if args.ignore_symlinks:
                    tracker.increment("symlinks_skipped")
                    log(f"[SKIP_SYMLINK] {src_path} (rel={rel})")
                    # treat skip as "done" (it was intentionally skipped)
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
                                        if args.log_dups:
                                            log(f"[DUP_SYMLINK] {src_path} == {dst_link} (target={target})")
                                        if args.move:
                                            src_path.unlink()
                                            tracker.increment("deleted_sources")
                                        tracker.mark_processed(file_key)
                                        continue
                                except Exception:
                                    pass

                            n = 1
                            chosen = None
                            while True:
                                alt = add_suffix(dst_link, n)
                                if not alt.exists() and not alt.is_symlink():
                                    chosen = alt
                                    break
                                if alt.is_symlink():
                                    try:
                                        if os.readlink(alt) == target:
                                            tracker.increment("skipped_dup")
                                            if args.log_dups:
                                                log(f"[DUP_SYMLINK] {src_path} == {alt} (target={target})")
                                            if args.move:
                                                src_path.unlink()
                                                tracker.increment("deleted_sources")
                                            chosen = None
                                            break
                                    except Exception:
                                        pass
                                n += 1

                            if chosen is not None:
                                ensure_symlink(chosen, target)
                                tracker.increment("renamed")
                                tracker.increment("symlinks_copied")
                                log(f"[RENAME_SYMLINK] {src_path} -> {chosen} (target={target})")
                                if args.move:
                                    src_path.unlink()
                                    tracker.increment("deleted_sources")
                        else:
                            ensure_symlink(dst_link, target)
                            tracker.increment("symlinks_copied")
                            log(f"[SYMLINK] {src_path} -> {dst_link} (target={target})")
                            if args.move:
                                src_path.unlink()
                                tracker.increment("deleted_sources")

                        tracker.mark_processed(file_key)
                    except Exception as e:
                        tracker.increment("errors")
                        log(f"[ERROR_SYMLINK] {src_path} : {type(e).__name__}: {e}")
                        # NOTE: do NOT mark processed; it will retry next run

                checkpoint_counter += 1
                if checkpoint_counter % 50 == 0:
                    tracker.save()
                continue

            # File handling
            try:
                src_size = src_path.stat().st_size
            except Exception:
                src_size = 0

            dst_path = dst_root / rel

            try:
                # Case 1: destination missing => copy
                if not dst_path.exists():
                    atomic_copy(src_path, dst_path)
                    dst_hash = cached_hash(dst_path, cache)
                    tracker.increment("copied")

                    if args.move:
                        src_h = cached_hash(src_path, cache)
                        if src_h != dst_hash:
                            raise RuntimeError("Move verify failed: src/dst hash mismatch")
                        src_path.unlink()
                        tracker.increment("deleted_sources")

                    # success => checkpoint + progress
                    tracker.mark_processed(file_key)
                    progress(src_size)

                    checkpoint_counter += 1
                    if checkpoint_counter % 50 == 0:
                        tracker.save()
                    continue

                # Case 2: destination exists
                try:
                    dst_size = dst_path.stat().st_size
                except FileNotFoundError:
                    atomic_copy(src_path, dst_path)
                    dst_hash = cached_hash(dst_path, cache)
                    tracker.increment("copied")

                    if args.move:
                        src_h = cached_hash(src_path, cache)
                        if src_h != dst_hash:
                            raise RuntimeError("Move verify failed: src/dst hash mismatch")
                        src_path.unlink()
                        tracker.increment("deleted_sources")

                    tracker.mark_processed(file_key)
                    progress(src_size)

                    checkpoint_counter += 1
                    if checkpoint_counter % 50 == 0:
                        tracker.save()
                    continue

                # If same size, hash compare for straight-duplicate
                if dst_size == src_size:
                    src_h = cached_hash(src_path, cache)
                    dst_h = cached_hash(dst_path, cache)
                    if src_h == dst_h:
                        tracker.increment("skipped_dup")
                        if args.log_dups:
                            log(f"[DUP] {src_path} == {dst_path} (sha256 {src_h})")
                        if args.move:
                            src_path.unlink()
                            tracker.increment("deleted_sources")

                        tracker.mark_processed(file_key)
                        progress(src_size)

                        checkpoint_counter += 1
                        if checkpoint_counter % 50 == 0:
                            tracker.save()
                        continue

                # Otherwise conflict: find unique "(n)" name, but dedupe against existing alts too
                src_h = cached_hash(src_path, cache)
                n = 1
                chosen: Optional[Path] = None
                while True:
                    alt = add_suffix(dst_path, n)

                    if not alt.exists():
                        chosen = alt
                        break

                    try:
                        if alt.stat().st_size == src_size:
                            alt_h = cached_hash(alt, cache)
                            if alt_h == src_h:
                                tracker.increment("skipped_dup")
                                if args.log_dups:
                                    log(f"[DUP] {src_path} == {alt} (sha256 {src_h})")
                                if args.move:
                                    src_path.unlink()
                                    tracker.increment("deleted_sources")
                                chosen = None
                                break
                    except Exception:
                        pass

                    n += 1

                if chosen is not None:
                    atomic_copy(src_path, chosen)
                    dst_hash = cached_hash(chosen, cache)
                    tracker.increment("renamed")
                    log(f"[RENAME] {src_path} -> {chosen} (conflict with {dst_path}, src_sha256={src_h})")

                    if args.move:
                        if src_h != dst_hash:
                            raise RuntimeError("Move verify failed: src/dst hash mismatch")
                        src_path.unlink()
                        tracker.increment("deleted_sources")

                tracker.mark_processed(file_key)
                progress(src_size)

                checkpoint_counter += 1
                if checkpoint_counter % 50 == 0:
                    tracker.save()

            except Exception as e:
                tracker.increment("errors")
                log(f"[ERROR] {src_path} -> {dst_path} : {type(e).__name__}: {e}")
                # NOTE: do NOT mark processed; it will retry next run

                if args.max_errors > 0 and tracker.stats["errors"] >= args.max_errors:
                    log(f"ABORTED: Maximum error count ({args.max_errors}) reached")
                    print(f"\nERROR: Maximum error count ({args.max_errors}) reached. Aborting.")
                    raise KeyboardInterrupt()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress has been saved.")
        log(f"INTERRUPTED at {format_ts()}")
        tracker.save()
    finally:
        if TQDM:
            pbar.close()
        else:
            sys.stdout.write("\n")
        tracker.save()
        cache.close()

    log(f"--- Summary {format_ts()} ---")
    log(f"Copied: {tracker.stats['copied']}")
    log(f"Renamed(conflicts): {tracker.stats['renamed']}")
    log(f"Skipped duplicates: {tracker.stats['skipped_dup']}")
    log(f"Symlinks copied: {tracker.stats['symlinks_copied']}")
    log(f"Symlinks skipped: {tracker.stats['symlinks_skipped']}")
    log(f"Deleted sources: {tracker.stats['deleted_sources']}")
    log(f"Errors: {tracker.stats['errors']}")

    print("\nDone.")
    print(
        f"Copied: {tracker.stats['copied']}, Renamed: {tracker.stats['renamed']}, "
        f"Skipped dups: {tracker.stats['skipped_dup']}, Symlinks copied: {tracker.stats['symlinks_copied']}, "
        f"Errors: {tracker.stats['errors']}"
    )
    print(f"Log: {log_path}")

    if tracker.stats["errors"] == 0:
        tracker.clear()
        print("Checkpoint cleared (run completed successfully)")
        return 0

    print(f"Checkpoint saved at: {checkpoint_path} (rerun same command to retry errors)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
