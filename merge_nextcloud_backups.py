#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

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
    # "file.txt" -> "file (1).txt" ; "file" -> "file (1)"
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


def iter_files(root: Path, exclude_names: set[str], exclude_globs: list[str]):
    for p in root.rglob("*"):
        try:
            rel = p.relative_to(root)
            if should_exclude(rel, exclude_names, exclude_globs):
                continue
            # Symlink policy: skip (safest)
            if p.is_symlink():
                yield ("symlink", p, rel)
                continue
            if p.is_file():
                yield ("file", p, rel)
        except Exception:
            continue


def format_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="Source backup root (repeatable)")
    ap.add_argument("--dst", required=True, help="Destination master root")
    ap.add_argument("--log", required=True, help="Log file path")
    ap.add_argument("--cache", required=True, help="SQLite cache file path")

    ap.add_argument(
        "--move",
        action="store_true",
        help="After a successful verified copy into dst (or rename target), delete the source file.",
    )
    ap.add_argument(
        "--log-dups",
        action="store_true",
        help="Also log straight duplicates (hash-identical collisions). Default: only renames/errors.",
    )

    ap.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Exact filename to exclude (repeatable). Example: --exclude-name nextcloud.log",
    )
    ap.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Glob to exclude by filename or relative path (repeatable). Example: --exclude-glob '*/cache/*'",
    )

    args = ap.parse_args()

    src_roots = [Path(s).resolve() for s in args.src]
    dst_root = Path(args.dst).resolve()
    log_path = Path(args.log).resolve()
    cache_path = Path(args.cache).resolve()

    dst_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Default excludes: nextcloud-specific + harmless OS noise
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

    def log(line: str) -> None:
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(line.rstrip("\n") + "\n")

    log(f"\n=== Merge run {format_ts()} ===")
    log(f"Sources: {', '.join(str(s) for s in src_roots)}")
    log(f"Destination: {dst_root}")
    log(f"Mode: {'MOVE (copy+verify+delete source)' if args.move else 'COPY'}")
    log(f"Cache DB: {cache_path}")
    log(f"Excludes (names): {sorted(exclude_names)}")
    if exclude_globs:
        log(f"Excludes (globs): {exclude_globs}")

    # Build list + total bytes
    all_items: list[Tuple[str, Path, Path, Path]] = []  # (kind, src_root, src_file, rel)
    total_bytes = 0

    for src_root in src_roots:
        for kind, src_file, rel in iter_files(src_root, exclude_names, exclude_globs):
            all_items.append((kind, src_root, src_file, rel))
            if kind == "file":
                try:
                    total_bytes += src_file.stat().st_size
                except Exception:
                    pass

    # Progress
    if TQDM:
        pbar = TQDM(total=total_bytes, unit="B", unit_scale=True, desc="Merging", smoothing=0.05)
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
                pct = (done / total_bytes * 100) if total_bytes else 100.0
                sys.stdout.write(f"\rProgress: {pct:6.2f}%  ({done}/{total_bytes} bytes)")
                sys.stdout.flush()
                last_print = now

    copied = 0
    skipped_dup = 0
    renamed = 0
    skipped_symlink = 0
    deleted_sources = 0
    errors = 0

    try:
        for kind, src_root, src_path, rel in all_items:
            if kind == "symlink":
                skipped_symlink += 1
                log(f"[SKIP_SYMLINK] {src_path} (rel={rel})")
                continue

            # kind == "file"
            try:
                src_size = src_path.stat().st_size
            except Exception:
                src_size = 0

            dst_path = dst_root / rel

            try:
                # Case 1: destination missing => copy
                if not dst_path.exists():
                    if not args.move:
                        atomic_copy(src_path, dst_path)
                    else:
                        # copy, verify, then delete source
                        atomic_copy(src_path, dst_path)
                    # cache hash of dst
                    _ = cached_hash(dst_path, cache)
                    copied += 1

                    if args.move:
                        # verify by hash before delete
                        src_h = cached_hash(src_path, cache)
                        dst_h = cached_hash(dst_path, cache)
                        if src_h != dst_h:
                            raise RuntimeError("Move verify failed: src/dst hash mismatch")
                        src_path.unlink()
                        deleted_sources += 1

                    progress(src_size)
                    continue

                # Case 2: destination exists
                # Fast check: size
                try:
                    dst_size = dst_path.stat().st_size
                except FileNotFoundError:
                    # race
                    atomic_copy(src_path, dst_path)
                    _ = cached_hash(dst_path, cache)
                    copied += 1

                    if args.move:
                        src_h = cached_hash(src_path, cache)
                        dst_h = cached_hash(dst_path, cache)
                        if src_h != dst_h:
                            raise RuntimeError("Move verify failed: src/dst hash mismatch")
                        src_path.unlink()
                        deleted_sources += 1

                    progress(src_size)
                    continue

                # If same size, hash compare for straight-duplicate
                if dst_size == src_size:
                    src_h = cached_hash(src_path, cache)
                    dst_h = cached_hash(dst_path, cache)
                    if src_h == dst_h:
                        skipped_dup += 1
                        if args.log_dups:
                            log(f"[DUP] {src_path} == {dst_path} (sha256 {src_h})")
                        if args.move:
                            # safe to delete source (identical content already in master)
                            src_path.unlink()
                            deleted_sources += 1
                        progress(src_size)
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
                                # Already present as an alternate name => straight dup
                                skipped_dup += 1
                                if args.log_dups:
                                    log(f"[DUP] {src_path} == {alt} (sha256 {src_h})")
                                if args.move:
                                    src_path.unlink()
                                    deleted_sources += 1
                                chosen = None
                                break
                    except Exception:
                        pass

                    n += 1

                if chosen is not None:
                    atomic_copy(src_path, chosen)
                    _ = cached_hash(chosen, cache)
                    renamed += 1
                    log(f"[RENAME] {src_path} -> {chosen} (conflict with {dst_path}, src_sha256={src_h})")

                    if args.move:
                        dst_h = cached_hash(chosen, cache)
                        if src_h != dst_h:
                            raise RuntimeError("Move verify failed: src/dst hash mismatch")
                        src_path.unlink()
                        deleted_sources += 1

                progress(src_size)

            except Exception as e:
                errors += 1
                log(f"[ERROR] {src_path} -> {dst_path} : {type(e).__name__}: {e}")
                progress(src_size)

    finally:
        if TQDM:
            pbar.close()
        else:
            sys.stdout.write("\n")
        cache.close()

    log(f"--- Summary {format_ts()} ---")
    log(f"Copied: {copied}")
    log(f"Renamed(conflicts): {renamed}")
    log(f"Skipped duplicates: {skipped_dup}")
    log(f"Skipped symlinks: {skipped_symlink}")
    log(f"Deleted sources (move-mode or dup cleanup): {deleted_sources}")
    log(f"Errors: {errors}")

    print("Done.")
    print(f"Copied: {copied}, Renamed: {renamed}, Skipped dups: {skipped_dup}, Errors: {errors}")
    print(f"Log: {log_path}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
