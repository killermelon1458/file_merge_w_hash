# File Merge with Hash-Based Deduplication

This repository contains utilities for **safely merging multiple backup directories into a single master copy**, while:

* Detecting **true duplicates using SHA-256 hashes**
* Preserving **directory structure**
* Handling **filename conflicts deterministically**
* Supporting **resume / crash recovery**
* Avoiding unnecessary re-hashing via a **SQLite cache**
* Optionally **moving (deleting) sources only after verification**

The primary script is:

* `merge_duplicates.py` — the production-grade merge engine

---

## Problem This Solves

When you have **multiple full backups** (e.g. repeated Nextcloud backups, rsync snapshots, manual copies), simple tools like `rsync` or `cp` cannot safely answer:

* Are these files *actually* identical?
* What if names collide but contents differ?
* What if a run is interrupted halfway through?
* How much space will I need *before* starting?
* Can I delete sources *only after* verifying integrity?

This script answers all of those **explicitly and safely**.

---

## Key Features

### ✅ Content-Based Deduplication

* Uses **SHA-256 hashes**, not filenames or timestamps
* Files with identical content are copied **once**
* Subsequent duplicates are skipped safely

### ✅ Deterministic Conflict Handling

If two files share a path but differ in content:

* Keeps the first
* Writes the next as:

  ```
  filename (1).ext
  filename (2).ext
  ```
* Also deduplicates against these alternates

### ✅ Resume Support (Crash-Safe)

* Progress is tracked in a **checkpoint file**
* Only **successful operations** are recorded
* If interrupted, re-running the same command resumes cleanly

### ✅ SQLite Hash Cache

* Stores `(path, size, mtime_ns) → sha256`
* Avoids re-hashing unchanged files
* Speeds up large, repeated runs dramatically

### ✅ Disk Space Safety Checks

Before copying, the script:

* Calculates remaining bytes
* Applies a configurable safety margin (default **10%**)
* Computes **best-case vs worst-case** final size bounds
* Aborts unless `--force-space` is explicitly provided

### ✅ Symlink Awareness

* Symlinks are copied by default
* Identical symlinks are deduplicated
* Can be skipped entirely with `--ignore-symlinks`

### ✅ Copy or Move Mode

* Default: **COPY** (non-destructive)
* Optional: **MOVE** (`--move`)

  * Source is deleted **only after hash verification**

---

## Usage

### Basic Example (Recommended)

Merge multiple backups into a master directory:

```bash
python3 merge_duplicates.py \
  --src /mnt/backups/backup_1 \
  --src /mnt/backups/backup_2 \
  --src /mnt/backups/backup_3 \
  --dst /mnt/master_backup \
  --log ~/logs/merge.log \
  --cache ~/cache/merge_hashes.sqlite
```

This will:

* Copy unique files into `/mnt/master_backup`
* Skip true duplicates
* Rename conflicts safely
* Allow resuming if interrupted

---

### Resume a Failed or Interrupted Run

Just run **the exact same command again**.

The script will:

* Load the checkpoint
* Skip already-successful files
* Retry only failed operations

---

### Move Mode (Destructive, Verified)

```bash
python3 merge_duplicates.py \
  --src /mnt/backups/backup_1 \
  --src /mnt/backups/backup_2 \
  --dst /mnt/master_backup \
  --log merge.log \
  --cache merge.sqlite \
  --move
```

⚠️ **Important:**
Files are deleted from the source **only after** a successful hash match.

---

## Command-Line Options

| Option              | Description                                 |
| ------------------- | ------------------------------------------- |
| `--src`             | Source directory (repeatable)               |
| `--dst`             | Destination master directory                |
| `--log`             | Log file path                               |
| `--cache`           | SQLite hash cache path                      |
| `--checkpoint`      | Override checkpoint path                    |
| `--move`            | Delete source files after verified copy     |
| `--log-dups`        | Log duplicate files explicitly              |
| `--space-margin`    | Disk safety margin (default 0.1 = 10%)      |
| `--force-space`     | Ignore disk space failure (not recommended) |
| `--max-errors`      | Abort after N errors (0 = unlimited)        |
| `--ignore-symlinks` | Skip symlinks entirely                      |
| `--exclude-name`    | Exact filename to exclude (repeatable)      |
| `--exclude-glob`    | Glob pattern to exclude (repeatable)        |

---

## Logging & Observability

The log file records:

* Every rename
* Every duplicate detection
* Every error (with exception type)
* A full summary at the end

Example log entries:

```
[RENAME] file.txt -> file (1).txt
[DUP] fileA.bin == fileB.bin (sha256 ...)
[ERROR] /src/file -> /dst/file : PermissionError
```

---

## Safety Guarantees

This script is designed to be **safe by default**:

* No source deletion unless `--move` is used
* No deletion without hash verification
* No silent overwrites
* No partial-file writes (atomic temp files used)
* Resume never replays failed operations blindly

If something goes wrong, **rerunning is safe**.

---

## Intended Use Cases

* Merging multiple Nextcloud backups
* Consolidating rsync snapshots
* Deduplicating archival drives
* Building a canonical “master” dataset from redundant copies

---

## License

MIT 

---

## Notes

This script is intentionally explicit and conservative.
Speed is secondary to **correctness, auditability, and safety**.

If you are looking for a “quick copy tool,” use `rsync`.
If you want **provable correctness**, use this.
