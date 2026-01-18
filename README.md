# file_merge_w_hash

A robust, duplicate-aware **backup and merge system** written in Python. This project is designed to safely consolidate multiple source directories into a single destination, while avoiding duplicate files, supporting resumable runs, and providing detailed logging and notifications.

This repo is the evolution of several earlier scripts and now represents the **v4 generation** of the backup runner and merge engine.

---

## ✨ Key Features

* 🔁 **Duplicate-aware merging**
  Files are compared using size + SHA-256 hashes to avoid storing duplicates.

* 📂 **Multiple sources → one destination**
  Merge many directories into a single master copy.

* ▶️ **Resume support**
  Interrupted runs can resume using checkpoint files.

* ⚡ **Hash caching (SQLite)**
  Greatly speeds up repeated runs on mostly-unchanged data sets.

* 🧪 **Dry-run mode**
  See what *would* happen without copying or deleting anything.

* 📝 **Per-job logging**
  Timestamped logs with byte counts and summaries.

* 📧 **Optional email notifications**
  Send success/failure notifications using environment-based credentials.

* 🛡️ **Disk space safety checks**
  Prevents running out of space mid-backup (with optional override).

* 🧩 **Config-driven**
  All behavior is controlled via an INI-style config file.

---

## 📁 Repository Layout

```
file_merge_w_hash/
├── backup_runner_4.py        # Orchestration layer (runs jobs from config)
├── merge_duplicates_4.py    # Core merge / deduplication engine
├── backup.config.example    # Example configuration (safe to commit)
├── pythonEmailNotify.py     # Email helper (SMTP via env vars)
├── email_debug.py           # Email testing / debugging helper
├── Old/                     # Archived legacy versions (v1–v3)
├── README.md
└── LICENSE
```

> ⚠️ **Note:** `backup.config` and other machine-specific configs should NOT be committed.

---

## 🚀 Typical Usage

### 1️⃣ Create a config file

Copy the example and customize it:

```bash
cp backup.config.example backup.config
```

Edit `backup.config` and define one or more jobs:

```ini
[job:pictures_backup]
enabled = true
src = C:\Users\You\Pictures
src = D:\CameraImports
dst = G:\Backups\Pictures
```

---

### 2️⃣ Run a single job

```bash
python backup_runner_4.py --config backup.config --job pictures_backup
```

---

### 3️⃣ Run all enabled jobs

```bash
python backup_runner_4.py --config backup.config --all
```

---

### 4️⃣ Dry-run (no changes)

```bash
python backup_runner_4.py --config backup.config --job pictures_backup --dry-run
```

---

## 🔍 How Deduplication Works

1. If the destination file does **not exist** → copy it
2. If it exists and **hash matches** → skip (duplicate)
3. If it exists but **differs** → copy as:

```
filename (1).ext
filename (2).ext
```

Hashes are cached in SQLite to avoid re-hashing unchanged files.

---

## 🔄 Resume & Checkpoints

* Each job maintains a checkpoint file
* Successfully processed files are recorded
* On rerun, only unfinished files are retried

Checkpoints are automatically cleared after a fully successful run.

---

## 📧 Email Notifications (Optional)

Email support uses environment variables (never stored in config files).

Example:

```bash
set EMAIL_ADDRESS=you@gmail.com
set EMAIL_PASSWORD=your_app_password
set MAIN_EMAIL_ADDRESS=you@gmail.com
```

Config options allow:

* Notify on failure only
* Notify on success
* Per-job or per-run summaries

---

## 🧠 Design Goals

* **Safety first** (no silent overwrites)
* **Idempotent runs** (safe to re-run)
* **Windows & Linux compatible**
* **Automation-friendly** (Task Scheduler / cron)
* **Clear audit trail** (logs + stats)

---

## 🕰️ Legacy Versions

Older versions are preserved in the `Old/` directory for reference:

* v1–v2: early prototypes
* v3 / v3.1: stabilized merge engine
* v4: current, config-driven architecture

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙌 Author

**Malachi Clifton**
GitHub: [https://github.com/killermelon1458](https://github.com/killermelon1458)

---

If you use this project and have ideas for improvements (retention policies, compression, encryption, etc.), feel free to fork or open an issue.
