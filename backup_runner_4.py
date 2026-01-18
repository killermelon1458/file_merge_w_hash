#!/usr/bin/env python3
"""
backup_runner.py

Backup Runner — config-centered orchestration layer for merge_duplicates_3.py

What this does:
- Loads an INI config file with one or more [job:...] sections
- Runs one job (--job NAME) or all enabled jobs (--all)
- Generates timestamped logs per job
- Uses merge_duplicates_3.py as the engine (copy + hash-cache + checkpoint)
- Optional email notifications via pythonEmailNotify.EmailSender using env vars
- Safe for Task Scheduler / cron (job lock files prevent overlap)
- Prints timestamps on runner output, plus per-job and total run times

Notes:
- The engine (merge_duplicates_3.py) already prints timestamped output and a progress bar
  (tqdm if installed). The runner avoids printing during the progress-heavy phase.
- If config/args are invalid, this script writes backup.config.example next to itself.

Requires:
- Python 3.8+ recommended (works on 3.13)
- merge_duplicates_3.py in same folder (or importable)
- If email enabled: pythonEmailNotify.py importable + SMTP creds in env vars

Typical usage:
  python backup_runner.py --config backup.config --job documents_speed_test
  python backup_runner.py --config backup.config --all
"""

from __future__ import annotations

import argparse
import configparser
import os
import sys
import time
import socket
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Engine import (your updated script name)
from merge_duplicates_4 import run_merge, MergeRunConfig

# Optional email backend
try:
    from pythonEmailNotify import EmailSender  # type: ignore
except Exception:
    EmailSender = None  # type: ignore


# =========================
# Versioning
# =========================
VERSION = "0.2.0"

# =========================
# Constants
# =========================
EXAMPLE_CONFIG_NAME = "backup.config.example"
DEFAULT_TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"
RUNNER_CONSOLE_TS_FMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_TIMEOUT_WARN_FRACTION = 0.8
DEFAULT_SPACE_MARGIN = 0.10
DEFAULT_MAX_ERRORS = 100


# =========================
# Data models
# =========================

@dataclass
class GlobalConfig:
    timestamp_format: str = DEFAULT_TIMESTAMP_FMT
    dry_run: bool = False

    # Timeout default (blank/omitted => no timeout)
    timeout_minutes_default: Optional[int] = None
    timeout_warn_fraction: float = DEFAULT_TIMEOUT_WARN_FRACTION

    # Email granularity:
    #   "per_job" => job-level emails (success/failure per job)
    #   "per_run" => one summary email at end (plus optional failure email on crash)
    email_granularity: str = "per_job"


@dataclass
class EmailConfig:
    enabled: bool = False

    # ENV VAR NAMES holding credentials
    creds_env_user: str = "EMAIL_ADDRESS"
    creds_env_pass: str = "EMAIL_PASSWORD"

    # Recipients:
    # - either to_addrs_env points to env var containing comma-separated addresses
    # - or to_addrs is explicit comma-separated in config
    to_addrs_env: Optional[str] = "MAIN_EMAIL_ADDRESS"
    to_addrs: List[str] = None  # type: ignore

    # SMTP (defaults to Gmail)
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587

    # Optional global email policy (applies when job fields not set)
    notify_on_success_default: bool = False
    notify_on_any_error_default: bool = True
    notify_on_error_count_default: Optional[int] = None
    notify_per_file_error_default: bool = False
    notify_on_timeout_warning_default: bool = True

    def __post_init__(self) -> None:
        if self.to_addrs is None:
            self.to_addrs = []


@dataclass
class JobConfig:
    name: str
    enabled: bool

    # One or more sources
    src_roots: List[Path]
    dst: Path

    # Logging
    log_dir: Optional[Path]
    log_prefix: str
    delete_logs_on_success: bool

    # Cache/checkpoint
    cache_path: Optional[Path]
    delete_cache_on_success: bool

    # Engine behavior
    move: bool
    ignore_symlinks: bool
    exclude_names: List[str]
    exclude_globs: List[str]
    log_dups: bool
    space_margin: float
    force_space: bool
    max_errors: int

    # Timeout (blank => none)
    timeout_minutes: Optional[int]

    # Notification policy (job-level overrides)
    notify_on_success: Optional[bool]
    notify_on_any_error: Optional[bool]
    notify_on_error_count: Optional[int]
    notify_per_file_error: Optional[bool]
    notify_on_timeout_warning: Optional[bool]


@dataclass
class JobRunResult:
    job_name: str
    ok: bool
    exit_code: int
    log_file: Path
    cache_path: Path
    checkpoint_path: Path
    stats: Dict[str, int]
    duration_seconds: float


# =========================
# Output helpers
# =========================

def now_ts(fmt: str) -> str:
    return time.strftime(fmt)


def print_ts(msg: str) -> None:
    # Runner-level timestamp prefix
    print(f"[{datetime.now().strftime(RUNNER_CONSOLE_TS_FMT)}] {msg}", flush=True)


def append_log_line(log_file: Path, msg: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime(RUNNER_CONSOLE_TS_FMT)}] {msg}\n")


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

def format_bytes(n: int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


# =========================
# Example config generation
# =========================

def generate_example_config(path: Path) -> None:
    example = f"""
# ================================
# Backup Runner Example Config
# ================================
# Version: {VERSION}
# Host: {socket.gethostname()}
#
# This file defines one or more backup jobs.
# Run one job:
#   python backup_runner.py --config backup.config --job documents_speed_test
# Run all enabled jobs:
#   python backup_runner.py --config backup.config --all
#
# IMPORTANT:
# - Paths can be absolute or relative
# - Secrets MUST be stored in environment variables (never put passwords in config)
# - For Windows paths, you can use backslashes (C:\\Users\\Name\\Documents)
#
# Notes on performance:
# - The engine caches hashes in SQLite (cache_path). Reusing the same cache makes
#   subsequent runs much faster for largely-unchanged libraries.
#
# ================================

[global]
# Timestamp format used for log filenames (strftime syntax)
timestamp_format = {DEFAULT_TIMESTAMP_FMT}

# Default dry run (can be overridden by CLI --dry-run)
dry_run = false

# Default timeout (minutes) applied to jobs that don't specify timeout_minutes.
# Leave blank or remove for no timeout.
timeout_minutes_default =

# When a job exceeds this fraction of timeout_minutes, the runner will warn (soft warning).
timeout_warn_fraction = {DEFAULT_TIMEOUT_WARN_FRACTION}

# Email granularity:
# - per_job: send per-job notifications (recommended)
# - per_run: send one summary email at end
email_granularity = per_job


[email]
# Enable or disable email notifications
enabled = false

# ENV VAR NAMES holding your SMTP login creds:
creds_env_user = EMAIL_ADDRESS
creds_env_pass = EMAIL_PASSWORD

# Recipients:
# Option A (recommended): recipients stored in an env var (comma-separated allowed)
to_addrs_env = MAIN_EMAIL_ADDRESS
# Option B: hardcode recipients here (comma-separated)
# to_addrs = you@example.com, other@example.com

# SMTP server settings (Gmail defaults shown)
smtp_server = smtp.gmail.com
smtp_port = 587

# Optional defaults (jobs can override)
notify_on_success_default = false
notify_on_any_error_default = true
notify_on_error_count_default =
notify_per_file_error_default = false
notify_on_timeout_warning_default = true


# ----------------
# Job Example
# ----------------
[job:documents_speed_test]
enabled = true

# One or more sources (comma-separated)
src = C:\\Users\\YourName\\Documents

# Destination root for this job
dst = D:\\Backups\\Crosshair backup\\Documents

# Logging
# If log_dir is omitted, logs default to the destination folder (dst)
log_dir =
log_prefix = documents_backup
delete_logs_on_success = false

# Cache
# If omitted, cache defaults to <dst>\\cache\\<job>.sqlite
cache_path =
delete_cache_on_success = false

# Engine behavior
move = false
ignore_symlinks = false
exclude_names =
exclude_globs =
log_dups = false
space_margin = {DEFAULT_SPACE_MARGIN}
force_space = false
max_errors = {DEFAULT_MAX_ERRORS}

# Timeout (job override; blank means "use global default"; remove both for no timeout)
timeout_minutes =

# Notifications (blank => use [email] defaults)
notify_on_success =
notify_on_any_error =
notify_on_error_count =
notify_per_file_error =
notify_on_timeout_warning =
""".strip() + "\n"

    path.write_text(example, encoding="utf-8")


# =========================
# Fatal / validation helpers
# =========================

def fatal(msg: str, *, generate_example: bool = True, exit_code: int = 2) -> None:
    print_ts(f"ERROR: {msg}")
    if generate_example:
        example_path = Path(EXAMPLE_CONFIG_NAME)
        if not example_path.exists():
            try:
                generate_example_config(example_path)
                print_ts(f"Generated example config: {example_path}")
            except Exception as e:
                print_ts(f"WARNING: Failed to write example config: {e}")
    raise SystemExit(exit_code)


def parse_bool(s: str, default: Optional[bool] = None) -> Optional[bool]:
    if s is None:
        return default
    v = s.strip().lower()
    if v == "":
        return default
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def parse_int_blank_ok(s: str) -> Optional[int]:
    if s is None:
        return None
    v = s.strip()
    if v == "":
        return None
    return int(v)


def parse_float_blank_ok(s: str, default: float) -> float:
    if s is None:
        return default
    v = s.strip()
    if v == "":
        return default
    return float(v)


def split_csv_paths(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def resolve_path(p: str) -> Path:
    # Expand %VARS% and ~
    expanded = os.path.expandvars(os.path.expanduser(p.strip()))
    return Path(expanded)


# =========================
# Email helpers
# =========================

def _get_env_or_fail(varname: str) -> str:
    val = os.environ.get(varname, "").strip()
    if not val:
        fatal(f"Missing required environment variable: {varname}", generate_example=True)
    return val


def resolve_recipients(email_cfg: EmailConfig) -> List[str]:
    if email_cfg.to_addrs_env:
        raw = os.environ.get(email_cfg.to_addrs_env, "").strip()
        if raw:
            return [x.strip() for x in raw.split(",") if x.strip()]
    if email_cfg.to_addrs:
        return list(email_cfg.to_addrs)
    return []


def build_email_sender(email_cfg: EmailConfig) -> EmailSender:
    if EmailSender is None:
        fatal("Email enabled but pythonEmailNotify.EmailSender is not importable.", generate_example=True)

    login = _get_env_or_fail(email_cfg.creds_env_user)
    password = _get_env_or_fail(email_cfg.creds_env_pass)

    recipients = resolve_recipients(email_cfg)
    default_recipient = recipients[0] if recipients else None

    return EmailSender(
        smtp_server=email_cfg.smtp_server,
        port=int(email_cfg.smtp_port),
        login=login,
        password=password,
        default_recipient=default_recipient,
    )


def read_error_lines(log_file: Path, limit: int = 30) -> List[str]:
    """Extract last N error-ish lines from the job log."""
    if not log_file.exists():
        return []
    lines: List[str] = []
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "[ERROR" in line or "ERROR:" in line:
                    lines.append(line.rstrip("\n"))
        return lines[-limit:]
    except Exception:
        return []


def send_email(email_cfg: EmailConfig, subject: str, body: str) -> None:
    if not email_cfg.enabled:
        return

    recipients = resolve_recipients(email_cfg)
    if not recipients:
        fatal("Email enabled but no recipients found (to_addrs_env/to_addrs).", generate_example=True)

    sender = build_email_sender(email_cfg)

    for r in recipients:
        try:
            sender.sendEmail(subject=subject, body=body, recipient=r, html=False)
        except Exception as e:
            # Don't crash backups due to email failures; print a warning.
            print_ts(f"WARNING: Failed to send email to {r}: {e}")


# =========================
# Config loading
# =========================

def load_config(path: Path) -> Tuple[GlobalConfig, EmailConfig, Dict[str, JobConfig]]:
    if not path.exists():
        fatal(f"Config file not found: {path}", generate_example=True)

    # IMPORTANT: disable interpolation so '%' timestamp formats work
    cfg = configparser.ConfigParser(interpolation=None)

    try:
        cfg.read(path, encoding="utf-8")
    except Exception as e:
        fatal(f"Failed to parse config: {e}", generate_example=True)

    # -------- global --------
    g = GlobalConfig()
    if cfg.has_section("global"):
        sc = cfg["global"]
        g.timestamp_format = sc.get("timestamp_format", fallback=DEFAULT_TIMESTAMP_FMT).strip() or DEFAULT_TIMESTAMP_FMT
        g.dry_run = sc.getboolean("dry_run", fallback=False)
        g.timeout_minutes_default = parse_int_blank_ok(sc.get("timeout_minutes_default", fallback=""))
        g.timeout_warn_fraction = parse_float_blank_ok(sc.get("timeout_warn_fraction", fallback=""), DEFAULT_TIMEOUT_WARN_FRACTION)
        gran = sc.get("email_granularity", fallback="per_job").strip().lower()
        g.email_granularity = gran if gran in ("per_job", "per_run") else "per_job"

    # -------- email --------
    e = EmailConfig()
    if cfg.has_section("email"):
        sc = cfg["email"]
        e.enabled = sc.getboolean("enabled", fallback=False)

        e.creds_env_user = sc.get("creds_env_user", fallback=e.creds_env_user).strip() or e.creds_env_user
        e.creds_env_pass = sc.get("creds_env_pass", fallback=e.creds_env_pass).strip() or e.creds_env_pass

        to_env = sc.get("to_addrs_env", fallback="").strip()
        e.to_addrs_env = to_env if to_env else None

        e.to_addrs = [x.strip() for x in sc.get("to_addrs", fallback="").split(",") if x.strip()]

        e.smtp_server = sc.get("smtp_server", fallback=e.smtp_server).strip() or e.smtp_server
        e.smtp_port = int(sc.get("smtp_port", fallback=str(e.smtp_port)).strip() or e.smtp_port)

        e.notify_on_success_default = sc.getboolean("notify_on_success_default", fallback=e.notify_on_success_default)
        e.notify_on_any_error_default = sc.getboolean("notify_on_any_error_default", fallback=e.notify_on_any_error_default)
        e.notify_on_error_count_default = parse_int_blank_ok(sc.get("notify_on_error_count_default", fallback=""))
        e.notify_per_file_error_default = sc.getboolean("notify_per_file_error_default", fallback=e.notify_per_file_error_default)
        e.notify_on_timeout_warning_default = sc.getboolean("notify_on_timeout_warning_default", fallback=e.notify_on_timeout_warning_default)

    # Validate email config early if enabled
    if e.enabled:
        _get_env_or_fail(e.creds_env_user)
        _get_env_or_fail(e.creds_env_pass)
        recips = resolve_recipients(e)
        if not recips:
            fatal("Email enabled but no recipients found (to_addrs_env/to_addrs).", generate_example=True)

    # -------- jobs --------
    jobs: Dict[str, JobConfig] = {}

    for section in cfg.sections():
        if not section.lower().startswith("job:"):
            continue

        name = section.split(":", 1)[1].strip()
        sc = cfg[section]

        try:
            enabled = sc.getboolean("enabled", fallback=True)

            raw_src = sc.get("src", fallback="").strip()
            if not raw_src:
                fatal(f"Job '{name}' missing required 'src' field.", generate_example=True)
            src_roots = [resolve_path(p) for p in split_csv_paths(raw_src)]
            if not src_roots:
                fatal(f"Job '{name}' has empty src list.", generate_example=True)

            raw_dst = sc.get("dst", fallback="").strip()
            if not raw_dst:
                fatal(f"Job '{name}' missing required 'dst' field.", generate_example=True)
            dst = resolve_path(raw_dst)

            log_dir_raw = sc.get("log_dir", fallback="").strip()
            log_dir = resolve_path(log_dir_raw) if log_dir_raw else None

            log_prefix = sc.get("log_prefix", fallback=name).strip() or name

            cache_path_raw = sc.get("cache_path", fallback="").strip()
            cache_path = resolve_path(cache_path_raw) if cache_path_raw else None

            delete_logs_on_success = sc.getboolean("delete_logs_on_success", fallback=False)
            delete_cache_on_success = sc.getboolean("delete_cache_on_success", fallback=False)

            move = sc.getboolean("move", fallback=False)
            ignore_symlinks = sc.getboolean("ignore_symlinks", fallback=False)

            exclude_names = [x.strip() for x in sc.get("exclude_names", fallback="").split(",") if x.strip()]
            exclude_globs = [x.strip() for x in sc.get("exclude_globs", fallback="").split(",") if x.strip()]

            log_dups = sc.getboolean("log_dups", fallback=False)

            space_margin = parse_float_blank_ok(sc.get("space_margin", fallback=""), DEFAULT_SPACE_MARGIN)
            force_space = sc.getboolean("force_space", fallback=False)
            max_errors = int(sc.get("max_errors", fallback=str(DEFAULT_MAX_ERRORS)).strip() or DEFAULT_MAX_ERRORS)

            # Timeout: blank => use global default; both blank => None
            timeout_raw = sc.get("timeout_minutes", fallback="").strip()
            timeout_minutes = int(timeout_raw) if timeout_raw else None
            if timeout_minutes is None:
                timeout_minutes = g.timeout_minutes_default

            # Notifications: blank => use email defaults
            notify_on_success = parse_bool(sc.get("notify_on_success", fallback=""), default=None)
            notify_on_any_error = parse_bool(sc.get("notify_on_any_error", fallback=""), default=None)
            notify_on_error_count = parse_int_blank_ok(sc.get("notify_on_error_count", fallback=""))
            notify_per_file_error = parse_bool(sc.get("notify_per_file_error", fallback=""), default=None)
            notify_on_timeout_warning = parse_bool(sc.get("notify_on_timeout_warning", fallback=""), default=None)

            jobs[name] = JobConfig(
                name=name,
                enabled=enabled,
                src_roots=src_roots,
                dst=dst,
                log_dir=log_dir,
                log_prefix=log_prefix,
                delete_logs_on_success=delete_logs_on_success,
                cache_path=cache_path,
                delete_cache_on_success=delete_cache_on_success,
                move=move,
                ignore_symlinks=ignore_symlinks,
                exclude_names=exclude_names,
                exclude_globs=exclude_globs,
                log_dups=log_dups,
                space_margin=space_margin,
                force_space=force_space,
                max_errors=max_errors,
                timeout_minutes=timeout_minutes,
                notify_on_success=notify_on_success,
                notify_on_any_error=notify_on_any_error,
                notify_on_error_count=notify_on_error_count,
                notify_per_file_error=notify_per_file_error,
                notify_on_timeout_warning=notify_on_timeout_warning,
            )

        except SystemExit:
            raise
        except Exception as e:
            fatal(f"Invalid job config [{section}]: {e}", generate_example=True)

    if not jobs:
        fatal("No [job:...] sections found in config.", generate_example=True)

    return g, e, jobs


# =========================
# Job runner
# =========================

def resolve_job_log_and_cache(job: JobConfig, ts_fmt: str) -> Tuple[Path, Path, Path]:
    # Default log_dir to destination folder (per your request)
    log_dir = job.log_dir or job.dst
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job.log_prefix}_{now_ts(ts_fmt)}.log"

    # Default cache to <dst>\cache\<job>.sqlite (keeps dst clean unless specified)
    cache_path = job.cache_path or (job.dst / "cache" / f"{job.name}.sqlite")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = cache_path.with_suffix(".checkpoint")
    return log_file, cache_path, checkpoint_path


def effective_notify_bool(job_val: Optional[bool], default_val: bool) -> bool:
    return default_val if job_val is None else bool(job_val)


def run_job(
    job: JobConfig,
    *,
    dry_run: bool,
    ts_fmt: str,
    global_cfg: GlobalConfig,
    email_cfg: EmailConfig,
) -> JobRunResult:
    job_start = time.time()

    log_file, cache_path, checkpoint_path = resolve_job_log_and_cache(job, ts_fmt)

    # Job lock to prevent overlap
    lock_path = cache_path.parent / f"{job.name}.lock"
    if lock_path.exists():
        msg = f"Job '{job.name}' appears to already be running (lock exists: {lock_path})."
        print_ts(f"ERROR: {msg}")
        append_log_line(log_file, f"ERROR: {msg}")
        # optional email if configured
        if email_cfg.enabled:
            if effective_notify_bool(job.notify_on_any_error, email_cfg.notify_on_any_error_default):
                send_email(email_cfg, f"[Backup Error] Locked job: {job.name}", msg)
        return JobRunResult(
            job_name=job.name,
            ok=False,
            exit_code=3,
            log_file=log_file,
            cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            stats={"errors": 1},
            duration_seconds=time.time() - job_start,
        )

    lock_path.write_text(
        f"pid={os.getpid()}\nstarted={datetime.now().isoformat()}\n",
        encoding="utf-8",
    )

    # Runner header (avoid printing during tqdm progress)
    print_ts(f"=== Running job: {job.name} ===")
    print_ts(f"Log: {log_file}")
    print_ts(f"Dry run: {dry_run}")
    append_log_line(log_file, f"Runner start job={job.name} dry_run={dry_run}")
    append_log_line(log_file, f"Sources: {', '.join(str(p) for p in job.src_roots)}")
    append_log_line(log_file, f"Destination: {job.dst}")

    if job.timeout_minutes:
        print_ts(f"Timeout: {job.timeout_minutes} minutes (soft warning at {global_cfg.timeout_warn_fraction:.0%})")
        append_log_line(log_file, f"Timeout_minutes={job.timeout_minutes} warn_fraction={global_cfg.timeout_warn_fraction}")

    # Build engine config
    merge_cfg = MergeRunConfig(
        src_roots=job.src_roots,
        dst_root=job.dst,
        log_path=log_file,
        cache_path=cache_path,
        checkpoint_path=checkpoint_path,
        move=job.move,
        ignore_symlinks=job.ignore_symlinks,
        exclude_names=job.exclude_names,
        exclude_globs=job.exclude_globs,
        log_dups=job.log_dups,
        space_margin=float(job.space_margin),
        force_space=bool(job.force_space),
        max_errors=int(job.max_errors),
        dry_run=bool(dry_run),
    )

    try:
        # Run engine
        result = run_merge(merge_cfg)
        ok = (result.exit_code == 0)
        stats = result.stats

        # Soft timeout warning (we don't kill the job; safe for resume)
        if job.timeout_minutes:
            elapsed_min = (time.time() - job_start) / 60.0
            if elapsed_min >= (job.timeout_minutes * float(global_cfg.timeout_warn_fraction)):
                warn = (
                    f"WARNING: Job '{job.name}' nearing timeout "
                    f"({elapsed_min:.1f}/{job.timeout_minutes} minutes). "
                    f"Recommendation: increase timeout_minutes for this job."
                )
                print_ts(warn)
                append_log_line(log_file, warn)

                if email_cfg.enabled and effective_notify_bool(job.notify_on_timeout_warning, email_cfg.notify_on_timeout_warning_default):
                    if global_cfg.email_granularity == "per_job":
                        send_email(email_cfg, f"[Backup Warning] Near timeout: {job.name}", warn)

        # Per-job total time
        elapsed = time.time() - job_start
        print_ts(
            f"Job '{job.name}' finished in {format_duration(elapsed)} "
            f"(ok={ok}, errors={stats.get('errors', 0)})"
        )
        print_ts(
            "Job bytes: "
            f"copied={format_bytes(int(stats.get('copied_bytes', 0)))}, "
            f"renamed={format_bytes(int(stats.get('renamed_bytes', 0)))}, "
            f"skipped={format_bytes(int(stats.get('skipped_bytes', 0)))}"
        )

        append_log_line(log_file, f"Runner job elapsed_seconds={elapsed:.3f} ok={ok} stats={stats}")

        return JobRunResult(
            job_name=job.name,
            ok=ok,
            exit_code=int(result.exit_code),
            log_file=log_file,
            cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            stats=stats,
            duration_seconds=elapsed,
        )

    finally:
        # Always release lock
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass


def post_job_notifications_and_cleanup(
    job: JobConfig,
    jr: JobRunResult,
    *,
    dry_run: bool,
    global_cfg: GlobalConfig,
    email_cfg: EmailConfig,
) -> None:
    # Resolve effective notification policy
    notify_on_success = effective_notify_bool(job.notify_on_success, email_cfg.notify_on_success_default)
    notify_on_any_error = effective_notify_bool(job.notify_on_any_error, email_cfg.notify_on_any_error_default)
    notify_per_file_error = effective_notify_bool(job.notify_per_file_error, email_cfg.notify_per_file_error_default)
    notify_on_timeout_warning = effective_notify_bool(job.notify_on_timeout_warning, email_cfg.notify_on_timeout_warning_default)
    _ = notify_on_timeout_warning  # already handled during run, but kept for clarity

    errors = int(jr.stats.get("errors", 0))
    threshold = job.notify_on_error_count if job.notify_on_error_count is not None else email_cfg.notify_on_error_count_default

    # Emails (only if per_job granularity)
    if email_cfg.enabled and global_cfg.email_granularity == "per_job":
        if jr.ok and notify_on_success:
            send_email(
                email_cfg,
                subject=f"[Backup Success] {jr.job_name}",
                body=(
                    f"Job succeeded.\n\n"
                    f"Job: {jr.job_name}\n"
                    f"Duration: {jr.duration_seconds:.1f} sec\n"
                    f"Log: {jr.log_file}\n"
                    f"Stats: {jr.stats}\n"
                ),
            )

        if (not jr.ok) or (errors > 0):
            should_send = False
            if notify_on_any_error and errors > 0:
                should_send = True
            if threshold is not None and errors >= int(threshold):
                should_send = True

            if should_send:
                err_lines = read_error_lines(jr.log_file, limit=30) if notify_per_file_error else []
                extra = ("\n\nRecent error lines:\n" + "\n".join(err_lines)) if err_lines else ""
                send_email(
                    email_cfg,
                    subject=f"[Backup Failure] {jr.job_name}",
                    body=(
                        f"Job failed or finished with errors.\n\n"
                        f"Job: {jr.job_name}\n"
                        f"Exit code: {jr.exit_code}\n"
                        f"Duration: {jr.duration_seconds:.1f} sec\n"
                        f"Log: {jr.log_file}\n"
                        f"Errors: {errors}\n"
                        f"Stats: {jr.stats}\n"
                        f"{extra}\n"
                    ),
                )

    # Cleanup on success (only when not dry-run and no errors)
    if jr.ok and (errors == 0) and (not dry_run):
        if job.delete_cache_on_success:
            try:
                if jr.cache_path.exists():
                    jr.cache_path.unlink()
                if jr.checkpoint_path.exists():
                    jr.checkpoint_path.unlink()
            except Exception:
                pass

        if job.delete_logs_on_success:
            try:
                if jr.log_file.exists():
                    jr.log_file.unlink()
            except Exception:
                pass


# =========================
# Legacy mode (no config)
# =========================

def legacy_mode(argv: List[str]) -> int:
    """
    Allow running the runner in a "legacy" way by accepting the engine's CLI-like args.
    Example:
      python backup_runner.py --src ... --dst ... --log ... --cache ...
    """
    ap = argparse.ArgumentParser(description="Backup Runner (legacy mode -> merge_duplicates_3)")
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--checkpoint")

    ap.add_argument("--move", action="store_true")
    ap.add_argument("--log-dups", action="store_true")
    ap.add_argument("--space-margin", type=float, default=DEFAULT_SPACE_MARGIN)
    ap.add_argument("--force-space", action="store_true")
    ap.add_argument("--max-errors", type=int, default=DEFAULT_MAX_ERRORS)
    ap.add_argument("--ignore-symlinks", action="store_true")
    ap.add_argument("--exclude-name", action="append", default=[])
    ap.add_argument("--exclude-glob", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)

    cfg = MergeRunConfig(
        src_roots=[resolve_path(x) for x in args.src],
        dst_root=resolve_path(args.dst),
        log_path=resolve_path(args.log),
        cache_path=resolve_path(args.cache),
        checkpoint_path=resolve_path(args.checkpoint) if args.checkpoint else None,
        move=bool(args.move),
        ignore_symlinks=bool(args.ignore_symlinks),
        exclude_names=list(args.exclude_name or []),
        exclude_globs=list(args.exclude_glob or []),
        log_dups=bool(args.log_dups),
        space_margin=float(args.space_margin),
        force_space=bool(args.force_space),
        max_errors=int(args.max_errors),
        dry_run=bool(args.dry_run),
    )

    r = run_merge(cfg)
    return int(r.exit_code)


# =========================
# Main
# =========================

def main() -> int:
    ap = argparse.ArgumentParser(description="Backup Runner")
    ap.add_argument("--config", help="Path to backup.config")
    ap.add_argument("--job", help="Run a single job by name")
    ap.add_argument("--all", action="store_true", help="Run all enabled jobs")
    ap.add_argument("--dry-run", action="store_true", help="Override config dry_run")
    ap.add_argument("--list-jobs", action="store_true", help="List jobs in config")
    ap.add_argument("--version", action="store_true", help="Print version and exit")

    args, unknown = ap.parse_known_args()

    if args.version:
        print(VERSION)
        return 0

    # If no config, treat remaining args as legacy engine invocation
    if not args.config:
        if unknown:
            return legacy_mode(unknown)
        print_ts("No --config provided. Either pass --config or run merge_duplicates_3.py directly.")
        print_ts(f"Tip: python backup_runner.py --config backup.config --job <name>")
        return 0

    script_start = time.time()

    # Config-driven mode
    global_cfg, email_cfg, jobs = load_config(resolve_path(args.config))

    if args.list_jobs:
        for j in jobs.values():
            print(f"{j.name} (enabled={j.enabled})")
        return 0

    if not args.job and not args.all:
        fatal("Must specify --job <name> or --all when using --config.", generate_example=True)

    ts_fmt = global_cfg.timestamp_format
    dry_run = bool(args.dry_run) or bool(global_cfg.dry_run)

    selected: List[JobConfig]
    if args.all:
        selected = [j for j in jobs.values() if j.enabled]
        if not selected:
            fatal("No enabled jobs found.", generate_example=True)
    else:
        if args.job not in jobs:
            fatal(f"Job not found: {args.job}", generate_example=True)
        selected = [jobs[args.job]]

    # Run jobs
    results: List[JobRunResult] = []
    any_failure = False

    try:
        for job in selected:
            jr = run_job(job, dry_run=dry_run, ts_fmt=ts_fmt, global_cfg=global_cfg, email_cfg=email_cfg)
            results.append(jr)

            post_job_notifications_and_cleanup(
                job,
                jr,
                dry_run=dry_run,
                global_cfg=global_cfg,
                email_cfg=email_cfg,
            )

            if not jr.ok or int(jr.stats.get("errors", 0)) > 0:
                any_failure = True

    except Exception as e:
        # Script failure email (best effort)
        if email_cfg.enabled:
            tb = traceback.format_exc()
            send_email(
                email_cfg,
                subject="[Backup Runner Failure] Script crashed",
                body=f"Backup runner crashed.\n\nError: {e}\n\nTraceback:\n{tb}",
            )
        raise

    # Total time
    total_elapsed = time.time() - script_start
    print_ts(
        f"All selected jobs finished in {format_duration(total_elapsed)} "
        f"(failures={any_failure})"
    )

    total_copied = sum(int(jr.stats.get("copied_bytes", 0)) for jr in results)
    total_renamed = sum(int(jr.stats.get("renamed_bytes", 0)) for jr in results)
    total_skipped = sum(int(jr.stats.get("skipped_bytes", 0)) for jr in results)

    print_ts(
        "Run bytes: "
        f"copied={format_bytes(total_copied)}, "
        f"renamed={format_bytes(total_renamed)}, "
        f"skipped={format_bytes(total_skipped)}"
    )

    # Per-run summary email if configured
    if email_cfg.enabled and global_cfg.email_granularity == "per_run":
        lines = []
        for jr in results:
            lines.append(
                f"- {jr.job_name}: ok={jr.ok}, exit={jr.exit_code}, "
                f"errors={jr.stats.get('errors', 0)}, time={jr.duration_seconds:.1f}s, log={jr.log_file}"
            )
        body = (
            "Backup run summary:\n\n"
            + "\n".join(lines)
            + f"\n\nTotal time: {total_elapsed:.1f}s\nFailures: {any_failure}\n"
        )
        subj = "[Backup Run] Completed with failures" if any_failure else "[Backup Run] Completed successfully"
        send_email(email_cfg, subj, body)

    return 1 if any_failure else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        # On any pre-run config/arg failure we generate example; on crash, also generate
        try:
            p = Path(EXAMPLE_CONFIG_NAME)
            if not p.exists():
                generate_example_config(p)
                print_ts(f"Generated example config: {p}")
        except Exception:
            pass
        raise
