#!/usr/bin/env python3
"""
Backup Runner — config‑centered orchestration layer for merge_duplicates.py

Design goals:
- INI‑based configuration with multiple jobs
- Backward compatible with legacy CLI usage
- Dry‑run support
- Per‑job logging, cache, timeout, notification policy
- Safe for scheduled (cron / Task Scheduler) execution

NOTE:
This file is intentionally structured and verbose. It is meant to be a
*production‑ready foundation*, not a minimal script.
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
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
from datetime import datetime
from merge_duplicates import run_merge, MergeRunConfig 

# =========================
# Versioning
# =========================
VERSION = "0.1.0"

# =========================
# Constants
# =========================
EXAMPLE_CONFIG_NAME = "backup.config.example"
DEFAULT_TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"

# =========================
# Data models
# =========================

@dataclass
class EmailConfig:
    enabled: bool
    provider: str
    creds_env_user: str
    creds_env_pass: str
    from_addr: str

    # Either use explicit to_addrs OR an env var containing recipients
    to_addrs: List[str]
    to_addrs_env: Optional[str] = None



@dataclass
class JobConfig:
    name: str
    enabled: bool
    src: Path
    dst: Path
    log_dir: Optional[Path]
    log_prefix: str
    delete_logs_on_success: bool
    cache_path: Optional[Path]
    delete_cache_on_success: bool
    move: bool
    ignore_symlinks: bool
    exclude_globs: List[str]
    timeout_minutes: Optional[int]

    # notification policy
    notify_on_success: bool
    notify_on_any_error: bool
    notify_on_error_count: Optional[int]
    notify_per_file_error: bool
    notify_on_timeout_warning: bool


# =========================
# Example config generation
# =========================

def generate_example_config(path: Path) -> None:
    """
    Writes a fully documented example backup.config file.
    This is generated automatically on config / argument failures.
    """

    example = f"""
# ================================
# Backup Runner Example Config
# ================================
# Version: {VERSION}
# Host: {socket.gethostname()}
#
# This file defines one or more backup jobs.
# Each job is fully self‑contained and can be run individually
# or together using --all.
#
# All paths may be absolute or relative.
# Secrets MUST be stored in environment variables.

[global]
# Timestamp format used for log filenames
# Uses Python strftime syntax
# Default: {DEFAULT_TIMESTAMP_FMT}
timestamp_format = {DEFAULT_TIMESTAMP_FMT}

# Default dry‑run behavior (can be overridden by CLI)
dry_run = false


[email]
# Enable or disable email notifications globally
enabled = false

# Notification backend
provider = pythonEmailNotify

# Environment variables holding credentials
creds_env_user = EMAIL_USER
creds_env_pass = EMAIL_PASS

# Email addressing
from_addr = backups@example.com
to_addrs = you@example.com, other@example.com


[job:documents_backup]
# Enable or disable this job
enabled = true

# Source and destination paths
src = C:\\Users\\YourName\\Documents
dst = D:\\Backups\\Documents

# Logging
# If log_dir is omitted, logs default to <dst>/logs
log_dir =
log_prefix = documents_backup
delete_logs_on_success = false

# Cache
# If omitted, cache defaults to <dst>/cache/<job>.sqlite
cache_path =
delete_cache_on_success = false

# Merge behavior
move = false
ignore_symlinks = false
exclude_globs = *.tmp, *.log

# Timeout (minutes)
# Leave blank or remove for no timeout
timeout_minutes =

# Notifications
notify_on_success = false
notify_on_any_error = true
notify_on_error_count = 10
notify_per_file_error = false
notify_on_timeout_warning = true

# ================================
# End of example config
# ================================
"""

    path.write_text(example.strip() + "\n", encoding="utf-8")


# =========================
# Utility helpers
# =========================

def fatal(msg: str, *, generate_example: bool = False) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    if generate_example:
        example_path = Path(EXAMPLE_CONFIG_NAME)
        if not example_path.exists():
            generate_example_config(example_path)
            print(f"Generated example config: {example_path}", file=sys.stderr)
    sys.exit(2)


def now_ts(fmt: str) -> str:
    return time.strftime(fmt)

# =========================
# EMAIL HELPERS (Ctrl+F: EMAIL HELPERS)
# =========================

from pythonEmailNotify import EmailSender  # uses your EmailSender class

def _get_env_or_fail(varname: str) -> str:
    val = os.environ.get(varname, "").strip()
    if not val:
        fatal(f"Missing required environment variable: {varname}", generate_example=True)
    return val

def resolve_email_recipients(email_cfg: EmailConfig) -> List[str]:
    # Prefer env var recipients if provided
    if getattr(email_cfg, "to_addrs_env", None):
        raw = _get_env_or_fail(email_cfg.to_addrs_env)  # e.g. MAIN_EMAIL_ADDRESS
        return [x.strip() for x in raw.split(",") if x.strip()]

    if email_cfg.to_addrs:
        return email_cfg.to_addrs

    fatal("Email enabled but no recipients configured (set to_addrs or to_addrs_env).", generate_example=True)
    return []

def build_email_sender(email_cfg: EmailConfig) -> EmailSender:
    """
    Builds EmailSender from env vars + config.
    Your EmailSender requires smtp_server + port + login + password + default_recipient. :contentReference[oaicite:1]{index=1}
    """
    login = _get_env_or_fail(email_cfg.creds_env_user)   # EMAIL_ADDRESS
    password = _get_env_or_fail(email_cfg.creds_env_pass)  # EMAIL_PASSWORD
    recipients = resolve_email_recipients(email_cfg)

    # Optional: allow smtp overrides via config, else default to Gmail SMTP.
    smtp_server = getattr(email_cfg, "smtp_server", None) or "smtp.gmail.com"
    port = int(getattr(email_cfg, "smtp_port", None) or 587)

    default_recipient = recipients[0] if recipients else None
    return EmailSender(
        smtp_server=smtp_server,
        port=port,
        login=login,
        password=password,
        default_recipient=default_recipient,
    )

def send_email(email_cfg: EmailConfig, subject: str, body: str, *, html: bool = False) -> None:
    """
    Sends to all recipients. EmailSender supports one recipient at a time,
    so we loop. :contentReference[oaicite:2]{index=2}
    """
    if not email_cfg or not email_cfg.enabled:
        return

    sender = build_email_sender(email_cfg)
    recipients = resolve_email_recipients(email_cfg)

    for r in recipients:
        try:
            sender.sendEmail(subject=subject, body=body, recipient=r, html=html)
        except Exception:
            # Don't crash the backup if email fails
            print(f"WARNING: Email send failed to {r}.", file=sys.stderr)


# =========================
# Config loading & validation
# =========================

def load_config(path: Path) -> tuple[Dict[str, str], Optional[EmailConfig], Dict[str, JobConfig]]:
    if not path.exists():
        fatal(f"Config file not found: {path}", generate_example=True)

    cfg = configparser.ConfigParser(interpolation=None)

    try:
        cfg.read(path, encoding="utf-8")
    except Exception as e:
        fatal(f"Failed to parse config: {e}", generate_example=True)

    # -------- global --------
    global_cfg = dict(cfg["global"]) if cfg.has_section("global") else {}

    # -------- email --------
    email_cfg = None
    if cfg.has_section("email"):
        try:
            enabled = cfg.getboolean("email", "enabled", fallback=False)
            if enabled:
                to_addrs_env = cfg.get("email", "to_addrs_env", fallback="").strip() or None
                to_addrs_raw = cfg.get("email", "to_addrs", fallback="").strip()

                # Allow either to_addrs_env OR to_addrs
                to_addrs = [x.strip() for x in to_addrs_raw.split(",") if x.strip()]

                email_cfg = EmailConfig(
                    enabled=True,
                    provider=cfg.get("email", "provider", fallback="pythonEmailNotify"),
                    creds_env_user=cfg.get("email", "creds_env_user"),
                    creds_env_pass=cfg.get("email", "creds_env_pass"),
                    from_addr=cfg.get("email", "from_addr", fallback=""),
                    to_addrs=to_addrs,
                    to_addrs_env=to_addrs_env,
                )
        except Exception as e:
            fatal(f"Invalid [email] config: {e}", generate_example=True)

    # -------- jobs --------
    jobs: Dict[str, JobConfig] = {}

    for section in cfg.sections():
        if not section.startswith("job:"):
            continue

        name = section.split(":", 1)[1]
        sc = cfg[section]

        try:
            timeout_raw = sc.get("timeout_minutes", fallback="").strip()
            timeout = int(timeout_raw) if timeout_raw else None

            jobs[name] = JobConfig(
                name=name,
                enabled=sc.getboolean("enabled", fallback=True),
                src=Path(sc.get("src")),
                dst=Path(sc.get("dst")),
                log_dir=Path(sc.get("log_dir")) if sc.get("log_dir", fallback="").strip() else None,
                log_prefix=sc.get("log_prefix", fallback=name),
                delete_logs_on_success=sc.getboolean("delete_logs_on_success", fallback=False),
                cache_path=Path(sc.get("cache_path")) if sc.get("cache_path", fallback="").strip() else None,
                delete_cache_on_success=sc.getboolean("delete_cache_on_success", fallback=False),
                move=sc.getboolean("move", fallback=False),
                ignore_symlinks=sc.getboolean("ignore_symlinks", fallback=False),
                exclude_globs=[x.strip() for x in sc.get("exclude_globs", fallback="").split(",") if x.strip()],
                timeout_minutes=timeout,
                notify_on_success=sc.getboolean("notify_on_success", fallback=False),
                notify_on_any_error=sc.getboolean("notify_on_any_error", fallback=True),
                notify_on_error_count=(int(x) if (x := sc.get("notify_on_error_count", fallback="").strip()) else None),
                notify_per_file_error=sc.getboolean("notify_per_file_error", fallback=False),
                notify_on_timeout_warning=sc.getboolean("notify_on_timeout_warning", fallback=True),
            )
        except Exception as e:
            fatal(f"Invalid job config [{section}]: {e}", generate_example=True)

    if not jobs:
        fatal("No jobs defined in config", generate_example=True)

    return global_cfg, email_cfg, jobs


def run_job(job: JobConfig, *, dry_run: bool, ts_fmt: str, email_cfg: Optional[EmailConfig]) -> bool:
    start_time = time.time()
    print(f"\n=== Running job: {job.name} ===")

    # Resolve log + cache defaults
    log_dir = job.log_dir or (job.dst / "logs")
    cache_path = job.cache_path or (job.dst / "cache" / f"{job.name}.sqlite")

    log_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Define log file BEFORE lock so lock errors can log
    log_file = log_dir / f"{job.log_prefix}_{now_ts(ts_fmt)}.log"

    # =========================
    # JOB LOCK (Ctrl+F: JOB LOCK)
    # =========================
    lock_path = cache_path.parent / f"{job.name}.lock"
    if lock_path.exists():
        msg = f"Job '{job.name}' appears to already be running (lock exists: {lock_path})."
        print(f"ERROR: {msg}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"ERROR: {msg}\n")
        if email_cfg and job.notify_on_any_error:
            send_email(email_cfg, f"[Backup Error] Locked job: {job.name}", msg)
        return False

    lock_path.write_text(
        f"pid={os.getpid()}\nstarted={datetime.now().isoformat()}\n",
        encoding="utf-8"
    )

    print(f"Log: {log_file}")
    print(f"Dry run: {dry_run}")
    if job.timeout_minutes:
        print(f"Timeout: {job.timeout_minutes} minutes")

    # =========================
    # MERGE ENGINE CALL (Ctrl+F: MERGE ENGINE CALL)
    # =========================
    checkpoint_path = cache_path.with_suffix(".checkpoint")

    try:
        merge_cfg = MergeRunConfig(
            src_roots=[job.src],
            dst_root=job.dst,
            log_path=log_file,
            cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            move=job.move,
            ignore_symlinks=job.ignore_symlinks,
            exclude_globs=job.exclude_globs,
            dry_run=dry_run,
        )

        result = run_merge(merge_cfg)
        success = (result.exit_code == 0)
        stats = result.stats

        # =========================
        # TIMEOUT WARNING (Ctrl+F: TIMEOUT WARNING)
        # =========================
        if job.timeout_minutes:
            elapsed_min = (time.time() - start_time) / 60
            if elapsed_min > job.timeout_minutes * 0.8:
                warn = (
                    f"WARNING: Job '{job.name}' nearing timeout "
                    f"({elapsed_min:.1f}/{job.timeout_minutes} minutes). "
                    f"Recommendation: increase timeout_minutes for this job."
                )
                print(warn)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(warn + "\n")

                if email_cfg and job.notify_on_timeout_warning:
                    send_email(
                        email_cfg,
                        subject=f"[Backup Warning] Near timeout: {job.name}",
                        body=f"{warn}\n\nLog: {log_file}\nSrc: {job.src}\nDst: {job.dst}\nStats: {stats}",
                        html=False,
                    )

        # =========================
        # EMAIL NOTIFICATIONS (Ctrl+F: EMAIL NOTIFICATIONS)
        # =========================
        if email_cfg and email_cfg.enabled:
            if success and job.notify_on_success:
                send_email(
                    email_cfg,
                    subject=f"[Backup Success] {job.name}",
                    body=f"Job succeeded.\n\nJob: {job.name}\nSrc: {job.src}\nDst: {job.dst}\nLog: {log_file}\nStats: {stats}",
                    html=False,
                )

            if not success:
                if job.notify_on_any_error or (job.notify_on_error_count and stats.get("errors", 0) >= job.notify_on_error_count):
                    send_email(
                        email_cfg,
                        subject=f"[Backup Failure] {job.name}",
                        body=(
                            f"Job failed or finished with errors.\n\n"
                            f"Job: {job.name}\nSrc: {job.src}\nDst: {job.dst}\nLog: {log_file}\n"
                            f"Errors: {stats.get('errors', 0)}\nStats: {stats}\n\n"
                            f"If timeout-related, increase timeout_minutes."
                        ),
                        html=False,
                    )

        # =========================
        # CLEANUP ON SUCCESS (Ctrl+F: CLEANUP ON SUCCESS)
        # =========================
        if success and (not dry_run):
            if job.delete_cache_on_success:
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                except Exception:
                    pass

            if job.delete_logs_on_success:
                try:
                    if log_file.exists():
                        log_file.unlink()
                except Exception:
                    pass

        return success

    finally:
        # Always release lock
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass



# =========================
# Main
# =========================

def main() -> int:
    ap = argparse.ArgumentParser(description="Backup Runner")
    ap.add_argument("--config", help="Path to backup.config")
    ap.add_argument("--job", help="Run a single job")
    ap.add_argument("--all", action="store_true", help="Run all enabled jobs")
    ap.add_argument("--dry-run", action="store_true", help="Override config dry-run")
    ap.add_argument("--list-jobs", action="store_true", help="List jobs in config")

    args, unknown = ap.parse_known_args()

    # Backward compatibility path
    if not args.config:
        print("No config provided — legacy mode not yet wired in this version")
        print("(merge_duplicates.py direct usage remains supported)")
        return 0

    # Config-driven mode
    global_cfg, email_cfg, jobs = load_config(Path(args.config))

    if args.list_jobs:
        for j in jobs.values():
            print(f"{j.name} (enabled={j.enabled})")
        return 0

    if not args.job and not args.all:
        fatal("Must specify --job <name> or --all when using --config", generate_example=True)

    ts_fmt = global_cfg.get("timestamp_format", DEFAULT_TIMESTAMP_FMT)
    dry_run = args.dry_run or global_cfg.get("dry_run", "false").lower() == "true"

    selected: List[JobConfig] = []

    if args.all:
        selected = [j for j in jobs.values() if j.enabled]
    else:
        if args.job not in jobs:
            fatal(f"Job not found: {args.job}", generate_example=True)
        selected = [jobs[args.job]]

    any_failure = False

    for job in selected:
        ok = run_job(job, dry_run=dry_run, ts_fmt=ts_fmt, email_cfg=email_cfg)
        if not ok:
            any_failure = True

    return 1 if any_failure else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        generate_example_config(Path(EXAMPLE_CONFIG_NAME))
        sys.exit(3)
