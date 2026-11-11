"""
memory_check_state.py - Small bookkeeping files for the Memory Check.

Kept separate from remember_check.py on purpose: this module imports nothing
heavy, so the session-start hook can read the one-time notice without loading
the rest of the check. Everything lives in the gitignored logs/ folder.

  memory_check_state.json    when the last run happened / failed. Written only
                             by the process that holds the run lock.
  memory_check.lock          exists while a check is running. Created with
                             O_EXCL, so two sessions starting at once can't
                             both start one; a lock left by a crash expires.
  memory_check_notice.txt    the one-time note for the next session start.
                             Presence means "not shown yet"; taking it is an
                             atomic rename, so it is shown exactly once.
  memory_check.log           errors from the background run (its own stderr
                             is discarded, so failures would otherwise be silent).
"""

import json
import os
import pathlib
import time

RUN_LOCK_MINUTES = 30
FAILURE_BACKOFF_HOURS = 24


def _logs(root):
    return pathlib.Path(root) / "logs"


def state_path(root):
    return _logs(root) / "memory_check_state.json"


def lock_path(root):
    return _logs(root) / "memory_check.lock"


def notice_path(root):
    return _logs(root) / "memory_check_notice.txt"


def log_path(root):
    return _logs(root) / "memory_check.log"


def read_state(root):
    try:
        data = json.loads(state_path(root).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_state(root, state):
    path = state_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)
    except OSError:
        pass


def lock_active(root, now=None):
    """True if a check is running right now (a lock older than the limit is a crash's leftover)."""
    try:
        age = (now or time.time()) - lock_path(root).stat().st_mtime
    except OSError:
        return False
    return age < RUN_LOCK_MINUTES * 60


def acquire_run_lock(root, now=None):
    """Take the run lock. Returns False if another check holds it."""
    path = lock_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if lock_active(root, now):
                return False
            try:
                path.unlink()                  # stale lock from a crashed run; clear it and retry
            except OSError:
                return False
        except OSError:
            return False
    return False


def release_run_lock(root):
    try:
        lock_path(root).unlink()
    except OSError:
        pass


def post_notice(root, text):
    """Leave a one-time note for the next session start (replaces any unread one)."""
    path = notice_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
    except OSError:
        pass


def take_notice(root):
    """The unread note, or None. Claiming is an atomic rename, so two sessions
    starting together can't both get it and a note posted at the same moment
    can't be overwritten."""
    path = notice_path(root)
    claimed = path.with_name(f"{path.name}.{os.getpid()}.claimed")
    try:
        os.replace(path, claimed)
    except OSError:
        return None
    try:
        return claimed.read_text().strip() or None
    except OSError:
        return None
    finally:
        try:
            claimed.unlink()
        except OSError:
            pass


def log(root, message):
    """Append a line to the check's log. Never raises."""
    try:
        path = log_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
    except OSError:
        pass
