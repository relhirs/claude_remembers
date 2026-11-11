#!/usr/bin/env python3
"""
semantic_helper.py - Keeps the embedding model loaded between prompts.

Only used when config.yaml sets `semantic_search: mode: always`. Loading the
model takes ~3 seconds and ~600 MB, which is too slow to pay on every prompt.
This small background process pays it once and then turns a piece of text into
a vector in milliseconds. It does ONE thing: text in, vector out. It never
opens the database, so there is nothing to go stale as new memories arrive;
the hook does the scoring itself.

How the hook uses it:
  - SessionStart starts it (ensure_running) if it isn't already up.
  - Each prompt asks it for the query vector with a short timeout.
  - If it is down or slow, that prompt simply runs keyword-only and the
    helper is started again for next time. A prompt never waits for a start.
  - After `idle_shutdown_min` minutes with no requests it exits and frees
    the memory.

Safety: it listens on 127.0.0.1 only, on a random port, and every request must
carry a random token that lives in logs/semantic_helper.json (readable only by
you). Plain localhost TCP is used instead of a Unix socket so it works the same
on macOS, Linux and Windows.

Run by hand for debugging:  python3 scripts/semantic_helper.py --idle-min 5
"""

import argparse
import hmac
import json
import os
import pathlib
import secrets
import socket
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOST = "127.0.0.1"
REQUEST_CAP_TIMEOUT_S = 1.5     # internal safety cap: a prompt never waits longer than this
MAX_REQUEST_BYTES = 64 * 1024
SPAWN_LOCK_STALE_S = 30
DEFAULT_IDLE_MIN = 30


def info_path(root=ROOT):
    return pathlib.Path(root) / "logs" / "semantic_helper.json"


def _lock_path(root=ROOT):
    return pathlib.Path(root) / "logs" / "semantic_helper.lock"


def read_info(root=ROOT):
    try:
        data = json.loads(info_path(root).read_text())
        return data if {"port", "token"} <= set(data) else None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Client side (used by the hooks)
# ---------------------------------------------------------------------------

def query(root, text, timeout=REQUEST_CAP_TIMEOUT_S):
    """Ask the helper for the embedding of `text`. Returns a list of floats, or
    None if the helper isn't running, is too slow, or answers badly."""
    info = read_info(root)
    if not info:
        return None
    try:
        with socket.create_connection((HOST, int(info["port"])), timeout=timeout) as conn:
            conn.settimeout(timeout)
            conn.sendall((json.dumps({"token": info["token"], "text": text}) + "\n").encode())
            buf = b""
            deadline = time.monotonic() + timeout
            while not buf.endswith(b"\n") and len(buf) < 1_000_000:
                if time.monotonic() > deadline:
                    return None
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
        reply = json.loads(buf.decode())
        vec = reply.get("vector")
        return vec if isinstance(vec, list) and vec else None
    except (OSError, ValueError):
        return None


def ping(root, timeout=0.5):
    """True if a helper is up and answering."""
    return query(root, "ping", timeout=timeout) is not None


def ensure_running(root=ROOT, idle_min=DEFAULT_IDLE_MIN, python=None):
    """Start the helper in the background if it isn't running. Never blocks.

    Returns "running", "started" or "failed". A short-lived lock file stops two
    sessions that open at the same moment from starting two helpers.
    """
    root = pathlib.Path(root)
    if read_info(root) and ping(root):
        return "running"
    lock = _lock_path(root)
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            if time.time() - lock.stat().st_mtime > SPAWN_LOCK_STALE_S:
                lock.unlink()           # a start that died; clear its lock
        except FileNotFoundError:
            pass
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return "started"                # another session is already starting it
    except OSError:
        return "failed"
    try:
        cmd = [python or sys.executable, str(pathlib.Path(__file__).resolve()),
               "--root", str(root), "--idle-min", str(idle_min)]
        quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: outlive the hook
            subprocess.Popen(cmd, creationflags=0x00000008 | 0x00000200, **quiet)
        else:
            subprocess.Popen(cmd, start_new_session=True, **quiet)
        return "started"
    except OSError:
        try:
            lock.unlink()
        except OSError:
            pass
        return "failed"


# ---------------------------------------------------------------------------
# Server side
# ---------------------------------------------------------------------------

def _write_info(root, port, token):
    path = info_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"port": port, "token": token, "pid": os.getpid()}, f)
    os.replace(tmp, path)


def serve(root, embed, idle_min=DEFAULT_IDLE_MIN, ready=None):
    """Serve embeddings until idle for `idle_min` minutes. `embed(text)` returns
    a list of floats; it is injected so tests don't need the real model."""
    token = secrets.token_hex(16)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, 0))
    server.listen(4)
    server.settimeout(1.0)
    port = server.getsockname()[1]
    _write_info(root, port, token)
    try:
        _lock_path(root).unlink()
    except OSError:
        pass
    if ready is not None:
        ready.set()
    last_used = time.monotonic()
    try:
        while time.monotonic() - last_used < idle_min * 60:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(2.0)
                try:
                    buf = b""
                    while not buf.endswith(b"\n") and len(buf) < MAX_REQUEST_BYTES:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                    req = json.loads(buf.decode())
                    if not hmac.compare_digest(str(req.get("token", "")), token):
                        conn.sendall(b'{"error":"bad token"}\n')
                        continue
                    text = req.get("text")
                    if not isinstance(text, str):
                        conn.sendall(b'{"error":"bad request"}\n')
                        continue
                    last_used = time.monotonic()
                    conn.sendall((json.dumps({"vector": embed(text)}) + "\n").encode())
                except (OSError, ValueError):
                    continue
    finally:
        server.close()
        try:
            info = read_info(root)
            if info and info.get("pid") == os.getpid():
                info_path(root).unlink()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Keep the embedding model warm between prompts")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--idle-min", type=float, default=DEFAULT_IDLE_MIN)
    args = parser.parse_args()

    import yaml
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from semantic_search import embed_query
    with open(pathlib.Path(args.root) / "config.yaml") as f:
        config = yaml.safe_load(f)
    embed_query(config, "warm up")          # load the model before announcing we're ready
    serve(args.root, lambda text: embed_query(config, text).tolist(), idle_min=args.idle_min)


if __name__ == "__main__":
    main()
