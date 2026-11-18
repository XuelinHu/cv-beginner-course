"""Manage this project's detached static server on Linux, using only Python stdlib."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
STATE = RUNTIME / "site-service.json"
PID_FILE = RUNTIME / "site.pid"
SERVER = ROOT / "scripts/serve_site.py"
APP = "industrial-vision-lab-static"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def write_state(state: dict) -> None:
    RUNTIME.mkdir(exist_ok=True)
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)
    PID_FILE.write_text(str(state.get("pid") or 0) + "\n", encoding="ascii")


def commandline(pid: int) -> list[str]:
    try:
        return [part.decode("utf-8") for part in (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0") if part]
    except (OSError, UnicodeDecodeError):
        return []


def alive(pid: int | None) -> bool:
    if not pid or pid <= 1:
        return False
    try:
        status = (Path("/proc") / str(pid) / "stat").read_text()
        return status.rsplit(") ", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def flag_value(args: list[str], name: str) -> str | None:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def owns_process(state: dict) -> bool:
    pid = state.get("pid")
    if not alive(pid):
        return False
    args = commandline(pid)
    return (str(SERVER) in args and flag_value(args, "--instance-id") == state.get("instance_id")
            and flag_value(args, "--port") == str(state.get("port")))


def health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
            data = json.load(response)
        return data if data.get("app") == APP else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def status(state: dict) -> dict:
    if not state.get("pid") or not alive(state.get("pid")):
        return {"status": "stopped", "port": state.get("port", 4034), "pid": None}
    if not owns_process(state):
        return {"status": "unverified_process", "port": state.get("port"), "pid": state.get("pid")}
    data = health(state["port"])
    if not data or data.get("pid") != state["pid"] or data.get("instance_id") != state.get("instance_id"):
        return {"status": "unhealthy", "port": state["port"], "pid": state["pid"]}
    return {"status": "running", "port": state["port"], "pid": state["pid"],
            "revision": data.get("revision"), "health": data}


def terminate(pid: int) -> None:
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 8
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if alive(pid):
        raise RuntimeError("Server has not exited; state is retained and a second server will not be started")


def stop(state: dict) -> dict:
    pid = state.get("pid")
    if alive(pid):
        if not owns_process(state):
            raise RuntimeError("PID does not identify this project's recorded server; refusing to stop it")
        terminate(pid)
    if state:
        state = {**state, "last_pid": pid, "pid": None, "status": "stopped", "stopped_at": now()}
        write_state(state)
    return {"status": "stopped", "port": state.get("port", 4034), "pid": None}


def adopt_legacy(port: int) -> None:
    """Only the exact old http.server for this project's site may be replaced."""
    if not PID_FILE.is_file():
        return
    raw = PID_FILE.read_text(encoding="ascii").strip()
    pid = int(raw) if raw.isdecimal() else 0
    if not alive(pid):
        return
    args = commandline(pid)
    try:
        module = args.index("-m")
        exact_module = args[module + 1:module + 3] == ["http.server", str(port)]
    except ValueError:
        exact_module = False
    directory = flag_value(args, "--directory")
    if not exact_module or directory != str(ROOT / "site"):
        raise RuntimeError("Legacy PID is not the exact static server for this site's directory and port")
    terminate(pid)


def start(state: dict, *, bind: str, port: int) -> dict:
    current = status(state)
    if current["status"] == "running":
        if state.get("port") != port or state.get("bind") != bind:
            raise RuntimeError("Server already runs with other settings; use restart to change them")
        return current
    if alive(state.get("pid")):
        raise RuntimeError("A recorded process is still alive; stop or inspect it before starting another")
    if listening(port):
        raise RuntimeError("Port is already in use. Use --adopt-legacy only for the original recorded static server")
    RUNTIME.mkdir(exist_ok=True)
    instance_id = uuid.uuid4().hex
    with (RUNTIME / "site.log").open("ab") as log:
        process = subprocess.Popen([sys.executable, "-u", str(SERVER), "--bind", bind,
                                    "--port", str(port), "--instance-id", instance_id],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True, close_fds=True)
    pending = {"app": APP, "pid": process.pid, "instance_id": instance_id,
               "bind": bind, "port": port, "started_at": now(), "status": "starting"}
    deadline = time.monotonic() + 8
    while process.poll() is None and time.monotonic() < deadline:
        data = health(port)
        if data and data.get("instance_id") == instance_id and data.get("pid") == process.pid:
            write_state({**pending, "status": "running"})
            return status(pending)
        time.sleep(0.15)
    if process.poll() is None:
        terminate(process.pid)
    raise RuntimeError("Static server did not become ready; inspect runtime/site.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "restart", "status"])
    parser.add_argument("--bind", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--adopt-legacy", action="store_true", help="Explicitly replace only the recorded old http.server for this exact site and port")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Detached process management uses Linux /proc. For Windows foreground preview, run scripts/serve_site.py")
    state = read_state()
    port = args.port if args.port is not None else state.get("port", 4034)
    bind = args.bind if args.bind is not None else state.get("bind", "0.0.0.0")
    if not 1 <= port <= 65535:
        parser.error("Invalid port")
    if args.adopt_legacy and args.action not in ("start", "restart"):
        parser.error("--adopt-legacy is only valid for start/restart")
    try:
        if args.action == "status":
            result = status(state)
        elif args.action == "stop":
            result = stop(state)
        else:
            if args.action == "restart":
                stop(state)
                state = read_state()
            if args.adopt_legacy:
                adopt_legacy(port)
            result = start(state, bind=bind, port=port)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] not in ("running", "stopped"):
            raise SystemExit(2)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
