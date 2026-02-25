"""
Launch the Local RAG API server and Open WebUI together.

Usage:
    python launch_openwebui_rag.py          Start both services
    python launch_openwebui_rag.py --stop   Stop both services

Prerequisites:
    - Python virtual environment set up (uv venv)
    - Dependencies installed (uv sync --group openwebui)

Environment variables:
    OPEN_WEBUI_PORT  — Open WebUI port (default: 3000)
    RAG_API_PORT     — RAG API port (read from .env or default: 8000)
"""

import atexit
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAG_PID_FILE = SCRIPT_DIR / ".rag-api.pid"
WEBUI_PID_FILE = SCRIPT_DIR / ".open-webui.pid"
OPEN_WEBUI_PORT = os.environ.get("OPEN_WEBUI_PORT", "3000")


def log(msg: str) -> None:
    """Print a prefixed log message."""
    print(f"[launch] {msg}")


def _find_venv_python() -> Path:
    """
    Locate the Python executable inside the virtual environment.

    Returns:
        Path: Path to the venv Python binary.

    Raises:
        SystemExit: If no virtual environment is found.
    """
    # Reason: Path.exists() does not resolve .exe on Windows, so we must
    # check both with and without the extension.
    for candidate in (SCRIPT_DIR / ".venv" / "Scripts" / "python.exe",
                      SCRIPT_DIR / ".venv" / "Scripts" / "python",
                      SCRIPT_DIR / ".venv" / "bin" / "python"):
        if candidate.exists():
            return candidate

    log("ERROR: Virtual environment not found. Run: uv venv --python 3.13")
    sys.exit(1)


def _pid_is_alive(pid: int) -> bool:
    """
    Check whether a process with the given PID is running.

    Args:
        pid (int): Process ID to check.

    Returns:
        bool: True if the process is alive.
    """
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_pid(pid_file: Path) -> int | None:
    """
    Read a PID from a file, returning None if missing or stale.

    Args:
        pid_file (Path): Path to the PID file.

    Returns:
        int | None: The PID if valid and alive, else None.
    """
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if _pid_is_alive(pid) else None


def _kill_pid(pid_file: Path, label: str) -> None:
    """
    Stop a process tracked by a PID file.

    Args:
        pid_file (Path): Path to the PID file.
        label (str): Human-readable service name for log messages.
    """
    if not pid_file.exists():
        log(f"No {label} PID file found.")
        return

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return

    if _pid_is_alive(pid):
        # Reason: On Windows os.kill with SIGTERM doesn't propagate to child
        # processes. Use taskkill /T to kill the entire process tree instead.
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.kill(pid, signal.SIGTERM)
        log(f"{label} stopped (PID {pid}).")
    else:
        log(f"{label} was not running.")

    pid_file.unlink(missing_ok=True)


def stop() -> None:
    """Stop both the RAG API server and Open WebUI."""
    log("Stopping services...")
    _kill_pid(RAG_PID_FILE, "RAG API server")
    _kill_pid(WEBUI_PID_FILE, "Open WebUI")


def start() -> None:
    """Start the RAG API server and Open WebUI as background processes."""
    venv_python = _find_venv_python()

    if shutil.which("open-webui") is None:
        log("ERROR: open-webui is not installed. Run: uv sync --group openwebui")
        sys.exit(1)

    # --- RAG API server ---
    existing = _read_pid(RAG_PID_FILE)
    if existing:
        log(f"RAG API server is already running (PID {existing}).")
    else:
        log("Starting RAG API server...")
        proc = subprocess.Popen(
            [str(venv_python), "run_api.py"],
            cwd=str(SCRIPT_DIR),
        )
        RAG_PID_FILE.write_text(str(proc.pid))
        log(f"RAG API server started (PID {proc.pid}).")

    # --- Open WebUI ---
    existing = _read_pid(WEBUI_PID_FILE)
    if existing:
        log(f"Open WebUI is already running (PID {existing}).")
    else:
        log(f"Starting Open WebUI on port {OPEN_WEBUI_PORT}...")
        # Reason: Open WebUI prints a Unicode banner that fails on Windows
        # cp1252 encoding. PYTHONUTF8=1 forces UTF-8 stdout.
        env = {**os.environ, "WEBUI_AUTH": "false", "PYTHONUTF8": "1"}
        proc = subprocess.Popen(
            ["open-webui", "serve", "--port", OPEN_WEBUI_PORT],
            cwd=str(SCRIPT_DIR),
            env=env,
        )
        WEBUI_PID_FILE.write_text(str(proc.pid))
        log(f"Open WebUI started (PID {proc.pid}).")

    rag_port = os.environ.get("RAG_API_PORT", "8000")

    print()
    log("=====================================")
    log(" Services are running:")
    log(f"   RAG API:    http://localhost:{rag_port}")
    log(f"   Open WebUI: http://localhost:{OPEN_WEBUI_PORT}")
    log("")
    log(" Next steps:")
    log(f"   1. Open http://localhost:{OPEN_WEBUI_PORT} in your browser")
    log("   2. Go to Settings > Connections")
    log("   3. Add OpenAI-compatible connection:")
    log(f"        URL: http://localhost:{rag_port}/v1")
    log("        Key: (leave blank or enter your RAG_API_KEY)")
    log("   4. Select the 'local-rag' model in the chat")
    log("=====================================")
    print()
    log("To stop both services: python launch_openwebui_rag.py --stop")

    # Keep the launcher alive so Ctrl+C can clean up both services
    atexit.register(stop)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()
        log("Caught Ctrl+C.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--stop", "-s", "stop"):
        stop()
    else:
        start()
