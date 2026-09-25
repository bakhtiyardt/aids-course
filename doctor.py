#!/usr/bin/env python3
"""
AI & DS — Session 1 setup check.

Run it with:      uv run python doctor.py
After homework:   uv run python doctor.py --full

Every check that fails prints the specific fix underneath it. Nothing here
touches the network except the call to your own machine on port 11434.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# tiny output helpers — no dependencies, works on a bare interpreter
# ----------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


GREEN, RED, YELLOW, DIM, BOLD = "32", "31", "33", "2", "1"

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    tag = c("[ok]  ", GREEN) if ok else c("[FAIL]", RED)
    print(f"  {tag} {label:<20s} {detail}")
    if not ok and fix:
        for line in fix.strip().splitlines():
            print(f"         {c('→ ' + line.strip(), YELLOW)}")
    RESULTS.append((label, ok, detail))
    return ok


def warn(label: str, detail: str) -> None:
    print(f"  {c('[note]', YELLOW)} {label:<20s} {detail}")


def rule(char: str = "-", n: int = 60) -> None:
    print("  " + char * n)


# ----------------------------------------------------------------------------
# individual checks
# ----------------------------------------------------------------------------

MIN_PY = (3, 10)
CORE_PACKAGES = ["httpx", "numpy", "pandas", "matplotlib", "jupyterlab"]
FULL_PACKAGES = ["torch", "transformers", "sentence_transformers",
                 "sklearn", "seaborn", "statsmodels", "pydantic"]

IMPORT_NAMES = {"jupyterlab": "jupyterlab", "sklearn": "sklearn",
                "sentence_transformers": "sentence_transformers"}

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST


def check_python() -> None:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PY
    check(
        "python", ok, f"{v.major}.{v.minor}.{v.micro}",
        fix=f"Need {MIN_PY[0]}.{MIN_PY[1]}+. Run:  uv python pin 3.12  then re-run this script.",
    )


def check_project() -> None:
    here = Path.cwd()
    pyproject = here / "pyproject.toml"
    ok = pyproject.exists()
    detail = here.name if ok else f"no pyproject.toml in {here.name}/"
    check(
        "uv project", ok, detail,
        fix="""You are not inside the course project.
               cd into the aids-course folder and run this again.
               If you never created it:  uv init aids-course && cd aids-course""",
    )
    if ok and not (here / "uv.lock").exists():
        warn("uv.lock", "missing — run `uv lock` before the end of the session")


def check_package(name: str, optional: bool = False) -> bool:
    import importlib.util

    mod = IMPORT_NAMES.get(name, name)
    try:
        found = importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        found = False

    if optional and not found:
        warn(name, "not installed (Stage 6 homework)")
        return False

    return check(
        name, found, "installed" if found else "missing",
        fix=f"uv add {name.replace('_', '-')}",
    )


def check_ollama_binary() -> None:
    path = shutil.which("ollama")
    check(
        "ollama binary", path is not None,
        path or "not on PATH",
        fix="""Install it from https://ollama.com/download
               macOS: launch the app once from Applications — that installs the CLI.""",
    )


def http_get_json(url: str, timeout: float = 4.0):
    """Stdlib only, so this works even if httpx failed to install."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode()), None
    except urllib.error.URLError as e:
        return None, getattr(e, "reason", e)
    except Exception as e:  # noqa: BLE001
        return None, e


def http_post_json(url: str, payload: dict, timeout: float = 240.0):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()), None
    except Exception as e:  # noqa: BLE001
        return None, e


def check_server() -> bool:
    data, err = http_get_json(f"{OLLAMA_HOST}/api/tags")
    ok = data is not None
    return check(
        "ollama server", ok,
        f"reachable at {OLLAMA_HOST}" if ok else f"no response ({err})",
        fix="""Ollama is installed but not running.
               macOS: launch Ollama from Applications (look for the menu-bar icon)
               Windows: launch Ollama from the Start menu (check the system tray)
               Linux: run `ollama serve` in a terminal you leave open""",
    )


def check_model() -> str | None:
    data, _ = http_get_json(f"{OLLAMA_HOST}/api/tags")
    if data is None:
        check("model", False, "cannot check — server is down",
              fix="Fix the server check above first.")
        return None

    models = [m.get("name", "") for m in data.get("models", [])]
    if not models:
        check("model", False, "no models pulled",
              fix="""Pull one sized for your RAM:
                     8 GB or less :  ollama pull qwen3:1.7b
                     16 GB+       :  ollama pull phi4-mini""")
        return None

    preferred = os.environ.get("COURSE_MODEL")
    chosen = preferred if preferred in models else models[0]
    check("model", True, chosen + (f"  (+{len(models)-1} more)" if len(models) > 1 else ""))
    return chosen


def check_generation(model: str) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one word: ready"}],
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": 16},
    }
    t0 = time.perf_counter()
    data, err = http_post_json(f"{OLLAMA_HOST}/api/chat", payload)
    wall = time.perf_counter() - t0

    if data is None:
        check("generation", False, f"failed ({err})",
              fix="""The server answered /api/tags but not /api/chat.
                     Try on the command line:  ollama run """ + model + """ "say hi"
                     If that hangs, restart Ollama.""")
        return

    ntok = data.get("eval_count") or 0
    tps = ntok / wall if wall > 0 else 0
    text = data.get("message", {}).get("content", "").strip().replace("\n", " ")
    check("generation", True, f"responded in {wall:.1f}s ({tps:.1f} tok/s)")
    print(f"         {c('model said: ' + text[:60], DIM)}")

    if tps < 4:
        warn("speed", "under 4 tok/s — labs will be slow. Try qwen3:1.7b instead.")

    # This number goes on the shared sheet.
    globals()["_THROUGHPUT"] = tps


# ----------------------------------------------------------------------------
# hardware profile
# ----------------------------------------------------------------------------

def total_ram_gb() -> float | None:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        pass
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / 1e9
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=15)
            return int(out.stdout.strip()) / 1e9
    except Exception:  # noqa: BLE001
        pass
    return None


def hardware_profile() -> dict:
    prof = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": total_ram_gb(),
        "python": platform.python_version(),
        "accelerator": "cpu",
    }
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            prof["accelerator"] = f"cuda:{torch.cuda.get_device_name(0)}"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            prof["accelerator"] = "apple-mps"
    except ImportError:
        if sys.platform == "darwin" and platform.machine() == "arm64":
            prof["accelerator"] = "apple-silicon (assumed; torch not installed)"
    return prof


# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Session 1 setup check")
    ap.add_argument("--full", action="store_true",
                    help="also require the Stage 6 homework packages")
    args = ap.parse_args()

    print()
    rule("=")
    print(f"  {c('AI & DS — Session 1 setup check', BOLD)}")
    rule("=")

    check_python()
    check_project()
    for pkg in CORE_PACKAGES:
        check_package(pkg)

    if args.full:
        print()
        print(f"  {c('Stage 6 packages', BOLD)}")
        for pkg in FULL_PACKAGES:
            check_package(pkg, optional=not args.full)

    print()
    check_ollama_binary()
    model = check_model() if check_server() else None
    if model:
        check_generation(model)

    # ---- summary ----
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed

    rule()
    print(f"  {passed} passed, {failed} failed")
    if failed == 0:
        print(f"  {c('ALL CLEAR — open the notebook.', GREEN)}")
    else:
        print(f"  {c('Fix the arrows above, then run this again.', RED)}")
        print(f"  {c('Stuck more than 3 minutes? Post in the help queue.', DIM)}")
    rule("=")

    # ---- the line you post to the shared sheet ----
    prof = hardware_profile()
    tps = globals().get("_THROUGHPUT")
    ram = f"{prof['ram_gb']:.0f}GB" if prof["ram_gb"] else "?"
    print()
    print(f"  {c('POST THIS LINE TO THE SHARED SHEET:', BOLD)}")
    print(f"  {prof['os']} | {prof['arch']} | {prof['cpu_cores']} cores | {ram} RAM | "
          f"{prof['accelerator']} | {f'{tps:.1f} tok/s' if tps else 'no generation'}")
    print()

    Path("setup_profile.json").write_text(
        json.dumps({**prof, "tok_per_s": tps,
                    "checks": {k: ok for k, ok, _ in RESULTS}}, indent=2)
    )
    print(f"  {c('(also saved to setup_profile.json)', DIM)}")
    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
