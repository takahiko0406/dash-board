"""
Run Conditional RMB Research Dashboard Monitor and send results to Telegram.

Purpose:
- This is NOT the production rebalance model.
- It runs the RMB/China-EM research dashboard monitor.
- It sends dashboard interpretation to Telegram.

Required GitHub Secrets:
- BOT_TOKEN
- CHAT_ID

Expected model file in same folder:
- model_c_plus_conditional_rmb_research_dashboard_monitor_004.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

WORKDIR = Path(__file__).resolve().parent
MODEL_SCRIPT = WORKDIR / "model_c_plus_conditional_rmb_research_dashboard_monitor_004.py"
DASHBOARD_TXT = WORKDIR / "conditional_rmb_activation_test_001_macro_rmb_dashboard_monitor.txt"
LOG_FILE = WORKDIR / "conditional_rmb_dashboard_github_run_log.txt"

TELEGRAM_LIMIT = 3900  # keep below Telegram 4096 char limit


def write_log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def send_telegram_message(text: str) -> None:
    bot_token = os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("CHAT_ID")

    if not bot_token or not chat_id:
        raise RuntimeError("Missing BOT_TOKEN or CHAT_ID environment variable / GitHub Secret.")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urlencode({
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    req = Request(url, data=payload, method="POST")
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            write_log(f"Telegram response HTTP {resp.status}: {body[:300]}")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram HTTPError {e.code}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"Telegram URLError: {e}") from e


def split_message(text: str, limit: int = TELEGRAM_LIMIT):
    chunks = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def run_model() -> subprocess.CompletedProcess:
    if not MODEL_SCRIPT.exists():
        raise FileNotFoundError(f"Model script not found: {MODEL_SCRIPT}")

    write_log(f"Running model: {MODEL_SCRIPT.name}")
    result = subprocess.run(
        [sys.executable, str(MODEL_SCRIPT)],
        cwd=str(WORKDIR),
        text=True,
        capture_output=True,
        timeout=60 * 60,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    write_log(f"Model exit code: {result.returncode}")
    if result.returncode != 0:
        raise RuntimeError(
            "Conditional RMB dashboard model failed.\n\n"
            f"STDOUT:\n{result.stdout[-3000:]}\n\n"
            f"STDERR:\n{result.stderr[-3000:]}"
        )
    return result


def build_telegram_text() -> str:
    if DASHBOARD_TXT.exists():
        dashboard_text = DASHBOARD_TXT.read_text(encoding="utf-8", errors="replace")
    else:
        dashboard_text = "Dashboard TXT file was not created. Check GitHub Actions log."

    header = (
        "🧪 CONDITIONAL RMB RESEARCH DASHBOARD\n"
        "Not production rebalance. Monitor only.\n"
        "RMB can affect the research challenger only when EM/reflation gate activates.\n\n"
    )
    return header + dashboard_text


def main() -> None:
    write_log("Starting Conditional RMB dashboard Telegram runner")
    run_model()

    text = build_telegram_text()
    chunks = split_message(text)
    write_log(f"Sending Telegram dashboard in {len(chunks)} chunk(s)")

    for i, chunk in enumerate(chunks, start=1):
        prefix = f"Part {i}/{len(chunks)}\n" if len(chunks) > 1 else ""
        send_telegram_message(prefix + chunk)
        time.sleep(1)

    write_log("Done")


if __name__ == "__main__":
    main()
