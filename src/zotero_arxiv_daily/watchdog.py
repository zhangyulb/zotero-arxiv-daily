"""Workflow watchdog and failure-notification helpers (stdlib-only for fast CI execution)."""

from __future__ import annotations

import datetime
from email.mime.text import MIMEText
import json
import os
import smtplib
import ssl
import sys
from typing import Any
import urllib.request


def send_alert_email(
    *,
    smtp_server: str,
    smtp_port: int,
    sender: str,
    receiver: str,
    password: str,
    subject: str,
    body: str,
    timeout: int = 30,
) -> None:
    """Send a plain-text alert email using STARTTLS -> SMTP_SSL -> plain SMTP fallback."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    ctx = ssl.create_default_context()
    server = None
    try:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=timeout)
        server.starttls(context=ctx)
    except Exception:
        if server is not None:
            try:
                server.close()
            except Exception:
                pass
        try:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, context=ctx, timeout=timeout)
        except Exception:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=timeout)

    try:
        server.login(sender, password)
        server.send_message(msg)
    finally:
        try:
            server.quit()
        except Exception:
            pass


def evaluate_recent_runs(
    runs: list[dict[str, Any]],
    *,
    now: datetime.datetime | None = None,
    lookback_hours: int = 24,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Return (problem_message_or_none, recent_runs) for workflow runs within lookback_hours."""
    current_time = now or datetime.datetime.now(datetime.timezone.utc)
    cutoff = current_time - datetime.timedelta(hours=lookback_hours)
    recent = [
        r
        for r in runs
        if datetime.datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) >= cutoff
    ]
    if not recent:
        return (
            f"No 'Send emails daily' run was created in the last {lookback_hours} hours "
            "(schedule likely disabled on the fork).",
            recent,
        )
    if all(
        r.get("status") == "completed" and r.get("conclusion") in ("failure", "cancelled", "timed_out")
        for r in recent
    ):
        return (
            f"Recent 'Send emails daily' run(s) in the last {lookback_hours} hours did not succeed.",
            recent,
        )
    return None, recent


def fetch_workflow_runs(repo: str, gh_token: str, workflow_file: str = "main.yml") -> list[dict[str, Any]]:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/runs?per_page=10",
        headers={
            "Authorization": f"token {gh_token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)["workflow_runs"]


def check_daily_run(env: dict[str, str] | None = None, now: datetime.datetime | None = None) -> int:
    environ = env if env is not None else os.environ
    repo = environ["GITHUB_REPOSITORY"]
    runs = fetch_workflow_runs(repo, environ["GH_TOKEN"])
    problem, recent = evaluate_recent_runs(runs, now=now)

    if problem is None:
        latest = recent[0].get("conclusion") or recent[0].get("status")
        print(f"OK: {len(recent)} run(s) in last 24h, latest={latest}")
        return 0

    print(problem)
    run_url = recent[0]["html_url"] if recent else f"https://github.com/{repo}/actions/workflows/main.yml"
    send_alert_email(
        smtp_server=environ["SMTP_SERVER"],
        smtp_port=int(environ["SMTP_PORT"]),
        sender=environ["SENDER"],
        receiver=environ["RECEIVER"],
        password=environ["SENDER_PASSWORD"],
        subject="[zotero-arxiv-daily] daily schedule missed or failed",
        body=f"{problem}\n\nRun: {run_url}",
    )
    print("Watchdog alert sent to", environ["RECEIVER"])
    return 1


def report_workflow_failure(env: dict[str, str] | None = None) -> int:
    environ = env if env is not None else os.environ
    run_url = f"https://github.com/{environ['GITHUB_REPOSITORY']}/actions/runs/{environ['GITHUB_RUN_ID']}"
    reason = environ.get("FAILURE_REASON", "unknown")
    body = f"The 'Send emails daily' workflow run did not complete ({reason}).\n\nRun: {run_url}"
    send_alert_email(
        smtp_server=environ["SMTP_SERVER"],
        smtp_port=int(environ["SMTP_PORT"]),
        sender=environ["SENDER"],
        receiver=environ["RECEIVER"],
        password=environ["SENDER_PASSWORD"],
        subject="[zotero-arxiv-daily] daily run did not complete",
        body=body,
    )
    print("Failure notification sent to", environ["RECEIVER"])
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    cmd = args[0] if args else "check"
    if cmd == "check":
        return check_daily_run()
    if cmd == "report-failure":
        return report_workflow_failure()
    raise ValueError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
