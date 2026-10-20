"""Tests for zotero_arxiv_daily.watchdog."""

import datetime
import smtplib
import urllib.request

import pytest

from zotero_arxiv_daily import watchdog


def _make_env() -> dict[str, str]:
    return {
        "GITHUB_REPOSITORY": "zhangyulb/zotero-arxiv-daily",
        "GITHUB_RUN_ID": "38659690411",
        "GH_TOKEN": "fake-token",
        "SMTP_SERVER": "smtp.example.com",
        "SMTP_PORT": "587",
        "SENDER": "sender@example.com",
        "RECEIVER": "receiver@example.com",
        "SENDER_PASSWORD": "secret",
    }


def test_send_alert_email_starttls_success(monkeypatch):
    sent = []

    class StubSMTP:
        def __init__(self, host, port, timeout=30):
            self.host = host
            self.port = port

        def starttls(self, context=None):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg):
            sent.append(msg)

        def quit(self):
            pass

    monkeypatch.setattr(smtplib, "SMTP", StubSMTP)
    watchdog.send_alert_email(
        smtp_server="smtp.example.com",
        smtp_port=587,
        sender="sender@example.com",
        receiver="receiver@example.com",
        password="secret",
        subject="Alert",
        body="Body text",
    )
    assert len(sent) == 1
    assert sent[0]["Subject"] == "Alert"


def test_send_alert_email_falls_back_to_ssl_and_closes_failed_starttls_socket(monkeypatch):
    sent = []
    closed = []

    class StubSMTPStarttlsFails:
        def __init__(self, host, port, timeout=30):
            pass

        def starttls(self, context=None):
            raise OSError("STARTTLS not supported")

        def close(self):
            closed.append(True)

    class StubSMTPSSL:
        def __init__(self, host, port, context=None, timeout=30):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg):
            sent.append(msg)

        def quit(self):
            pass

    monkeypatch.setattr(smtplib, "SMTP", StubSMTPStarttlsFails)
    monkeypatch.setattr(smtplib, "SMTP_SSL", StubSMTPSSL)
    watchdog.send_alert_email(
        smtp_server="smtp.example.com",
        smtp_port=465,
        sender="sender@example.com",
        receiver="receiver@example.com",
        password="secret",
        subject="SSL Alert",
        body="Body text",
    )
    assert closed == [True]
    assert len(sent) == 1


def test_send_alert_email_falls_back_to_plain_when_ssl_raises_wrong_version(monkeypatch):
    sent = []
    smtp_calls = []

    class StubSMTP:
        def __init__(self, host, port, timeout=30):
            smtp_calls.append((host, port))

        def starttls(self, context=None):
            raise OSError("STARTTLS failed")

        def close(self):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg):
            sent.append(msg)

        def quit(self):
            pass

    class StubSMTPSSLFails:
        def __init__(self, *args, **kwargs):
            raise OSError("[SSL: WRONG_VERSION_NUMBER]")

    monkeypatch.setattr(smtplib, "SMTP", StubSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", StubSMTPSSLFails)
    watchdog.send_alert_email(
        smtp_server="smtp.example.com",
        smtp_port=587,
        sender="sender@example.com",
        receiver="receiver@example.com",
        password="secret",
        subject="Plain Alert",
        body="Body text",
    )
    assert len(smtp_calls) == 2
    assert len(sent) == 1


def test_evaluate_recent_runs_accepts_previous_utc_day_within_24h():
    now = datetime.datetime(2026, 10, 15, 3, 0, 0, tzinfo=datetime.timezone.utc)
    runs = [
        {
            "created_at": "2026-10-14T22:00:05Z",
            "status": "completed",
            "conclusion": "success",
            "html_url": "https://github.com/example/repo/actions/runs/1",
        }
    ]
    problem, recent = watchdog.evaluate_recent_runs(runs, now=now)
    assert problem is None
    assert len(recent) == 1


def test_evaluate_recent_runs_accepts_in_progress_run():
    now = datetime.datetime(2026, 10, 15, 3, 0, 0, tzinfo=datetime.timezone.utc)
    runs = [
        {
            "created_at": "2026-10-15T02:00:00Z",
            "status": "in_progress",
            "conclusion": None,
            "html_url": "https://github.com/example/repo/actions/runs/2",
        }
    ]
    problem, recent = watchdog.evaluate_recent_runs(runs, now=now)
    assert problem is None
    assert len(recent) == 1


def test_evaluate_recent_runs_flags_missing_or_failed_runs():
    now = datetime.datetime(2026, 10, 15, 3, 0, 0, tzinfo=datetime.timezone.utc)
    problem_empty, _ = watchdog.evaluate_recent_runs([], now=now)
    assert problem_empty is not None and "No 'Send emails daily' run" in problem_empty

    failed_runs = [
        {
            "created_at": "2026-10-15T00:22:50Z",
            "status": "completed",
            "conclusion": "cancelled",
            "html_url": "https://github.com/example/repo/actions/runs/3",
        }
    ]
    problem_failed, _ = watchdog.evaluate_recent_runs(failed_runs, now=now)
    assert problem_failed is not None and "did not succeed" in problem_failed


def test_check_daily_run_and_report_workflow_failure(monkeypatch):
    alerts = []
    now = datetime.datetime(2026, 10, 15, 3, 0, 0, tzinfo=datetime.timezone.utc)

    monkeypatch.setattr(
        watchdog,
        "send_alert_email",
        lambda **kwargs: alerts.append(kwargs),
    )
    monkeypatch.setattr(
        watchdog,
        "fetch_workflow_runs",
        lambda repo, token: [
            {
                "created_at": "2026-10-15T00:22:50Z",
                "status": "completed",
                "conclusion": "cancelled",
                "html_url": "https://github.com/zhangyulb/zotero-arxiv-daily/actions/runs/38659690411",
            }
        ],
    )

    rc_fail = watchdog.check_daily_run(env=_make_env(), now=now)
    assert rc_fail == 1
    assert len(alerts) == 1

    monkeypatch.setattr(
        watchdog,
        "fetch_workflow_runs",
        lambda repo, token: [
            {
                "created_at": "2026-10-14T22:00:05Z",
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://github.com/zhangyulb/zotero-arxiv-daily/actions/runs/1",
            }
        ],
    )
    rc_ok = watchdog.check_daily_run(env=_make_env(), now=now)
    assert rc_ok == 0

    rc_report = watchdog.report_workflow_failure(env=_make_env())
    assert rc_report == 0
    assert len(alerts) == 2


def test_fetch_workflow_runs_and_main_cli(monkeypatch):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"workflow_runs": [{"id": 1}]}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda req: FakeResp())
    runs = watchdog.fetch_workflow_runs("owner/repo", "tok")
    assert runs == [{"id": 1}]

    monkeypatch.setattr(watchdog, "check_daily_run", lambda: 0)
    monkeypatch.setattr(watchdog, "report_workflow_failure", lambda: 0)
    assert watchdog.main(["check"]) == 0
    assert watchdog.main(["report-failure"]) == 0
    with pytest.raises(ValueError, match="Unknown command"):
        watchdog.main(["invalid"])
