"""
Tests for notifications/notifier.py

Mocks SMTP and file I/O — no real email server required.
Run with: pytest tests/test_notifier.py -v
"""

import smtplib
from unittest.mock import MagicMock, mock_open, patch

import pytest

from notifications.notifier import _log_to_file, send_reservation_notification

SUBJECT = "[Slytherin] Test Notification"
BODY = "Reservation details go here."


# ---------------------------------------------------------------------------
# File-fallback path
# ---------------------------------------------------------------------------

class TestFileFallback:
    def test_falls_back_when_smtp_not_configured(self):
        """When smtp_host is None, _log_to_file must be called (not SMTP)."""
        with patch("notifications.notifier._log_to_file") as mock_log:
            send_reservation_notification(
                subject=SUBJECT,
                body=BODY,
                smtp_host=None,
                smtp_port=465,
                smtp_user=None,
                smtp_password=None,
                admin_email=None,
            )
            mock_log.assert_called_once_with(SUBJECT, BODY)

    def test_log_to_file_writes_subject(self):
        """_log_to_file must write a string that includes the subject."""
        with patch("builtins.open", mock_open()) as m:
            _log_to_file(SUBJECT, BODY)

            written = "".join(call.args[0] for call in m().write.call_args_list)
            assert SUBJECT in written

    def test_log_to_file_writes_body(self):
        """_log_to_file must write a string that includes the body."""
        with patch("builtins.open", mock_open()) as m:
            _log_to_file(SUBJECT, BODY)

            written = "".join(call.args[0] for call in m().write.call_args_list)
            assert BODY in written


# ---------------------------------------------------------------------------
# SMTP path
# ---------------------------------------------------------------------------

class TestSmtpPath:
    def test_smtp_ssl_called_when_configured(self):
        """When all SMTP settings are present, smtplib.SMTP_SSL must be used."""
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            mock_server = mock_ssl.return_value.__enter__.return_value
            mock_ssl.return_value.__exit__ = MagicMock(return_value=False)

            send_reservation_notification(
                subject=SUBJECT,
                body=BODY,
                smtp_host="smtp.example.com",
                smtp_port=465,
                smtp_user="user@example.com",
                smtp_password="secret",
                admin_email="admin@example.com",
            )

            mock_ssl.assert_called_once_with("smtp.example.com", 465)
            mock_server.login.assert_called_once_with("user@example.com", "secret")
            mock_server.sendmail.assert_called_once()

    def test_smtp_failure_falls_back_to_file(self):
        """SMTP connection error must be caught and fall back to file — not raise."""
        with patch("smtplib.SMTP_SSL", side_effect=smtplib.SMTPException("connection refused")):
            with patch("builtins.open", mock_open()):
                # Must not raise
                send_reservation_notification(
                    subject=SUBJECT,
                    body=BODY,
                    smtp_host="smtp.example.com",
                    smtp_port=465,
                    smtp_user="user@example.com",
                    smtp_password="secret",
                    admin_email="admin@example.com",
                )