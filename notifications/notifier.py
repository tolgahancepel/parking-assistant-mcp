"""
Reservation notification dispatcher.

Primary channel: SMTP email (if SMTP_HOST is configured in .env).
Fallback: append to admin_notifications.log (useful for local development).

This module is stateless — it only sends; it does not track state.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def send_reservation_notification(
    subject: str,
    body: str,
    smtp_host: str | None,
    smtp_port: int,
    smtp_user: str | None,
    smtp_password: str | None,
    admin_email: str | None,
) -> None:
    """
    Send the notification email to the admin.
    Falls back to file logging when SMTP is not configured.
    """
    if smtp_host and admin_email and smtp_user and smtp_password:
        _send_email(subject, body, smtp_host, smtp_port, smtp_user, smtp_password, admin_email)
    else:
        _log_to_file(subject, body)


def _send_email(
    subject: str,
    body: str,
    host: str,
    port: int,
    user: str,
    password: str,
    to_addr: str,
) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL(host, port) as server:
            server.login(user, password)
            server.sendmail(user, to_addr, msg.as_string())
        logger.info("Notification email sent to %s", to_addr)
    except Exception as exc:
        logger.error("Failed to send email: %s. Falling back to file log.", exc)
        _log_to_file(subject, body)


def _log_to_file(subject: str, body: str) -> None:
    """Write notification to admin_notifications.log as a fallback."""
    log_path = "admin_notifications.log"
    separator = "=" * 60
    entry = f"\n{separator}\nSUBJECT: {subject}\n{separator}\n{body}\n"
    with open(log_path, "a") as f:
        f.write(entry)
    logger.info("Notification written to %s (SMTP not configured).", log_path)
