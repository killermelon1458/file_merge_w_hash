#!/usr/bin/env python3
"""
email_debug.py

Small debug script for pythonEmailNotify.py.
- Reads EMAIL_ADDRESS / EMAIL_PASSWORD / MAIN_EMAIL_ADDRESS from env vars
- Attempts to send a test email
- Prints clear errors if anything is missing or Gmail rejects creds
"""

import os
import sys
import traceback

from pythonEmailNotify import EmailSender


def get_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def main() -> int:
    try:
        login = get_env("EMAIL_ADDRESS")
        password = get_env("EMAIL_PASSWORD")
        to_addr = get_env("MAIN_EMAIL_ADDRESS")

        sender = EmailSender(
            smtp_server="smtp.gmail.com",
            port=587,
            login=login,
            password=password,
            default_recipient=to_addr,
        )

        subject = "Backup email debug test"
        body = (
            "This is a test email from email_debug.py.\n\n"
            f"From (login): {login}\n"
            f"To: {to_addr}\n"
        )

        # Send to explicit recipient (overrides default)
        sender.sendEmail(subject=subject, body=body, recipient=to_addr, html=False)

        print("SUCCESS: Test email sent.")
        return 0

    except Exception as e:
        print("FAILED: Could not send test email.")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
