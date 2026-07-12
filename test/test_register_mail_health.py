from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.register import mail_provider
from services.register.mail_health import MailHealthTracker


class RegisterMailHealthTests(unittest.TestCase):
    def test_registration_disallowed_rate_disables_provider_domain_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "mail-health.json"
            tracker = MailHealthTracker(
                path,
                min_risk_attempts=5,
                min_success_rate=0.5,
                cooldown_seconds=3600,
            )
            mailbox = {
                "provider": "temp_mail",
                "provider_ref": "temp_mail#1",
                "address": "private-local-part@bad.example",
            }

            for _ in range(4):
                tracker.record(
                    mailbox,
                    success=False,
                    error="registration_disallowed: cannot create account",
                )
            tracker.record(mailbox, success=True)

            self.assertTrue(tracker.is_disabled("temp_mail#1", "bad.example"))
            [item] = tracker.snapshot()
            self.assertEqual(item["provider"], "temp_mail")
            self.assertEqual(item["provider_ref"], "temp_mail#1")
            self.assertEqual(item["domain"], "bad.example")
            self.assertEqual(item["success"], 1)
            self.assertEqual(item["registration_disallowed"], 4)
            self.assertEqual(item["risk_success_rate"], 20.0)
            self.assertTrue(item["disabled"])
            self.assertNotIn("private-local-part", path.read_text(encoding="utf-8"))

            reloaded = MailHealthTracker(
                path,
                min_risk_attempts=5,
                min_success_rate=0.5,
                cooldown_seconds=3600,
            )
            self.assertTrue(reloaded.is_disabled("temp_mail#1", "bad.example"))

    def test_network_failures_do_not_disable_domain_as_risk_rejections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = MailHealthTracker(
                Path(tmp_dir) / "mail-health.json",
                min_risk_attempts=3,
                min_success_rate=0.5,
                cooldown_seconds=3600,
            )
            mailbox = {
                "provider": "temp_mail",
                "provider_ref": "temp_mail#1",
                "address": "user@network.example",
            }

            for _ in range(10):
                tracker.record(mailbox, success=False, error="request timeout")

            self.assertFalse(tracker.is_disabled("temp_mail#1", "network.example"))
            [item] = tracker.snapshot()
            self.assertEqual(item["fail"], 10)
            self.assertEqual(item["registration_disallowed"], 0)
            self.assertEqual(item["risk_attempts"], 0)

    def test_domain_rotation_skips_temporarily_disabled_domain(self):
        with patch.object(
            mail_provider.mail_health_tracker,
            "is_disabled",
            side_effect=lambda provider_ref, domain: domain == "bad.example",
        ):
            selected = mail_provider._next_domain(
                ["bad.example", "good.example"],
                provider_ref="temp_mail#1",
            )

        self.assertEqual(selected, "good.example")


if __name__ == "__main__":
    unittest.main()
