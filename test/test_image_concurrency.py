from __future__ import annotations

import unittest

from services.image_concurrency import ImageConcurrencyGate, ImageConcurrencyLimitExceeded


class ImageConcurrencyGateTests(unittest.TestCase):
    def test_admin_ten_plus_user_twenty_fill_global_thirty(self) -> None:
        gate = ImageConcurrencyGate()
        admin = {"id": "admin", "role": "admin"}
        user = {"id": "user", "role": "user", "image_concurrency_limit": 20}
        leases = [gate.try_acquire(admin, global_limit=30) for _ in range(10)]
        with self.assertRaises(ImageConcurrencyLimitExceeded) as admin_error:
            gate.try_acquire(admin, global_limit=30)
        self.assertEqual(admin_error.exception.scope, "owner")
        self.assertEqual(admin_error.exception.limit, 10)

        leases.extend(gate.try_acquire(user, global_limit=30) for _ in range(20))
        self.assertEqual(gate.snapshot()["active_total"], 30)
        with self.assertRaises(ImageConcurrencyLimitExceeded) as global_error:
            gate.try_acquire(
                {"id": "other", "role": "user", "image_concurrency_limit": 1},
                global_limit=30,
            )
        self.assertEqual(global_error.exception.scope, "global")

        for lease in leases:
            lease.release()
            lease.release()
        self.assertEqual(gate.snapshot()["active_total"], 0)

    def test_user_limit_is_independent_for_each_key(self) -> None:
        gate = ImageConcurrencyGate()
        first = {"id": "first", "role": "user", "image_concurrency_limit": 2}
        second = {"id": "second", "role": "user", "image_concurrency_limit": 2}
        leases = [gate.try_acquire(first, global_limit=30) for _ in range(2)]
        leases.extend(gate.try_acquire(second, global_limit=30) for _ in range(2))
        with self.assertRaises(ImageConcurrencyLimitExceeded) as error:
            gate.try_acquire(first, global_limit=30)
        self.assertEqual(error.exception.scope, "owner")
        for lease in leases:
            lease.release()


if __name__ == "__main__":
    unittest.main()
