from __future__ import annotations

import unittest

from api.support import _take_cyclic_batch


class AccountWatcherTests(unittest.TestCase):
    def test_cyclic_batch_rotates_through_large_account_pool(self) -> None:
        items = [f"token-{index}" for index in range(12)]

        first, offset = _take_cyclic_batch(items, 0, 10)
        second, offset = _take_cyclic_batch(items, offset, 10)

        self.assertEqual(first, items[:10])
        self.assertEqual(second, [*items[10:], *items[:8]])
        self.assertEqual(offset, 8)

    def test_cyclic_batch_returns_small_pool_once(self) -> None:
        batch, offset = _take_cyclic_batch(["a", "b"], 0, 10)

        self.assertEqual(batch, ["a", "b"])
        self.assertEqual(offset, 0)


if __name__ == "__main__":
    unittest.main()
