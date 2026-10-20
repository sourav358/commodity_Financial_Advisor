
import unittest

import commodity_finance_assistance


class TestPackage(unittest.TestCase):
    def test_package_can_be_imported(self):
        self.assertIsNotNone(commodity_finance_assistance)


if __name__ == "__main__":
    unittest.main()