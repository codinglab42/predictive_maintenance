from __future__ import annotations

import unittest

from scripts.extract_ims import _normalized_member_name


class IMSExtractionTests(unittest.TestCase):
    def test_third_archive_internal_fourth_test_name_is_normalized(self) -> None:
        self.assertEqual(
            _normalized_member_name("3rd_test", "4th_test/txt/2004.03.04.09.27.46"),
            "3rd_test/txt/2004.03.04.09.27.46",
        )

    def test_series_archive_paths_must_match_expected_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected 2nd_test path"):
            _normalized_member_name("2nd_test", "unexpected/file.txt")


if __name__ == "__main__":
    unittest.main()
