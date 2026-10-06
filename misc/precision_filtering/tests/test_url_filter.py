import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tldextract import TLDExtract


SCRIPT_PATH = Path(__file__).parents[1] / "run_precision_filtering.py"
REPRODUCER_PATH = Path(__file__).parents[1] / "reproduce_tld_bug.py"


def load_filter_class():
    """Load the filter class without executing the script's SLURM jobs."""
    source = SCRIPT_PATH.read_text()
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Decontaminate"
    )
    namespace = {}
    exec(compile("".join(source.splitlines(keepends=True)[:class_node.end_lineno]), SCRIPT_PATH, "exec"), namespace)
    return namespace["Decontaminate"]


class URLFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("TLDEXTRACT_CACHE", str(Path(__file__).parent / ".tldextract-cache"))
        cls.filter_class = load_filter_class()

    def make_filter(self, lang_code, extensions):
        # Use tldextract's bundled suffix snapshot to keep tests offline and deterministic.
        with patch("tldextract.TLDExtract", lambda: TLDExtract(suffix_list_urls=())):
            return self.filter_class(lang_code, f"{lang_code}_Latn", whitelist_words=extensions)

    def assert_url_match(self, url_filter, url, extension):
        document = SimpleNamespace(metadata={"url": url})
        self.assertTrue(url_filter.url_filter(document))
        self.assertEqual(document.metadata["url_match"], extension)

    def test_tld_matches_root_path_and_subdomain_urls(self):
        for lang_code, extension in [("hrv", ".hr"), ("gsw", ".ch"), ("goh", ".de")]:
            url_filter = self.make_filter(lang_code, [extension])
            for url in [
                f"https://example{extension}",
                f"https://example{extension}/article",
                f"https://sub.example{extension}/article",
                f"https://EXAMPLE{extension.upper()}/article",
                f"https://example{extension}:8443/article",
                f"https://example{extension}./article",
            ]:
                with self.subTest(lang_code=lang_code, url=url):
                    self.assert_url_match(url_filter, url, extension)

    def test_multiple_configured_tlds_are_matched(self):
        url_filter = self.make_filter("dip", [".sd", ".ss"])

        self.assert_url_match(url_filter, "https://example.sd/article", ".sd")
        self.assert_url_match(url_filter, "https://example.ss/article", ".ss")

    def test_tld_text_outside_hostname_does_not_match(self):
        url_filter = self.make_filter("hrv", [".hr"])

        for url in ["https://example.com/archive.hr/article", "https://example.hr.example.com/article"]:
            with self.subTest(url=url):
                self.assertFalse(url_filter.url_filter(SimpleNamespace(metadata={"url": url})))

    def test_reproducer_calls_base_and_current_pipeline(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPRODUCER_PATH),
                "--text",
                (
                    "Ovo je jednostavan tekst na hrvatskom jeziku. Govori o "
                    "svakodnevnom životu ljudi u Zagrebu, njihovom radu, učenju "
                    "i druženju tijekom godine."
                ),
                "--url",
                "https://example.hr/article",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Base pipeline decision:    REJECTED", result.stdout)
        self.assertIn("Current pipeline decision: ACCEPTED", result.stdout)
        self.assertIn("Wordlist match count:      0", result.stdout)
        self.assertIn("Bug reproduced:            YES", result.stdout)


if __name__ == "__main__":
    unittest.main()
